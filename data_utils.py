import json
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

class RvlnLoRADataset(Dataset):
    def __init__(self, data_path, processor, tokenizer, image_root=".", history_len=4, current_len=1, query_tokens=32):
        print(f"正在加载 JSON文件: {data_path}")
        with open(data_path, "r", encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_root = image_root
        
        self.history_len = history_len
        self.current_len = current_len
        self.query_tokens = query_tokens
        
        # 总共需要的图片数量 (4 + 1 = 5)
        self.total_len = history_len + current_len
        
        # 定义特殊 Token
        self.hist_token = "<history>"
        self.curr_token = "<current>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # ================= 1. 图片路径对齐 =================
        def normalize_paths(path_list):
            if len(path_list) >= self.total_len:
                return path_list[-self.total_len:]
            else:
                pad_num = self.total_len - len(path_list)
                return [None] * pad_num + path_list

        rgb_paths = normalize_paths(item.get("images", []))
        depth_paths = normalize_paths(item.get("depth_images", []))

        # ================= 2. 图片加载 =================
        def load_image(path):
            if path is None:
                return Image.new('RGB', (224, 224), (0, 0, 0))
            full_path = os.path.join(self.image_root, path)
            try:
                return Image.open(full_path).convert("RGB")
            except Exception as e:
                return Image.new('RGB', (224, 224), (0, 0, 0))

        rgb_images = [load_image(p) for p in rgb_paths]
        depth_images = [load_image(p) for p in depth_paths]

        # ================= 3. 生成 Q-Former 输入 =================
        human_input = item["conversations"][0]["value"]
        
        inputs = self.processor(
            images=rgb_images,
            text=[human_input], 
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        depth_inputs = self.processor(images=depth_images, return_tensors="pt")

        # ================= 4. LLM 输入处理 (生成式模式) =================
        # Token 扩展：为视觉特征预留位置
        hist_replacement = self.hist_token * (self.history_len * self.query_tokens)
        curr_replacement = self.curr_token * (self.current_len * self.query_tokens)
        
        expanded_human_input = human_input.replace(
            self.hist_token, hist_replacement
        ).replace(
            self.curr_token, curr_replacement
        )
        
        # 构造 LLM Prompt
        llm_prompt = f"USER: {expanded_human_input} ASSISTANT:"
        
        # 获取原始文本回复 (例如: "{'Route': 8}")
        # 不再提取数字，而是直接学习生成这段文本
        gpt_response = item["conversations"][1]["value"]

        return {
            "pixel_values": inputs.pixel_values,         # [N, 3, H, W]
            "depth_pixel_values": depth_inputs.pixel_values, # [N, 3, H, W]
            "qformer_input_ids": inputs.qformer_input_ids[0],
            "qformer_attention_mask": inputs.qformer_attention_mask[0],
            "llm_prompt": llm_prompt,     # 提问部分
            "llm_answer": gpt_response    # 回答部分
        }

class DataCollatorForRvln:
    def __init__(self, processor, tokenizer, qformer_tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer
        self.qformer_tokenizer = qformer_tokenizer
        
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        # 1. 堆叠图像
        pixel_values_rgb = torch.stack([item["pixel_values"] for item in batch])
        pixel_values_depth = torch.stack([item["depth_pixel_values"] for item in batch])
        
        # 2. 堆叠 Q-Former Input
        qformer_input_ids = torch.stack([item["qformer_input_ids"] for item in batch])
        qformer_attention_mask = torch.stack([item["qformer_attention_mask"] for item in batch])
        
        # ================= 3. 处理 LLM 输入 (拼接 Prompt + Answer) =================
        input_ids_list = []
        labels_list = []
        
        for item in batch:
            prompt = item["llm_prompt"]
            answer = item["llm_answer"]
            
            # 分别 Tokenize 提示词和答案
            # add_special_tokens=False 是为了精准控制 Mask 边界
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
            
            # 拼接: [BOS] + Prompt + Answer + [EOS]
            # 注意：某些 Tokenizer 需要手动加 BOS，有些不需要，视具体模型而定
            # 这里演示最通用的拼接方式
            
            input_ids = prompt_ids + answer_ids + [self.tokenizer.eos_token_id]
            
            # 构建 Labels: Prompt 部分设为 -100 (不计算 Loss)，Answer 部分保留
            labels = [-100] * len(prompt_ids) + answer_ids + [self.tokenizer.eos_token_id]
            
            # 转换为 Tensor
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
        
        # 4. 动态 Padding
        # input_ids 使用 pad_token_id 填充
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        
        # labels 使用 -100 填充 (CrossEntropyLoss 的 ignore_index)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        
        # 5. 生成 Attention Mask (非 padding 部分为 1)
        attention_mask = input_ids_padded.ne(self.pad_token_id).long()

        return {
            "pixel_values": pixel_values_rgb,
            "depth_pixel_values": pixel_values_depth,
            "qformer_input_ids": qformer_input_ids,
            "qformer_attention_mask": qformer_attention_mask,
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded  # 标准的 Causal LM labels
        }