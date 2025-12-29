import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class InstructBlipLoRADataset(Dataset):
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
        # 逻辑：不足 5 张前面补 None，超过 5 张取最后 5 张
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
                # print(f"Load Error: {full_path}")
                return Image.new('RGB', (224, 224), (0, 0, 0))

        # 加载并转换为 PIL List
        rgb_images = [load_image(p) for p in rgb_paths]
        depth_images = [load_image(p) for p in depth_paths]

        # 使用 Processor 转为 Tensor [5, 3, 224, 224]
        # 注意：return_tensors="pt" 会返回 Batch 维度 [1, 5, 3, H, W] 或 [5, 3, H, W]，取决于 processor 实现
        # 这里我们假设 processor 输出 [N_images, 3, H, W]
        pixel_values_rgb = self.processor(images=rgb_images, return_tensors="pt").pixel_values
        pixel_values_depth = self.processor(images=depth_images, return_tensors="pt").pixel_values

        # ================= 3. 文本处理 =================
        # 解析 Human 和 GPT
        human_input = item["conversations"][0]["value"]
        gpt_response = item["conversations"][1]["value"]
        
        # --- Q-Former Prompt ---
        # Q-Former 只需要看原始指令，不需要看展开后的特殊 Token
        # 它可以帮助从图片中提取与文本相关的特征
        qformer_prompt = human_input

        # --- LLM Full Text (Token 扩展) ---
        # 这里的逻辑非常关键：
        # 我们需要在文本中预留出足够的 "坑位" 给视觉特征。
        # 1 个 <history> 对应 4张图 * 32个Token = 128 个坑位
        # 1 个 <current> 对应 1张图 * 32个Token = 32 个坑位
        
        # 构造重复字符串
        hist_replacement = self.hist_token * (self.history_len * self.query_tokens)
        curr_replacement = self.curr_token * (self.current_len * self.query_tokens)
        
        # 替换 Prompt 中的标签
        expanded_human_input = human_input.replace(
            self.hist_token, hist_replacement
        ).replace(
            self.curr_token, curr_replacement
        )
        
        # 构造符合 Vicuna/LLaMA 格式的对话
        # 格式: USER: ... ASSISTANT: ...
        llm_prompt = f"USER: {expanded_human_input} ASSISTANT:"
        llm_full_text = f"{llm_prompt}{gpt_response}</s>"

        return {
            "pixel_values_rgb": pixel_values_rgb,       # Tensor
            "pixel_values_depth": pixel_values_depth,   # Tensor
            "qformer_prompt": qformer_prompt,           # String (for Collator to tokenize)
            "llm_prompt": llm_prompt,                   # String (for calculating mask length)
            "llm_full_text": llm_full_text              # String (Expanded, for Collator)
        }

class DataCollatorForInstructBlip:
    def __init__(self, processor, tokenizer, qformer_tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer
        self.qformer_tokenizer = qformer_tokenizer
        
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.ignore_index = -100

    def __call__(self, batch):
        # 1. 堆叠图像 [Batch, 5, 3, 224, 224]
        pixel_values_rgb = torch.stack([item["pixel_values_rgb"] for item in batch])
        pixel_values_depth = torch.stack([item["pixel_values_depth"] for item in batch])
        
        # 2. Tokenize Q-Former Input (Batch 处理)
        qformer_prompts = [item["qformer_prompt"] for item in batch]
        qformer_inputs = self.qformer_tokenizer(
            qformer_prompts, 
            padding=True, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # 3. Tokenize LLM Input (Batch 处理)
        # 这里处理的是已经扩展过 Token 的长文本
        llm_full_texts = [item["llm_full_text"] for item in batch]
        llm_inputs = self.tokenizer(
            llm_full_texts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048, # 因为展开了 Token，长度会增加，建议调大 max_length
            add_special_tokens=False # 我们已经在 text 里加了 USER/ASSISTANT/EOS
        )
        
        input_ids = llm_inputs.input_ids
        attention_mask = llm_inputs.attention_mask
        labels = input_ids.clone()
        
        # 4. Mask User Prompt (只计算 GPT 回复的 Loss)
        for i, item in enumerate(batch):
            prompt = item["llm_prompt"]
            # 对 Prompt 单独编码以获取长度
            prompt_ids = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                add_special_tokens=False,
                truncation=True, 
                max_length=2048
            ).input_ids[0]
            
            prompt_len = len(prompt_ids)
            
            # 确保不越界
            prompt_len = min(prompt_len, labels.size(1))
            
            # Mask 掉 Prompt
            labels[i, :prompt_len] = self.ignore_index
            
            # Mask 掉 Padding
            labels[i, input_ids[i] == self.pad_token_id] = self.ignore_index

        return {
            "pixel_values_rgb": pixel_values_rgb,       # [B, 5, 3, 224, 224]
            "pixel_values_depth": pixel_values_depth,   # [B, 5, 3, 224, 224]
            "qformer_input_ids": qformer_inputs.input_ids,
            "qformer_attention_mask": qformer_inputs.attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }