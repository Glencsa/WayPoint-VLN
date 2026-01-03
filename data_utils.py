import json
import os
import re
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

        # ================= 3. 生成 Q-Former 输入 (修复 inputs 未定义问题) =================
        human_input = item["conversations"][0]["value"]
        
        # Processor 同时处理图片和文本，生成 Q-Former 需要的所有特征
        # 这里 text 参数用于生成 qformer_input_ids
        inputs = self.processor(
            images=rgb_images,
            text=[human_input], 
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        # 单独处理深度图
        depth_inputs = self.processor(images=depth_images, return_tensors="pt")

        # ================= 4. LLM 输入处理 (分类模式) =================
        # Token 扩展：为视觉特征预留位置
        hist_replacement = self.hist_token * (self.history_len * self.query_tokens)
        curr_replacement = self.curr_token * (self.current_len * self.query_tokens)
        
        expanded_human_input = human_input.replace(
            self.hist_token, hist_replacement
        ).replace(
            self.curr_token, curr_replacement
        )
        
        # 构造 LLM Prompt (注意：分类任务不需要把 GPT 回复拼接到这里)
        llm_prompt = f"USER: {expanded_human_input} ASSISTANT:"

        # ================= 5. 提取分类标签 (Label Parsing) =================
        gpt_response = item["conversations"][1]["value"]
        # 目标是从 "{'Route': 8}" 中提取 8
        try:
            # 正则提取数字 (-1 到 8)
            match = re.search(r"(-?\d+)", gpt_response)
            if match:
                route_val = int(match.group(1))
            else:
                # 没找到数字，默认设为 -1 (Stop) 对应的索引
                route_val = -1
        except:
            route_val = -1
            
        # 映射逻辑: Route -1 -> Class 0, Route 0 -> Class 1, ..., Route 8 -> Class 9
        class_label = route_val + 1 
        
        # 安全限制，防止越界 (0-9)
        class_label = max(0, min(9, class_label))

        return {
            "pixel_values": inputs.pixel_values,         # [N, 3, H, W]
            "depth_pixel_values": depth_inputs.pixel_values, # [N, 3, H, W]
            "qformer_input_ids": inputs.qformer_input_ids[0],
            "qformer_attention_mask": inputs.qformer_attention_mask[0],
            "llm_prompt": llm_prompt,                        # 字符串，交给 Collator 去 Tokenize
            "class_labels": class_label                      # 整数
        }

class DataCollatorForRvln:
    def __init__(self, processor, tokenizer, qformer_tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer
        self.qformer_tokenizer = qformer_tokenizer
        
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        # 1. 堆叠图像 [Batch, 5, 3, 224, 224]
        pixel_values_rgb = torch.stack([item["pixel_values_rgb"] for item in batch])
        pixel_values_depth = torch.stack([item["pixel_values_depth"] for item in batch])
        
        # 2. 堆叠 Q-Former Input (Dataset 里已经做了 Padding，直接 Stack)
        qformer_input_ids = torch.stack([item["qformer_input_ids"] for item in batch])
        qformer_attention_mask = torch.stack([item["qformer_attention_mask"] for item in batch])
        
        # 3. 处理 LLM 输入 (Tokenize & Dynamic Padding)
        llm_prompts = [item["llm_prompt"] for item in batch]
        
        llm_inputs = self.tokenizer(
            llm_prompts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048, # 预留给 visual tokens
            add_special_tokens=False # Prompt 里已经手动加了 USER/ASSISTANT
        )
        
        input_ids = llm_inputs.input_ids
        attention_mask = llm_inputs.attention_mask
        
        # 4. 处理分类标签
        class_labels = torch.tensor([item["class_labels"] for item in batch], dtype=torch.long)

        return {
            "pixel_values": pixel_values_rgb,            # 适配模型参数名
            "depth_pixel_values": pixel_values_depth,    # 适配模型参数名
            "qformer_input_ids": qformer_input_ids,
            "qformer_attention_mask": qformer_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "class_labels": class_labels                 # 传入 forward 计算 CrossEntropyLoss
        }