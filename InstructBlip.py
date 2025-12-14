import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor, InstructBlipConfig, BertTokenizer


# ==============================================================================
# 1. 模型定义 (InstructBlipMultiTask)
# ==============================================================================
class InstructBlipMultiTask(InstructBlipForConditionalGeneration):
    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)
        
        # 定义 ITM Head
        # InstructBlip Q-Former 输出 hidden_size 默认 768
        self.qformer_hidden_size = config.qformer_config.hidden_size
        
        # 二分类头: [Batch, Hidden] -> [Batch, 2]
        self.itm_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.qformer_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2) 
        )
        
        # 初始化 Head 权重
        self._init_weights(self.itm_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward_itm(self, pixel_values, input_ids, attention_mask):
            """
            专门用于 ITM 训练的前向传播
            """
            # 1. Vision Encoder 提取图像特征
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
            image_embeds = vision_outputs.last_hidden_state
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

            # 2. 准备 Q-Former Query Tokens
            # query_tokens shape: [1, 32, 768] -> [batch, 32, 768]
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            # === 【关键修复】扩展 Attention Mask ===
            # Q-Former 内部会将 query_tokens (32个) 和 input_ids (文本) 拼接
            # 因此我们需要手动把 attention_mask 也扩展
            # 新 mask = [ones(32), text_attention_mask]
            
            batch_size = input_ids.shape[0]
            # 创建 Query 部分的 Mask (全为1，表示可见)
            query_attention_mask = torch.ones(
                (batch_size, query_tokens.shape[1]), # [batch, 32]
                dtype=torch.long, 
                device=input_ids.device
            )
            # 拼接：先 queries, 后 text
            qformer_attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)

            # 3. Q-Former 交互
            query_outputs = self.qformer(
                input_ids=input_ids,
                attention_mask=qformer_attention_mask, # <--- 传入扩展后的 mask
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            
            # [batch, 32, 768]
            qformer_features = query_outputs.last_hidden_state
            
            # 4. Pooling (平均池化)
            pooled_features = torch.mean(qformer_features, dim=1)
            head_dtype = self.itm_head[1].weight.dtype 
            pooled_features = pooled_features.to(head_dtype)
            # 5. Classification
            itm_logits = self.itm_head(pooled_features)
            
            return itm_logits