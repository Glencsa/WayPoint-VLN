import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    InstructBlipForConditionalGeneration, 
    InstructBlipConfig, 
    AutoModelForDepthEstimation
)



class DepthCrossAttentionFusion(nn.Module):
    def __init__(self, rgb_dim, depth_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 1. 为了让 Depth 能和 RGB 计算 Attention，先把 Depth 投影到 RGB 维度
        self.depth_proj = nn.Linear(depth_dim, rgb_dim)
        
        # 2. LayerNorm (Pre-Norm 结构)
        self.norm_rgb = nn.LayerNorm(rgb_dim)
        self.norm_depth = nn.LayerNorm(rgb_dim)
        
        # 3. Cross Attention
        # query = RGB, key/value = Depth
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=rgb_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 4. Feed Forward Network (FFN)
        self.norm_ffn = nn.LayerNorm(rgb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(rgb_dim, rgb_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rgb_dim * 4, rgb_dim)
        )
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        # 初始化 Trick: 
        # 让 Cross Attention 的输出投影层初始值很小，使得初始阶段 RGB 特征主要保留原始语义
        # 避免初始阶段混乱的 Attention 破坏预训练的 RGB 特征
        nn.init.constant_(self.cross_attn.out_proj.weight, 0)
        nn.init.constant_(self.cross_attn.out_proj.bias, 0)

    def forward(self, rgb_embeds, depth_embeds):
        """
        Input:
            rgb_embeds:   [B, N_rgb, C_rgb]   (Query)
            depth_embeds: [B, N_depth, C_depth] (Key, Value) 
            注意：N_rgb 和 N_depth 可以不相等！Cross-Attention 会处理对齐。
        """
        # 1. 投影并归一化 Depth
        # [B, N_depth, C_depth] -> [B, N_depth, C_rgb]
        depth_feat = self.depth_proj(depth_embeds)
        depth_feat = self.norm_depth(depth_feat)
        
        # 2. 归一化 RGB (作为 Query)
        rgb_feat_norm = self.norm_rgb(rgb_embeds)
        
        # 3. Cross Attention 计算
        # Query=RGB, Key=Depth, Value=Depth
        attn_output, _ = self.cross_attn(
            query=rgb_feat_norm,
            key=depth_feat,
            value=depth_feat
        )
        
        # 4. 残差连接 1 (RGB + Attention)
        rgb_fused = rgb_embeds + self.dropout(attn_output)
        
        # 5. FFN + 残差连接 2
        rgb_fused = rgb_fused + self.dropout(self.ffn(self.norm_ffn(rgb_fused)))
        
        return rgb_fused

    

class InstructBlipMultiTask(InstructBlipForConditionalGeneration):
    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)
        
        # ... (Depth Model 加载部分代码保持不变) ...
        depth_model_name = "./Depth-Anything-V2-Small-hf"
        print(f"Loading Depth Backbone: {depth_model_name}...")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name)
        self.depth_backbone = self.depth_model.backbone
        
        if hasattr(self.depth_model, "head"):
            del self.depth_model.head
            
        for param in self.depth_backbone.parameters():
            param.requires_grad = False
            
        # ... (Register Buffers 保持不变) ...
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.rgb_hidden_size = config.vision_config.hidden_size 
        self.depth_hidden_size = self.depth_backbone.config.hidden_size
        
        # [修改点 1]：替换为 Cross Attention 融合模块
        # 注意：InstructBLIP 的 hidden size 较大 (1408)，head 数量选 8 或 16 都可以
        self.visual_fusion = DepthCrossAttentionFusion(
            rgb_dim=self.rgb_hidden_size,    # 1408
            depth_dim=self.depth_hidden_size, # 384
            num_heads=8 # 1408 / 8 = 176 维度每头，整除即可
        )

        self.qformer_hidden_size = config.qformer_config.hidden_size
        self.itm_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.qformer_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2) 
        )
        
        self._init_weights(self.itm_head)
        # 融合模块内部已经有 init 逻辑，这里只需调用即可
        # self.visual_fusion._init_weights() # 构造函数里已经调用了

    # ... (_init_weights 函数保持不变) ...
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # [修改点 2]：不再需要 _align_depth_to_rgb 函数
    # Cross Attention 天然支持不同长度序列的交互，
    # 只要 Depth 是 [B, N_depth, C] 格式即可。
        
    def forward_itm(self, pixel_values, input_ids, attention_mask):
            # 1. 提取 RGB 特征 (Semantic Tower)
            rgb_outputs = self.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
            rgb_embeds = rgb_outputs.last_hidden_state # [B, N, 1408], 类型通常是 bfloat16

            # 2. 提取 Depth 特征 (Geometric Tower)
            with torch.no_grad():
                self.depth_backbone.eval()
                
                # 这里的转换是为了适配 depth backbone 的输入要求
                images_unnorm = pixel_values * self.clip_std + self.clip_mean
                depth_input = (images_unnorm - self.imagenet_mean) / self.imagenet_std
                
                # 如果 depth backbone 是 float32，这里可能需要把输入转回 float32 防止报错
                # 但通常混合精度下没问题。如果有问题，可以用 .float()
                depth_outputs = self.depth_backbone(depth_input, output_hidden_states=True)
                
                # depth_raw: [B, N_depth, 384]
                depth_raw = depth_outputs.hidden_states[-1] 
                
                # <=== 【核心修复】 ===>
                # 强制将 depth 输出转为和 rgb_embeds 一样的类型 (bfloat16)
                if depth_raw.dtype != rgb_embeds.dtype:
                    depth_raw = depth_raw.to(rgb_embeds.dtype)

            # 3. Cross Attention 融合
            # 现在两个输入都是 bfloat16 了
            image_embeds = self.visual_fusion(rgb_embeds, depth_raw)

            # 创建 Visual Mask
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

            # 4. Q-Former 交互
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            batch_size = input_ids.shape[0]
            query_attention_mask = torch.ones(
                (batch_size, query_tokens.shape[1]), 
                dtype=torch.long, 
                device=input_ids.device
            )
            qformer_attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)

            query_outputs = self.qformer(
                input_ids=input_ids,
                attention_mask=qformer_attention_mask,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            
            qformer_features = query_outputs.last_hidden_state
            pooled_features = torch.mean(qformer_features, dim=1)
            
            head_dtype = self.itm_head[1].weight.dtype 
            pooled_features = pooled_features.to(head_dtype)
            
            itm_logits = self.itm_head(pooled_features)
            
            return itm_logits