import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    InstructBlipForConditionalGeneration, 
    InstructBlipConfig, 
    AutoModelForDepthEstimation
)



class GatedDepthFusion(nn.Module):
    def __init__(self, rgb_dim, depth_dim):
        super().__init__()

        # 1. 深度特征投影层 (将 Depth 映射到 RGB 维度)
        self.depth_proj = nn.Linear(depth_dim, rgb_dim)
        
        # 2. 门控生成器 (Gate Generator)
        # 输入: RGB + 投影后的 Depth
        # 输出: 1个标量 (针对每个Token) 范围 [0, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(rgb_dim * 2, rgb_dim // 2),
            nn.ReLU(),
            nn.Linear(rgb_dim // 2, 1),
            nn.Sigmoid() # 保证输出在 0~1 之间
        )
        
        # 3. 层归一化 (可选，有助于稳定)
        self.norm = nn.LayerNorm(rgb_dim)
        
        # === 关键：初始化 ===
        self._init_weights()

    def _init_weights(self):
        # 深度投影正常初始化
        nn.init.xavier_normal_(self.depth_proj.weight)
        nn.init.constant_(self.depth_proj.bias, 0)
        
        # 【核心 Trick】将门控网络的最后一层初始化为极小值
        # 这样初始状态下 Gate 输出接近 0.5 (如果bias是0) 或 0
        # 我们可以强制让 gate 的 bias 变为负数，让初始 gate 接近 0
        nn.init.constant_(self.gate_net[-2].weight, 0)
        nn.init.constant_(self.gate_net[-2].bias, 0) # Sigmoid(-3) ≈ 0.04 (接近关闭)

    def forward(self, rgb_embeds, depth_embeds):
        """
        rgb_embeds: [B, N, C]
        depth_embeds: [B, N, C_depth]
        """
        # 1. 投影深度特征
        target_dtype = self.depth_proj.weight.dtype 
        
        # 将输入强转为与权重一致的类型
        if rgb_embeds.dtype != target_dtype:
            rgb_embeds = rgb_embeds.to(target_dtype)
            
        if depth_embeds.dtype != target_dtype:
            depth_embeds = depth_embeds.to(target_dtype)
        depth_proj = self.depth_proj(depth_embeds)
        
        # 2. 计算门控值 (Gate)
        # 拼接特征用来判断重要性
        concat_feats = torch.cat([rgb_embeds, depth_proj], dim=-1)
        gate = self.gate_net(concat_feats) # [B, N, 1]
        
        # 3. 加权融合 (Residual)
        # 初始阶段 gate 接近 0，相当于 fused ≈ rgb_embeds (纯RGB)，保证收敛
        # 随着训练，gate 会在重要区域变大
        fused_features = rgb_embeds + gate * depth_proj
        
        return self.norm(fused_features)
    

    

# ==============================================================================
# 模型定义: Dual-Tower InstructBlip (RGB + Depth)
# ==============================================================================
class InstructBlipMultiTask(InstructBlipForConditionalGeneration):
    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)
        
        # -----------------------------------------------------------
        # 1. 新增: Depth Anything V2 Backbone (几何专家)
        # -----------------------------------------------------------
        depth_model_name = "./Depth-Anything-V2-Small-hf"
        print(f"Loading Depth Backbone: {depth_model_name}...")
        
        # 加载预训练的 Depth 模型
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name)
        self.depth_backbone = self.depth_model.backbone
        
        # 移除 Depth 模型的 Head 以节省显存 (我们只需要 backbone 特征)
        if hasattr(self.depth_model, "head"):
            del self.depth_model.head
            
        # 冻结 Depth Backbone (通常不需要微调它)
        for param in self.depth_backbone.parameters():
            param.requires_grad = False
            
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # 获取维度信息
        # InstructBLIP Vision (CLIP-ViT-g) 通常是 1408
        self.rgb_hidden_size = config.vision_config.hidden_size 
        # Depth Anything Small 通常是 384
        self.depth_hidden_size = self.depth_backbone.config.hidden_size
        
        # # 我们将 RGB 和 Depth 拼接，然后投影回 Q-Former 期望的维度
        # # Input: RGB_Dim + Depth_Dim -> Output: RGB_Dim (Q-Former Cross-Attn Dim)
        # self.visual_fusion_proj = nn.Linear(
        #     self.rgb_hidden_size + self.depth_hidden_size, 
        #     self.rgb_hidden_size
        # )
        # self.visual_layer_norm = nn.LayerNorm(self.rgb_hidden_size)
        self.visual_fusion = GatedDepthFusion(
                    rgb_dim=self.rgb_hidden_size,    # 1408
                    depth_dim=self.depth_hidden_size # 384
                )
        self.qformer_hidden_size = config.qformer_config.hidden_size
        self.itm_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.qformer_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2) 
        )
        
        self._init_weights(self.itm_head)
        self.visual_fusion._init_weights()
        # self._init_weights(self.visual_fusion_proj)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _align_depth_to_rgb(self, rgb_embeds, depth_embeds):
            """
            修改后的对齐函数：
            输入: 
                rgb_embeds: [B, 257, 1408] (含 CLS)
                depth_embeds: [B, 257, 384] (含 CLS)
            输出: 
                depth_final: [B, 257, 384] (对齐后的 Patches + 原装 CLS)
            """
            # 1. 分离 Depth 的 CLS 和 Patches
            depth_cls = depth_embeds[:, 0:1, :]      # [B, 1, 384]
            depth_patches = depth_embeds[:, 1:, :]   # [B, N_depth, 384]
            
            # 2. 获取 RGB 的 Patch 数量（去掉 CLS）
            # 假设 RGB 也有 CLS，如果 RGB 是 257，那 patch 就是 256
            num_patches_rgb = rgb_embeds.shape[1] - 1 
            grid_size_rgb = int(num_patches_rgb**0.5) # e.g., 16
            
            # 3. 对 Depth Patches 进行空间对齐
            B, N_depth, C_depth = depth_patches.shape
            grid_size_depth = int(N_depth**0.5)
            
            # 变成 2D图 -> 插值 -> 变回序列
            depth_map = depth_patches.permute(0, 2, 1).view(B, C_depth, grid_size_depth, grid_size_depth)
            
            depth_map_resized = F.interpolate(
                depth_map, 
                size=(grid_size_rgb, grid_size_rgb), 
                mode='bilinear', 
                align_corners=False
            )
            
            depth_patches_aligned = depth_map_resized.flatten(2).transpose(1, 2) # [B, N_rgb, 384]
            depth_final = torch.cat([depth_cls, depth_patches_aligned], dim=1) # [B, 1+N_rgb, 384]
            
            return depth_final


    def forward_itm(self, pixel_values, input_ids, attention_mask):
        """
        融合了 Depth 信息的 ITM 前向传播
        """
        # =================================================
        # 1. 提取 RGB 特征 (Semantic Tower)
        # =================================================
        # vision_outputs.last_hidden_state: [B, N_rgb, 1408]
        rgb_outputs = self.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        rgb_embeds = rgb_outputs.last_hidden_state
        # =================================================
        # 2. 提取 Depth 特征 (Geometric Tower)
        # =================================================
        with torch.no_grad(): # 确保 Depth 塔不更新参数
            # Depth Anything 需要 3 通道输入，直接复用 pixel_values
            # 注意：如果 Depth 和 RGB 要求的 normalize 不同，这里可能需要反归一化再归一化
            # 但通常为了效率，直接输入影响在可接受范围内
            self.depth_backbone.eval() # 确保是 eval 模式
            
            # 像素值域转换
            images_unnorm = pixel_values * self.clip_std + self.clip_mean
            depth_input = (images_unnorm - self.imagenet_mean) / self.imagenet_std
            depth_outputs = self.depth_backbone(depth_input, output_hidden_states=True)

            # 取最后一层特征
            depth_embeds = depth_outputs.hidden_states[-1].to(dtype=rgb_embeds.dtype)


        # 2. 调用上面的新函数 (不需要再手动补0了)
        depth_aligned = self._align_depth_to_rgb(rgb_embeds, depth_embeds)
        image_embeds = self.visual_fusion(rgb_embeds, depth_aligned)

        # # 3. 此时 rgb_embeds 和 depth_aligned 长度都是 257，直接拼
        # fused_embeds = torch.cat([rgb_embeds, depth_aligned], dim=-1)
        
        # # 投影回 Q-Former 认识的维度: [B, N, 1792] -> [B, N, 1408]
        # image_embeds = self.visual_fusion_proj(fused_embeds)
        # image_embeds = self.visual_layer_norm(image_embeds)
        
        # 创建 Visual Mask
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # =================================================
        # 4. Q-Former 交互 (后续逻辑不变)
        # =================================================
        # 准备 Query Tokens [B, 32, 768]
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # 扩展 Attention Mask (Query + Text)
        batch_size = input_ids.shape[0]
        query_attention_mask = torch.ones(
            (batch_size, query_tokens.shape[1]), 
            dtype=torch.long, 
            device=input_ids.device
        )
        qformer_attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)

        # 输入 Q-Former
        query_outputs = self.qformer(
            input_ids=input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, # <--- 这是一个融合了 RGB+Depth 的特征
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        
        # Pooling & Classification
        qformer_features = query_outputs.last_hidden_state
        pooled_features = torch.mean(qformer_features, dim=1)
        
        # 确保类型匹配 (混合精度训练时很重要)
        head_dtype = self.itm_head[1].weight.dtype 
        pooled_features = pooled_features.to(head_dtype)
        
        itm_logits = self.itm_head(pooled_features)
        
        return itm_logits