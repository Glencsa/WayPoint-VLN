import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor

class DepthViTEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__()
        print(f"Loading ImageNet Pretrained ViT: {model_name}...")
        
        # 1. 加载纯净的 ViT 模型 (去掉分类头，只留 Transformer Encoder)
        # add_pooling_layer=False 表示我们不需要那个用于分类的 Pooler，我们要所有 token 的特征
        self.vit = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        
        # 冻结参数 (根据你的需求，也可以选择解冻微调)
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # 2. 获取 Hidden Size (ViT-Base 通常是 768)
        self.hidden_size = self.vit.config.hidden_size 

    def forward(self, depth_images):
        """
        Input: 
            depth_images: [B, 1, H, W] 或者 [B, 3, H, W]
                          如果是 [B, 1, ...], 我们会自动扩展为 3 通道
        Output:
            features: [B, N_patches + 1, 768] (+1 是因为有 CLS token)
        """
        # 1. 处理通道数
        # ImageNet 预训练模型第一层卷积期待 3 通道输入
        if depth_images.shape[1] == 1:
            # 简单的复制策略：将单通道深度重复 3 次
            # 这种做法虽然简单，但在迁移学习中非常有效，因为它保留了边缘梯度信息
            depth_input = depth_images.repeat(1, 3, 1, 1)
        else:
            depth_input = depth_images

        # 2. 尺寸检查 (ViT 对输入尺寸比较敏感，通常是 224x224)
        # 如果你的输入尺寸不是 224，HF 的 ViT 会尝试进行插值处理 Positional Embedding，
        # 但为了性能和效果，建议在输入前 resize 到 224
        if depth_input.shape[-1] != 224 or depth_input.shape[-2] != 224:
             depth_input = torch.nn.functional.interpolate(
                depth_input, size=(224, 224), mode='bicubic', align_corners=False
             )

        # 3. 前向传播
        outputs = self.vit(pixel_values=depth_input)
        
        # last_hidden_state: [B, 197, 768] (196个 patch token + 1个 CLS token)
        features = outputs.last_hidden_state
        
        return features