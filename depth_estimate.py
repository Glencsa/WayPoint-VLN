import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np

class DepthEstimator:
    def __init__(self, model_id="Depth-Anything-V2-Small-hf", device="cuda"):
        """
        初始化深度估计模型
        Args:
            model_id: Hugging Face 模型 ID，推荐使用 V2 Small 版本，速度快效果好
            device: 'cuda' or 'cpu'
        """
        print(f"Loading Depth Anything model: {model_id}...")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval() # 开启评估模式

    def predict_depth(self, image: Image.Image, return_type="pil"):
        """
        输入 RGB 图像，输出估计的深度图
        Args:
            image: PIL.Image 对象 (RGB)
            return_type: 'pil' (返回可视化灰度图) 或 'tensor' (返回原始深度数值)
        
        Returns:
            PIL.Image (如果 return_type='pil')
            torch.Tensor [1, 1, H, W] (如果 return_type='tensor')
        """
        # 1. 预处理 (Resize, Normalize)
        # Depth Anything 默认会将图像 resize 到 518x518 进行推理
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 2. 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 3. 后处理：插值回原始尺寸
        # 模型输出的尺寸通常小于原图，需要插值放大
        # image.size 是 (W, H)，torch interpolate 需要 (H, W)
        original_size = image.size[::-1] 
        
        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=original_size,
            mode="bicubic",
            align_corners=False,
        )

        # --- 分支 A: 如果你是为了喂给 InstructBlip 模型训练 ---
        if return_type == "tensor":
            # 返回原始的深度值 (数值越大表示越深/或者视差越大，取决于具体模型训练目标)
            # 通常建议归一化到 0-1 之间以便神经网络处理
            depth_min = prediction.min()
            depth_max = prediction.max()
            normalized_depth = (prediction - depth_min) / (depth_max - depth_min)
            return normalized_depth # [1, 1, H, W]

        # --- 分支 B: 如果你是为了可视化或保存图片 ---
        elif return_type == "pil":
            # 转换为 numpy
            depth_numpy = prediction.squeeze().cpu().numpy()
            
            # 归一化到 0-255 用于可视化
            depth_min = depth_numpy.min()
            depth_max = depth_numpy.max()
            depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min)
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            
            # 转为 PIL Image (L模式: 8位灰度)
            return Image.fromarray(depth_uint8)

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    import requests
    from io import BytesIO

    # 1. 初始化模型 (全局只运行一次)
    estimator = DepthEstimator(device="cuda" if torch.cuda.is_available() else "cpu")

    # 2. 准备一张测试图
    img = "test.jpeg"
    image = Image.open(img).convert("RGB")
    
    # 3. 获取深度图 (Tensor 格式，用于你的 InstructBlip 训练)
    depth_tensor = estimator.predict_depth(image, return_type="tensor")
    print(f"Depth Tensor Shape: {depth_tensor.shape}") # 预期: [1, 1, H, W]
    
    # 4. 获取深度图 (PIL 格式，用于查看)
    depth_image = estimator.predict_depth(image, return_type="pil")
    depth_image.save("depth_result.png")
    print("Depth image saved to depth_result.png")