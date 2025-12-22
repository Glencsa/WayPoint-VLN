import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np
import cv2  # 引入 OpenCV 用于颜色映射

class DepthEstimator:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf", device="cuda"):
        """
        初始化深度估计模型
        Args:
            model_id: Hugging Face 模型 ID
            device: 'cuda' or 'cpu'
        """
        print(f"Loading Depth Anything model: {model_id}...")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def predict_depth(self, image: Image.Image, return_type="pil", colormap=cv2.COLORMAP_INFERNO):
        """
        输入 RGB 图像，输出估计的深度图
        Args:
            image: PIL.Image 对象 (RGB)
            return_type: 
                - 'pil': 返回可视化的彩色深度图 (PIL Image)
                - 'tensor': 返回原始深度数值 (torch.Tensor)
            colormap: cv2 的颜色映射模式 (例如 cv2.COLORMAP_INFERNO, cv2.COLORMAP_JET)
        
        Returns:
            PIL.Image (彩色) 或 torch.Tensor
        """
        # 1. 预处理
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 2. 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 3. 后处理：插值回原始尺寸
        original_size = image.size[::-1] # (H, W)
        
        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=original_size,
            mode="bicubic",
            align_corners=False,
        )

        # --- 分支 A: 返回 Tensor (用于训练) ---
        if return_type == "tensor":
            depth_min = prediction.min()
            depth_max = prediction.max()
            normalized_depth = (prediction - depth_min) / (depth_max - depth_min)
            return normalized_depth # [1, 1, H, W]

        # --- 分支 B: 返回彩色 PIL 图片 (用于可视化) ---
        elif return_type == "pil":
            # 1. 转为 numpy
            depth_numpy = prediction.squeeze().cpu().numpy()
            
            # 2. 归一化到 0-255
            depth_min = depth_numpy.min()
            depth_max = depth_numpy.max()
            
            # 防止除以0
            if depth_max - depth_min > 1e-6:
                depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = depth_numpy
                
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            
            # 3. 【核心修改】应用伪彩色映射
            # applyColorMap 需要输入 uint8 格式
            # COLORMAP_INFERNO 是目前深度图最流行的配色 (黑->紫->红->黄)
            # COLORMAP_JET 是传统的彩虹色 (蓝->绿->红)
            depth_color_bgr = cv2.applyColorMap(depth_uint8, colormap)
            
            # 4. 颜色空间转换 (OpenCV 默认是 BGR，PIL 需要 RGB)
            depth_color_rgb = cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)
            
            # 5. 转回 PIL Image
            return Image.fromarray(depth_color_rgb)

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    import os
    
    # 检查是否有 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 初始化模型
    # 注意：确保 model_id 正确，这里使用官方 V2 Small 路径
    estimator = DepthEstimator(model_id="./Depth-Anything-V2-Small-hf", device=device)

    # 2. 读取测试图 (请确保当前目录下有 test.jpg，或者修改为你自己的图片路径)
    img_path = "step_3.jpg" 
    
    if not os.path.exists(img_path):
        # 如果没有图片，创建一个随机噪点图测试
        print(f"警告: {img_path} 不存在，生成随机图片进行测试...")
        image = Image.fromarray(np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8))
    else:
        image = Image.open(img_path).convert("RGB")
    
    print("正在推理...")

    # 4. 获取并保存彩色深度图 (使用 JET 彩虹色)
    depth_image_jet = estimator.predict_depth(image, return_type="pil", colormap=cv2.COLORMAP_JET)
    depth_image_jet.save("depth_result_jet.png")
    print("✅ 保存成功: depth_result_jet.png (彩虹色)")