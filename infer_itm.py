import os 
import torch
import torch.nn as nn
from PIL import Image
import requests
import numpy as np
import cv2

from models.depth_estimate import DepthEstimator
from models.rvln import RvlnMultiTask
from transformers import (
    InstructBlipProcessor,
    BertTokenizer,
    InstructBlipConfig,
    AutoTokenizer
)

def run_inference():
    # =================================================
    # 1. 基础配置
    # =================================================
    MODEL_ID = "./instructblip-vicuna-7b"
    CHECKPOINT_PATH = "checkpoint/latest_checkpoint.pth"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # ITM 推理建议使用 float16 或 bfloat16
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    print(f"正在初始化 (Device: {DEVICE}, Main Dtype: {DTYPE})...")

    # =================================================
    # 2. 配置 Tokenizer 和 Config (必须步骤，防止报错)
    # =================================================
    # 即使 ITM 不用 <history>，模型初始化检查也需要它们
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    special_tokens = {"additional_special_tokens": ["<history>", "<current>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    hist_id = tokenizer.convert_tokens_to_ids("<history>")
    curr_id = tokenizer.convert_tokens_to_ids("<current>")
    
    config = InstructBlipConfig.from_pretrained(MODEL_ID)
    config.history_token_id = hist_id
    config.current_token_id = curr_id

    # =================================================
    # 3. 加载模型
    # =================================================
    print(">>> 正在加载模型...")
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
    qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    estimator = DepthEstimator(model_id="./Depth-Anything-V2-Small-hf", device=DEVICE)
    
    model = RvlnMultiTask.from_pretrained(
        MODEL_ID, 
        config=config,
        torch_dtype=DTYPE
    )
    model.language_model.resize_token_embeddings(len(tokenizer))
    
    # 加载微调权重
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📥 加载权重: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        if 'depth_backbone' in checkpoint:
            model.depth_backbone.load_state_dict(checkpoint['depth_backbone'], strict=True)
        if 'visual_fusion' in checkpoint:
            model.visual_fusion.load_state_dict(checkpoint['visual_fusion'], strict=True)
        if 'itm_head' in checkpoint:
            model.itm_head.load_state_dict(checkpoint['itm_head'], strict=True)
        if 'qformer' in checkpoint:
            model.qformer.load_state_dict(checkpoint['qformer'], strict=True)
        if 'query_tokens' in checkpoint:
            model.query_tokens.data = checkpoint['query_tokens'].data
    else:
        print("⚠️ 未找到权重，使用随机初始化参数！")

    model.to(DEVICE)
    model.eval()
    
    # 深度模型通常需要 float32 保证精度，或者跟主模型一致
    if hasattr(model, 'depth_model'):
        model.depth_model.to(dtype=torch.float32)

    # =================================================
    # 4. 准备单张测试图
    # =================================================
    print("\n>>> 准备测试数据...")
    img_path = "images/test2.jpg"
    if not os.path.exists(img_path):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        raw_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        raw_image.save(img_path)
    else:
        raw_image = Image.open(img_path).convert("RGB")

    # 1. 获取 RGB Tensor [1, 3, H, W]
    inputs_rgb = processor(images=raw_image, return_tensors="pt")
    pixel_values = inputs_rgb.pixel_values.to(DEVICE, dtype=DTYPE)

    # 2. 获取 Depth Tensor [1, 3, H, W] (Processor通常输出3通道)
    # 你的 forward_itm 里有兼容逻辑：if shape[1]==1: repeat
    depth_pil = estimator.predict_depth(raw_image, return_type="pil", colormap=cv2.COLORMAP_JET)
    inputs_depth = processor(images=depth_pil, return_tensors="pt")
    # 注意：这里保持 float32 传进去，因为 forward_itm 内部会做 .to(dtype) 转换
    depth_values = inputs_depth.pixel_values.to(DEVICE, dtype=torch.float32)

    # =================================================
    # 5. 执行 ITM (一对多匹配)
    # =================================================
    print("\n" + "="*40)
    print("测试: 单图 vs 多文本匹配")
    print("="*40)
    
    # 定义候选文本
    test_texts = [
        "A photo of two cats sleeping on a pink blanket.",  # 这里的描述请根据你的测试图修改
        "A view of a modern kitchen with a refrigerator.",
        "Find the toilet."
    ]
    
    # 【关键步骤】数据对齐
    # 现在的输入是 1 张图，但有 N 个文本。
    # 我们需要把 Image Tensor 在 Batch 维度复制 N 次，变成 [N, 3, H, W]
    # 这样 forward_itm 里的逻辑 num_images_per_sample 就会等于 1
    
    batch_size = len(test_texts)
    
    # 扩展 RGB: [1, 3, H, W] -> [B, 3, H, W]
    batch_pixel_values = pixel_values.repeat(batch_size, 1, 1, 1)
    
    # 扩展 Depth: [1, 3, H, W] -> [B, 3, H, W]
    batch_depth_values = depth_values.repeat(batch_size, 1, 1, 1)
    
    # Tokenize 文本: [B, Seq_Len]
    text_inputs = qformer_tokenizer(
        test_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    ).to(DEVICE)
    
    print("正在计算匹配分数...")
    
    with torch.no_grad():
        # 调用你提供的 forward_itm
        # 此时输入维度：
        # pixel_values:       [3, 3, 224, 224]
        # depth_pixel_values: [3, 3, 224, 224]
        # input_ids:          [3, 32]
        logits = model.forward_itm(
            pixel_values=batch_pixel_values,
            depth_pixel_values=batch_depth_values,
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        
        # Softmax 获取概率 (假设 class 1 是匹配，class 0 是不匹配)
        # 你的 itm_head 输出维度是 [B, 2]
        probs = torch.softmax(logits, dim=1)
        
    print("\n>>> 匹配结果:")
    for i, text in enumerate(test_texts):
        # index 1 通常代表 "Match" (取决于你的训练 Label 设置，通常 1=Pos, 0=Neg)
        score = probs[i][1].item() 
        
        # 可视化进度条
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        print(f"Text: {text:<45} | Score: {score:.4f} | {bar}")

if __name__ == "__main__":
    run_inference()
