import os 
import torch
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO
import numpy as np # 新增 numpy 用于检查 NaN

# 引入定义好的模型类
from models.InstructBlip import InstructBlipMultiTask
from transformers import (
    InstructBlipProcessor,
    BertTokenizer
)

def run_inference():

    MODEL_ID = "./instructblip-vicuna-7b"
    CHECKPOINT_PATH = "./checkpoints_itm_cross_attn/best_checkpoint.pth" 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"正在初始化 (Device: {DEVICE}, Main Dtype: {DTYPE})...")

    # Load Model and Processor
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
    qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("加载 InstructBlipMultiTask 基础模型...")
    model = InstructBlipMultiTask.from_pretrained(
        MODEL_ID, 
        torch_dtype=DTYPE
    )
    model.to(DEVICE)
    if hasattr(model, 'depth_model'):
        model.depth_model.to(dtype=torch.float32)
        print("Depth Model 已强制转换为 Float32")
    else:
        print("警告：未检测到 depth_model，请检查模型定义！")
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📥 发现训练权重: {CHECKPOINT_PATH}，正在加载...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu") # 先加载到 CPU 防止显存波动
        try:
            model.visual_fusion.load_state_dict(checkpoint['visual_fusion'], strict=True)
            model.visual_fusion.to(device=DEVICE, dtype=DTYPE) 
            print(f"Visual Fusion 加载成功")
        except KeyError:
            print("错误: Checkpoint 中找不到 'visual_fusion'！")
        except Exception as e:
            print(f"Visual Fusion 加载报错: {e}")

        try:
            model.itm_head.load_state_dict(checkpoint['itm_head'], strict=True)
            model.itm_head.to(device=DEVICE, dtype=DTYPE)
            print(f"ITM Head 加载成功")
        except KeyError:
            print("错误: Checkpoint 中找不到 'itm_head'！")
        
    else:
        print(f"未找到权重文件: {CHECKPOINT_PATH}")

    model.eval()

    # data preparation
    print("\n准备测试图片...")
    img_path = "test7.jpg"
    
    if not os.path.exists(img_path):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        print(f"   本地无图片，正在下载示例图片: {url}")
        raw_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        raw_image.save("test.jpeg")
    else:
        raw_image = Image.open(img_path).convert("RGB")

    # =================================================
    # Task 1: 自回归文本生成
    # =================================================
    print("\n" + "="*40)
    print("测试 1: 自回归文本生成")
    print("="*40)
    
    prompt = "Describe this image in detail."
    inputs_gen = processor(images=raw_image, text=prompt, return_tensors="pt").to(DEVICE)
    inputs_gen["pixel_values"] = inputs_gen["pixel_values"].to(dtype=DTYPE)
    
    with torch.no_grad():
        outputs = model.generate(**inputs_gen, max_new_tokens=500)
    
    print(f"Prompt: {prompt}")
    print(f"Output: {processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()}")

    # =================================================
    # Task 2: 图文匹配 (ITM)
    # =================================================
    print("\n" + "="*40)
    print("测试 2: 图文匹配 (ITM)")
    print("="*40)
    
    test_texts = [
        "Imagine you are a robot, and the image shows your current perspective. Your task is to get to the bathroom. Tell me if going in this direction will get you to the bathroom.", 
        "A red sports car driving on the highway", 
        "Imagine you are a robot, and the image shows your current perspective. Your task is to get to the living room and find the white chair. Tell me if going in this direction will get you to there." 
    ]
    
    image_inputs = processor(images=raw_image, return_tensors="pt").to(DEVICE)
    pixel_values = image_inputs.pixel_values.to(dtype=DTYPE) 
    
    # 【修改 5】输入数据安全检查
    if torch.isnan(pixel_values).any():
        print("致命错误: 输入图像 Tensor 包含 NaN！")
        return

    text_inputs = qformer_tokenizer(
        test_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    ).to(DEVICE)
    
    pixel_values_expanded = pixel_values.repeat(len(test_texts), 1, 1, 1)

    print("正在计算 Cross-Attention Fusion 及 ITM 分数...")
    with torch.no_grad():
        logits = model.forward_itm(
            pixel_values=pixel_values_expanded,
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        probs = torch.softmax(logits, dim=1)
    
    print("\n匹配结果:")
    for i, text in enumerate(test_texts):
        score_match = probs[i][1].item()
        
        # 安全处理，防止之前没捕获的 NaN 导致 int() 报错
        if np.isnan(score_match):
            bar_len = 0
            score_str = "NaN"
        else:
            bar_len = int(score_match * 20)
            score_str = f"{score_match:.6f}"
            
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"Text: '{text}'")
        print(f"Score: {score_str} | {bar}")
        print("-" * 30)

if __name__ == "__main__":
    run_inference()