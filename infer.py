import os 
import torch
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO
from train_itm import InstructBlipMultiTask
from transformers import (
    InstructBlipForConditionalGeneration, 
    InstructBlipConfig, 
    InstructBlipProcessor,
    BertTokenizer
)


def run_inference():
    # --- 配置 ---
    MODEL_ID = "./instructblip-vicuna-7b"
    # 【关键】修改为你实际保存权重的路径
    CHECKPOINT_PATH = "checkpoints_itm_instructblip/checkpoint_step_1214.pth" 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"1. 正在加载模型和处理器 (Device: {DEVICE})...")
    
    # A. 加载处理器
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
    qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # B. 加载基础模型
    try:
        model = InstructBlipMultiTask.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except:
        model = InstructBlipMultiTask.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)

    # C. 【新增】加载训练好的 ITM Head 权重
    if os.path.exists(CHECKPOINT_PATH):
        print(f"   发现训练权重: {CHECKPOINT_PATH}，正在加载...")
        # 1. 读取权重文件
        itm_state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        
        # 2. 加载到 itm_head 模块
        # strict=True 确保键值完全匹配
        model.itm_head.load_state_dict(itm_state_dict, strict=True)
        
        # 3. 确保数据类型和设备正确
        # 建议把 Head 转为 float32 以保证推理精度，或者跟随模型 float16
        model.itm_head.to(model.device) 
        print("   ✅ 权重加载成功！")
    else:
        print(f"   ❌ 未找到权重文件: {CHECKPOINT_PATH}")
        print("   ⚠️ 警告：正在使用随机初始化的 Head，ITM 分数将无效！")

    model.eval() # 开启评估模式

    # --- 准备测试图片 ---
    print("\n2. 准备测试图片...")
    img_url = "test.jpeg" 
    try:
        raw_image = Image.open(img_url).convert("RGB")
    except:
        # 如果网络不通，生成一张纯色图防止报错
        print("   下载失败，使用纯色图代替...")
        raw_image = Image.new('RGB', (224, 224), color='red')

    # =================================================
    # 任务一：自回归生成
    # =================================================
    print("\n" + "="*40)
    print("任务测试 1: 自回归文本生成")
    print("="*40)
    
    prompt = "Describe this image in detail."
    inputs_gen = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs_gen, max_new_tokens=1000)
    
    print(f"Prompt: {prompt}")
    print(f"Model Output: {processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()}")

    # =================================================
    # 任务二：图文匹配 (ITM) - 使用训练好的权重
    # =================================================
    print("\n" + "="*40)
    print("任务测试 2: 图文匹配打分 (使用训练权重)")
    print("="*40)
    
    test_texts = [
        "A person holding a camera on the beach", # 匹配 (如果是上面的网图)
        "A red car driving on the street"         # 不匹配
    ]
    
    image_inputs = processor(images=raw_image, return_tensors="pt").to(model.device)
    pixel_values = image_inputs.pixel_values.to(dtype=torch.float16) # 确保 fp16
    
    text_inputs = qformer_tokenizer(
        test_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    ).to(model.device)
    
    pixel_values = pixel_values.repeat(len(test_texts), 1, 1, 1)

    with torch.no_grad():
        logits = model.forward_itm(
            pixel_values=pixel_values,
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        probs = torch.softmax(logits, dim=1)
    
    for i, text in enumerate(test_texts):
        score_match = probs[i][1].item()
        print(f"Text: '{text}'")
        print(f"ITM Probability: {score_match:.6f}")
        # 如果训练成功，匹配的句子分数应接近 1.0，不匹配的应接近 0.0
        print("-" * 20)

if __name__ == "__main__":
    # 确保 InstructBlipMultiTask 类已经在同一个文件中定义
    run_inference()