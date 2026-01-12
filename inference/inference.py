import os 
import torch
import torch.nn as nn
from PIL import Image
import requests
import numpy as np
from io import BytesIO
import sys
current_path = os.path.abspath(__file__)
inference_dir = os.path.dirname(current_path)
project_root = os.path.dirname(inference_dir)
sys.path.append(project_root)
from utils.utils import prepare_inputs_for_generate
# 引入定义好的模型类
try:
    from models.rvln import RvlnMultiTask
except ImportError:
    raise ImportError("请确保 models/rvln.py 存在，并且其中定义了 RvlnMultiTask 类。")

from transformers import (
    InstructBlipProcessor,
    BertTokenizer
)

# ================= 配置区域 =================
# RVLN 合并后的权重路径 (用于 Task 1 生成)
RVLN_MODEL_PATH = "output/rvln_merged_final"
# 基础 Vicuna 路径 (用于加载 Processor)
BASE_PROCESSOR_PATH = "./instructblip-vicuna-7b"
# ITM / Stage1 权重路径 (用于 Task 2 ITM)
ITM_CHECKPOINT_PATH = "output/stage1_checkpoint/latest_checkpoint.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 # 推理建议 fp16

# RVLN 序列参数
HISTORY_LEN = 4
CURRENT_LEN = 1
TOTAL_LEN = 5
QUERY_TOKENS = 32

def load_combined_model():
    print(f"正在初始化 (Device: {DEVICE}, Main Dtype: {DTYPE})...")

    # 1. 加载 Processor & Tokenizer
    # 优先尝试从合并路径加载，失败则回退基础路径
    try:
        processor = InstructBlipProcessor.from_pretrained(RVLN_MODEL_PATH)
    except:
        print(f"⚠️ 无法从 {RVLN_MODEL_PATH} 加载 Processor，使用基础路径...")
        processor = InstructBlipProcessor.from_pretrained(BASE_PROCESSOR_PATH)
    
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    
    
    hist_id = tokenizer.convert_tokens_to_ids("<history>")
    curr_id = tokenizer.convert_tokens_to_ids("<current>")
    vocab_size = len(tokenizer)
    print(f"   -> Tokenizer IDs: <history>={hist_id}, <current>={curr_id}, Vocab={vocab_size}")

    # 3. 加载 ITM 专用的 Q-Former Tokenizer
    qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 4. 加载 RvlnMultiTask 模型
    print(f"加载 RvlnMultiTask 模型: {RVLN_MODEL_PATH} ...")
    model = RvlnMultiTask.from_pretrained(
        RVLN_MODEL_PATH, 
        torch_dtype=DTYPE
    )
    model.to(DEVICE)
    model.eval()

    # 5. [关键] ID 强制同步 (防止生成乱码)
    print("🔧 执行 ID 同步...")
    model.config.history_token_id = hist_id
    model.config.current_token_id = curr_id
    
    # Resize embedding 如果需要
    if model.language_model.get_input_embeddings().weight.shape[0] < vocab_size:
        model.language_model.resize_token_embeddings(vocab_size)

    # 6. [可选] 加载额外的 ITM 权重
    # 如果 merged_model 里没有包含 stage 1 的 itm_head 权重，这里手动加载
    if os.path.exists(ITM_CHECKPOINT_PATH):
        print(f"📥 发现 ITM 权重: {ITM_CHECKPOINT_PATH}，正在加载覆盖...")
        checkpoint = torch.load(ITM_CHECKPOINT_PATH, map_location="cpu")
        if 'depth_backbone' in checkpoint:
            model.depth_backbone.load_state_dict(checkpoint['depth_backbone'], strict=True)
        else :
            print("   ⚠️ 警告: ITM 权重中未找到 depth_backbone 部分，跳过该部分加载。")
        if 'visual_fusion' in checkpoint:
            model.visual_fusion.load_state_dict(checkpoint['visual_fusion'], strict=True)
        else :
            print("   ⚠️ 警告: ITM 权重中未找到 visual_fusion 部分，跳过该部分加载。")
        if 'itm_head' in checkpoint:
            model.itm_head.load_state_dict(checkpoint['itm_head'], strict=True)
        else :
            print("   ⚠️ 警告: ITM 权重中未找到 itm_head 部分，跳过该部分加载。")
        if 'qformer' in checkpoint:
            model.qformer.load_state_dict(checkpoint['qformer'], strict=True)
        else :
            print("   ⚠️ 警告: ITM 权重中未找到 qformer 部分，跳过该部分加载。")
        if 'query_tokens' in checkpoint:
            model.query_tokens.data = checkpoint['query_tokens'].data.to(DEVICE)
        else :
            print("   ⚠️ 警告: ITM 权重中未找到 query_tokens 部分，跳过该部分加载。")
        # 加载 ITM Head
        if 'itm_head' in checkpoint:
            try:
                # 注意：RvlnMultiTask 可能将 itm_head 放在了不同位置，视你的类定义而定
                # 这里假设结构兼容
                msg = model.itm_head.load_state_dict(checkpoint['itm_head'], strict=False)
                model.itm_head.to(dtype=DTYPE)
                print(f"   -> ITM Head 加载成功: {msg}")
            except Exception as e:
                print(f"   ⚠️ ITM Head 加载失败 (可能结构不匹配): {e}")
        
        # 深度模型转换
        if hasattr(model, 'depth_model'):
            model.depth_model.to(dtype=torch.float32)

    return model, processor, qformer_tokenizer

def run_inference():
    # 初始化
    model, processor, qformer_tokenizer = load_combined_model()

    # 准备测试数据 (通用)
    img_path = "test_data/rgb.jpg"
    depth_path = "test_data/depth.jpg"
    raw_image = Image.open(img_path).convert("RGB")

    # 准备深度图 (如果没有，用纯黑替代测试)
    depth_image = Image.open(depth_path).convert("L")

    # =================================================
    # Task 1: RVLN 导航生成 (替换了原来的 Text Generation)
    # =================================================
    print("\n" + "="*40)
    print("测试 1: RVLN 导航指令预测")
    print("="*40)
    
    instruction = "go to the bedroom and the mirror is in front of you."
    
    # 模拟 RVLN 输入队列 (假设只有当前帧)
    rgb_queue = [raw_image]
    depth_queue = [depth_image]
    
    print(f"Instruction: {instruction}")
    
    # 预处理 (RVLN 专用)
    rvln_inputs = prepare_inputs_for_generate(rgb_queue, depth_queue, instruction, processor, DEVICE)
    print("🚀 RVLN 生成中...")
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=rvln_inputs["pixel_values"],
            depth_pixel_values=rvln_inputs["depth_pixel_values"],
            qformer_input_ids=rvln_inputs["qformer_input_ids"],
            qformer_attention_mask=rvln_inputs["qformer_attention_mask"],
            input_ids=rvln_inputs["input_ids"],
            attention_mask=rvln_inputs["attention_mask"],
            max_new_tokens=100,
            do_sample=False
        )
    
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print(f"🤖 RVLN Output: {output_text.strip()}")


    # =================================================
    # Task 2: 图文匹配 (ITM) (保留并适配)
    # =================================================
    print("\n" + "="*40)
    print("测试 2: 图文匹配 (ITM)")
    print("="*40)
    
    test_texts = [
        "A photo of two cats sleeping on a sofa.", 
        "A red sports car driving on the highway", 
        instruction 
    ]
    
    
    print("正在计算 ITM 分数...")
    
    # 1. 准备图像 Tensor (扩展到 5 帧)
    itm_rgb_queue = [raw_image]
    itm_depth_queue = [depth_image] # 深度图也需要
    
    # 复用函数拿到 Tensor [1, 5, 3, H, W]
    dummy_input = prepare_inputs_for_generate(itm_rgb_queue, itm_depth_queue, "dummy", processor, DEVICE)
    pixel_values_5d = dummy_input["pixel_values"] # [1, 5, 3, H, W]
    depth_pixel_values_5d = dummy_input["depth_pixel_values"] # [1, 5, 3, H, W]
    
    # 检查 NaN
    if torch.isnan(pixel_values_5d).any():
        print("致命错误: 输入图像 Tensor 包含 NaN！")
        return

    # 扩展 batch 维度以匹配 text 数量
    pixel_values_expanded = pixel_values_5d.repeat(len(test_texts), 1, 1, 1, 1)
    depth_values_expanded = depth_pixel_values_5d.repeat(len(test_texts), 1, 1, 1, 1)
    # 2. 准备文本
    text_inputs = qformer_tokenizer(
        test_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    ).to(DEVICE)

    current_pixel_values = pixel_values_expanded[:, -1, :, :, :]
    current_depth_values = depth_values_expanded[:, -1, :, :, :]

    print(f"Input Shape for ITM: {current_pixel_values.shape} (Expected: [B, 3, H, W])")

    with torch.no_grad():
        # 调用 forward_itm
        logits = model.forward_itm(
            pixel_values=current_pixel_values, 
            depth_pixel_values=current_depth_values,
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        probs = torch.softmax(logits, dim=1)
    
    print("\n匹配结果:")
    for i, text in enumerate(test_texts):
        score_match = probs[i][1].item()
        
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