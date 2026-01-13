import os
import torch
import numpy as np
from PIL import Image
from transformers import InstructBlipProcessor
import sys
current_path = os.path.abspath(__file__)
inference_dir = os.path.dirname(current_path)
project_root = os.path.dirname(inference_dir)
sys.path.append(project_root)
from utils.utils import prepare_inputs_for_generate
try:
    from models.rvln import RvlnMultiTask
except ImportError:
    raise ImportError("请确保 models/rvln.py 存在，并且其中定义了 RvlnMultiTask 类。")


CHECKPOINT_PATH = "output/rvln_merged_final"  
stage1_checkpoint = "output/stage1_checkpoint/latest_checkpoint.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  

def load_model():
    print(f"Loading model from: {CHECKPOINT_PATH}")
    processor = InstructBlipProcessor.from_pretrained(CHECKPOINT_PATH)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hist_id = tokenizer.convert_tokens_to_ids("<history>")
    curr_id = tokenizer.convert_tokens_to_ids("<current>")
    vocab_size = len(tokenizer)
    print(f"   -> Tokenizer IDs: <history>={hist_id}, <current>={curr_id}, Vocab={vocab_size}")
    # Load Model
    model = RvlnMultiTask.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    if os.path.exists(stage1_checkpoint):
        print(f"📥 发现 ITM 权重: {stage1_checkpoint}，正在加载覆盖...")
        checkpoint = torch.load(stage1_checkpoint, map_location="cpu")
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
    model.eval()
    # model_emb_size = model.language_model.get_input_embeddings().weight.shape[0]
    # print(f"   -> Model Embedding Size: {model_emb_size}")
    # model.language_model.resize_token_embeddings(len(tokenizer))
    print("Model loaded successfully!")
    return model, processor


def run_inference(model, processor, rgb_input, depth_input, instruction):
    """
    rgb_input: 可以是单张图片路径(str)，也可以是路径列表(list[str])
    depth_input: 同上
    """
    # 统一转为 list 格式方便处理
    if not isinstance(rgb_input, list):
        rgb_input = [rgb_input]
    if not isinstance(depth_input, list):
        depth_input = [depth_input]
        
    print(f"\n📸 Processing sequence (len={len(rgb_input)})...")
    
    # 1. 预处理 (自动补齐)
    inputs = prepare_inputs_for_generate(rgb_input, depth_input, instruction, processor, model.device)
    # print("input:"  , inputs)
    # 2. 生成
    print("🚀 Generating...")
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=inputs["pixel_values"],
            depth_pixel_values=inputs["depth_pixel_values"],
            qformer_input_ids=inputs["qformer_input_ids"],
            qformer_attention_mask=inputs["qformer_attention_mask"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.0 
        )

    # 3. 解码
    print("output:", outputs)
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print("-" * 40)
    print(f"📝 Prediction: {output_text.strip()}")
    print("-" * 40)

if __name__ == "__main__":
    # 初始化
    model, processor = load_model()
    
    instruction = 'Walk past the foot of the bed and exit the bedroom through the double doors ahead of you. Once out of the bedroom take a quick dogleg to the left and enter the large room with a chandelier ahead of you.'
    
    # 场景 1: 只有当前一张图 (刚启动)
    # 系统会自动补齐为: [黑, 黑, 黑, 黑, Img1]
    rgb_1 = ["test_data/rgb.jpg"]
    depth_1 = ["test_data/depth.jpg"]
    run_inference(model, processor, rgb_1, depth_1, instruction)

    # # 场景 2: 已经走了几步 (历史队列)
    # # 系统会自动取最后5张: [Img1, Img2, Img3, Img4, Img5] (假设 Img5 是当前)
    # # 这里用同一个图模拟多帧
    # rgb_history = [rgb_1[0]] * 6  # 模拟有6张图
    # depth_history = [depth_1[0]] * 6
    # run_inference(model, processor, rgb_history, depth_history, instruction)