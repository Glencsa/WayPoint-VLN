import numpy as np
import torch
from PIL import Image

def print_trainable_parameters(model):
    """打印可训练参数统计"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


def preprocess_logits_for_metrics(logits, labels):
    """
    预处理：1. 取 argmax 节省显存  2. 提前做好 Shift (错位)
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # 取预测出的 token id
    pred_ids = logits.argmax(dim=-1)
    
    # *** 关键修复：Shift 操作 ***
    # 模型在位置 t 输出的 pred_ids[t]，其实是预测 labels[t+1]
    # 所以我们需要把 preds 向左移一位，或者把 labels 向右移一位来对齐
    # 这里我们返回移位后的 preds: 去掉最后一个，因为它预测的是未来未知的
    return pred_ids[..., :-1]

def compute_metrics(eval_pred):
    """
    计算指标
    """
    # 解包
    preds = eval_pred.predictions # 已经在上面移位过了: [Batch, Seq-1]
    labels = eval_pred.label_ids  # 原始标签: [Batch, Seq]
    
    # 对应的，Labels 也要去掉第一个 (因为第一个 token 没有人预测它)
    # 这样 Preds[t] 就和 Labels[t+1] 对齐了
    shift_labels = labels[..., 1:]
    
    # 创建 Mask：只计算非 Padding 且非 Prompt (-100) 的部分
    mask = shift_labels != -100
    
    # 计算准确率
    # 只有在 mask 为 True 的地方才进行对比
    if mask.sum() > 0:
        acc = (preds[mask] == shift_labels[mask]).mean()
    else:
        acc = 0.0
        
    return {"accuracy": acc}

# inference


def prepare_inputs_for_generate(rgb_queue, depth_queue, instruction, processor, device,target_len=5,query_len=32):
    """
    rgb_queue: list, 包含图片路径(str)或PIL Image对象。可以是 1 张，也可以是多张。
    depth_queue: list, 同上。
    逻辑：自动将输入列表补齐或截取为 TOTAL_LEN (5张)。不足的前面补黑图，超过的取最后5张。
    """

    # --- 内部辅助函数：加载单张图片 ---
    def _load_as_rgb(item):
        if item is None:
            return Image.new('RGB', (224, 224), (0, 0, 0))
        if isinstance(item, str): # 如果是路径
            try:
                return Image.open(item).convert("RGB")
            except:
                print(f"⚠️ 无法加载图片: {item}，使用全黑代替")
                return Image.new('RGB', (224, 224), (0, 0, 0))
        elif isinstance(item, Image.Image): # 如果已经是 PIL Image
            return item.convert("RGB")
        elif isinstance(item, np.ndarray): # 如果是 numpy 数组
            # ================= [关键修改] =================
            # 1. 处理维度问题: (H, W, 1) -> (H, W)
            if item.ndim == 3 and item.shape[2] == 1:
                item = item.squeeze(2)  # 去掉最后一个维度
            
            # 2. 确保数据是 uint8 类型 (防止 float32 报错)
            if item.dtype != np.uint8:
                # 简单的归一化兼容
                if item.max() <= 1.0:
                    item = (item * 255).astype(np.uint8)
                else:
                    item = item.astype(np.uint8)
            # ============================================
            
            return Image.fromarray(item).convert("RGB")
        else:
            return Image.new('RGB', (224, 224), (0, 0, 0))

    # --- 内部辅助函数：序列补齐/截断 ---
    def pad_and_crop_sequence(raw_list):
        # 1. 先把所有输入加载为 PIL Image
        loaded_imgs = [_load_as_rgb(x) for x in raw_list]
        
        # 2. 计算需要补多少张
        current_len = len(loaded_imgs)
        pad_len = target_len - current_len
        
        if pad_len > 0:
            # 不够 5 张：前面补黑图
            # 例如只有 [Curr]，补齐后变成 [Black, Black, Black, Black, Curr]
            black_img = Image.new('RGB', (224, 224), (0, 0, 0))
            final_imgs = [black_img] * pad_len + loaded_imgs
        else:
            # 超过或等于 5 张：取最后 5 张 (最近的观测)
            final_imgs = loaded_imgs[-target_len:]
            
        return final_imgs

    # ================= 1. 处理图片队列 =================
    final_rgb_imgs = pad_and_crop_sequence(rgb_queue)
    final_depth_imgs = pad_and_crop_sequence(depth_queue)

    # ================= 2. Processor 处理图像 =================
    # InstructBlipProcessor 会将 list 视为 batch
    # 输出 shape: [5, 3, 224, 224] -> unsqueeze -> [1, 5, 3, 224, 224]
    
    inputs = processor(
        images=final_rgb_imgs,
        text=[instruction], # 这里其实只为了占位，重点是下面的 prompt 构造
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    
    depth_inputs = processor(
        images=final_depth_imgs,
        return_tensors="pt"
    )

    # ================= 3. 构造文本 Prompt =================
    # 注意：这里我们手动构造 Prompt，因为 Processor 默认的处理逻辑可能不包含重复 Token
    
    hist_token = "<history>"
    curr_token = "<current>"
    history_len = target_len - 1 
    # 构造 <history>...<history> 串 (4 * 32)
    hist_replacement = hist_token * (history_len * query_len)
    # 构造 <current>...<current> 串 (1 * 32)
    curr_replacement = curr_token * query_len
    
    raw_prompt = (
        f"Imagine you are a robot designed for navigation tasks. Your instruction is {instruction}.\n"
        f"You are provided with:\n"
        f"- Historical observations(four images): {hist_token} \n"
        f"- Current observation: {curr_token}, there are some routes on the current observation.\n\n"
        f"Your task is to select the best route number based on these routes, you can also choose the 0 to turn left 30\u00b0 ,choose the 8 to turn right 30\u00b0, or return -1 to Stop. \n"
        f" The format of the result is {{'Route': number -1~8}}"
    )
    
    # 执行字符串替换，把单个 token 扩展成 32 个
    expanded_prompt = raw_prompt.replace(hist_token, hist_replacement).replace(curr_token, curr_replacement)
    
    final_prompt = f"USER: {expanded_prompt} ASSISTANT:"

    text_inputs = processor(
        text=final_prompt,
        return_tensors="pt",
        padding="longest",
        truncation=True
    )
    
    # 处理 Q-Former 输入 (Batch=5)
    # 这一步非常关键：qformer input ids 必须是 [1, 5, 32] 这种形状
    qformer_input_ids = inputs.qformer_input_ids.to(device)
    qformer_attention_mask = inputs.qformer_attention_mask.to(device)

    return {
        "pixel_values": inputs.pixel_values.unsqueeze(0).to(device, dtype=torch.float16), # [1, 5, 3, H, W]
        "depth_pixel_values": depth_inputs.pixel_values.unsqueeze(0).to(device, dtype=torch.float16),
        "input_ids": text_inputs.input_ids.to(device),
        "attention_mask": text_inputs.attention_mask.to(device),
        "qformer_input_ids": qformer_input_ids,
        "qformer_attention_mask": qformer_attention_mask
    }
