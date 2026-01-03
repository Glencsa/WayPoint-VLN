import numpy as np
import torch


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