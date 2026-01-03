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


def compute_metrics(eval_pred):
    """
    专门用于 Trainer 评估验证集时的回调函数
    """
    logits, labels = eval_pred
    # logits 是 numpy 数组，需要取 argmax
    predictions = np.argmax(logits, axis=-1)
    
    # 计算准确率
    acc = (predictions == labels).mean()
    
    return {
        "accuracy": acc
    }