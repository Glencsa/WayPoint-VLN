import os
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np 
import torch.nn.functional as F
from torch.utils.data import random_split
import swanlab
from transformers import (
    InstructBlipProcessor,
    InstructBlipConfig,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from swanlab.integration.huggingface import SwanLabCallback
from models.rvln import RvlnMultiTask 
from data_utils import RvlnLoRADataset, DataCollatorForRvln
from utils import *

# ==========================================
# 3. 自定义 Trainer (确保保存 Embeddings)
# ==========================================
class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        """重写保存逻辑，确保 LoRA + Embeddings + Tokenizer 都能被保存"""
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存 LoRA 和 modules_to_save (embed_tokens)
        super().save_model(output_dir, _internal_call)
        
        # 2. 保存 Tokenizer
        if self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)
            
            print(f"Model (LoRA + Embeddings) saved to {output_dir}")

class WeightedTrainer(CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 获取数字 -1, 0-8 的 Token ID
        # 注意：不同 Tokenizer 对数字的处理不同，有可能是 "8" 也有可能是 " 8" (带空格)
        # 这里把常见可能都加进去，确保万无一失
        self.target_tokens = set()
        for i in range(-1, 9): # -1 到 8
            # 纯数字
            self.target_tokens.add(self.tokenizer.convert_tokens_to_ids(str(i)))
            # 带空格的数字 (SentencePiece 常见)
            self.target_tokens.add(self.tokenizer.convert_tokens_to_ids(" " + str(i)))
        
        # 处理 "-1" 这种情况，Tokenzier 可能会把它拆成 "-" 和 "1"
        # 如果你想把 "-" 也加权，可以加上
        self.target_tokens.add(self.tokenizer.convert_tokens_to_ids("-"))

        # 权重倍数：关键 Token 的 Loss 放大 10 倍
        self.key_token_weight = 10.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写 Loss 计算逻辑，对数字 Token 进行加权
        """
        # 1. 正常的前向传播
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        # 2. 获取 Logits
        logits = outputs.get("logits")
        
        # 3. 移位 (Shift) 以适配 Causal LM
        # 预测第 i 个 token 用的是第 i-1 个 token 的输出
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 4. 展平
        batch_size, seq_len, vocab_size = shift_logits.shape
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        
        # 5. 计算未缩减的 CrossEntropy Loss (reduction='none')
        # 这样我们会得到每一个 Token 的 Loss，而不是一个平均值
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # 只需要计算 label != -100 的部分
        token_losses = loss_fct(flat_logits, flat_labels)
        
        # 6. 创建权重 Mask
        # 默认权重为 1.0
        weights = torch.ones_like(token_losses)
        
        # 找到 Label 是数字的地方，将权重设为 10.0
        # 这是一个 Tensor 操作，速度很快
        for target_id in self.target_tokens:
            weights[flat_labels == target_id] = self.key_token_weight
            
        # 7. 应用权重
        weighted_loss = token_losses * weights
        
        # 8. 取平均 (只对非 Mask 的部分取平均)
        # 统计有效 Token 数量 (labels != -100)
        active_elements = (flat_labels != -100).sum()
        
        if active_elements > 0:
            final_loss = weighted_loss.sum() / active_elements
        else:
            final_loss = weighted_loss.sum()

        return (final_loss, outputs) if return_outputs else final_loss


class ClassificationTrainer(CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_buffer = []
        self.last_eval_visual_step = -1

    def generate_gaussian_target(self, labels, num_classes, sigma=1.0):
        """
        生成高斯软标签
        labels: [Batch_Size] 真实的类别索引
        num_classes: 总类别数
        sigma: 高斯分布的标准差，控制"宽容度"。Sigma越大，允许的误差范围越宽。
        """
        device = labels.device
        batch_size = labels.size(0)
        
        # 1. 创建所有类别的索引 [1, num_classes] -> [Batch, num_classes]
        # 这里假设 Index 0 是 Stop，不参与距离计算，所以我们只处理 1~N
        # 如果你的类别定义不同，请相应调整
        range_tensor = torch.arange(num_classes, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 2. 扩展标签维度 [Batch, 1]
        target_tensor = labels.unsqueeze(1)
        
        # 3. 计算距离 (绝对值距离)
        # distance: [Batch, Num_Classes]
        distance = torch.abs(range_tensor - target_tensor)
        
        # --- 进阶：如果是全景图(0和8是相邻的)，可以使用环形距离 ---
        # distance = torch.min(distance, num_classes - 1 - distance) # 仅当首尾相接时开启
        
        # 4. 生成高斯分布
        # exp(- dist^2 / (2 * sigma^2))
        scores = torch.exp(- (distance.float() ** 2) / (2 * sigma ** 2))
        
        # 5. 特殊处理 Stop 标签 (Index 0)
        # 假设 Index 0 是 "Stop/停"，它不应该和 "Index 1 (方向0)" 相近
        # 逻辑：
        # - 如果真实标签是 0: 目标就是 One-hot [1, 0, 0...]
        # - 如果真实标签是 >0: 目标是在 1~N 之间的高斯分布，且 Index 0 的概率设为 0
        
        # 创建一个 mask，标记哪些样本的 GT 是 0
        is_stop_token = (labels == 0) # [Batch]
        
        # 对于 GT != 0 的样本，把 Index 0 的概率强制设为 0 (或者极小值)
        scores[:, 0] = 0.0
        
        # 6. 归一化 (让概率和为 1)
        # 加上 epsilon 防止除零
        probs = scores / (scores.sum(dim=1, keepdim=True) + 1e-9)
        
        # 7. 对于 GT == 0 的样本，强制恢复为 Hard Label [1, 0, 0, ...]
        # 构造 One-hot
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, target_tensor, 1.0)
        
        # 组合：如果是 Stop 则用 One-hot，否则用高斯分布
        final_targets = torch.where(is_stop_token.unsqueeze(1), one_hot, probs)
        
        return final_targets

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Loss 计算 (高斯软标签版) + 准确率累积
        """
        labels = inputs.get("class_labels")
        if labels is None:
            labels = inputs.get("labels")
            
        outputs = model(**inputs)
        logits = outputs.get("logits") # [Batch, Num_Classes]
        
        loss = None
        if logits is not None:
            # --- 核心修改：使用 Soft Target Cross Entropy ---
            
            # 1. 生成软标签目标
            # sigma=1.0 表示相邻的 1 个单位 Loss 也很小
            # sigma=0.5 表示要求比较严格
            # sigma=2.0 表示非常宽容
            num_classes = logits.size(-1)
            soft_targets = self.generate_gaussian_target(labels, num_classes, sigma=1.5)
            
            # 2. 计算 Loss
            # CrossEntropyLoss(pred, soft_target) 等价于 -sum(target * log_softmax(pred))
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 样本维度的 Loss: [Batch]
            # 公式: KL Divergence (忽略常数项) -> Cross Entropy
            loss_per_sample = -torch.sum(soft_targets * log_probs, dim=-1)
            loss = loss_per_sample.mean()

        # 4. 计算准确率 (保持不变，准确率还是看硬指标)
        if logits is not None:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                micro_acc = (preds == labels).float().mean().item()
                
                if model.training:
                    self.acc_buffer.append(micro_acc)
                    if len(self.acc_buffer) >= self.args.gradient_accumulation_steps:
                        avg_acc = sum(self.acc_buffer) / len(self.acc_buffer)
                        self.log({"train/accuracy": avg_acc})
                        self.acc_buffer = []

        self._handle_visualization(model, inputs, preds, labels)
        return (loss, outputs) if return_outputs else loss
    def _handle_visualization(self, model, inputs, preds, labels):
        """
        控制何时截图并上传到 SwanLab
        """
        current_step = self.state.global_step

        # 情况 1: 正在训练 (Training)
        # 每 50 步记录一次训练集的图
        if model.training:
            if self.is_world_process_zero() and current_step % 50 == 0:
                self._log_visuals(inputs, preds, labels, prefix="Train")

        # 情况 2: 正在验证 (Evaluation)
        # model.training 为 False
        else:
            # 只有在主进程，且当前这一轮 Eval 还没记录过图片时，才记录
            # (Trainer 在 Eval 过程中 global_step 是不会变的)
            if self.is_world_process_zero() and self.last_eval_visual_step != current_step:
                self._log_visuals(inputs, preds, labels, prefix="Eval")
                # 标记这一轮已经记录过了
                self.last_eval_visual_step = current_step

    def _log_visuals(self, inputs, preds, labels, prefix="Train"):
        """
        执行具体的上传操作
        prefix: 用于区分是 'Train' 还是 'Eval'
        """
        try:
            idx = 0 # 取 Batch 第一张图
            
            # 1. 还原文本
            instruction_text = self.processing_class.decode(
                inputs["input_ids"][idx], 
                skip_special_tokens=True
            )
            display_text = instruction_text[:100] + "..." if len(instruction_text) > 100 else instruction_text

            # 2. 还原图片
            # [Batch, 5, 3, H, W] -> 取最后一帧 -> [3, H, W]
            rgb_tensor = inputs["pixel_values"][idx][-1] 
            rgb_img = self._tensor_to_pil(rgb_tensor)

            depth_tensor = inputs["depth_pixel_values"][idx][-1]
            depth_img = self._tensor_to_pil(depth_tensor, is_depth=True)

            # 3. 构建 Caption
            pred_val = preds[idx].item() - 1  # 还原回 -1~8
            gt_val = labels[idx].item() - 1
            status = "✅" if pred_val == gt_val else "❌"
            
            caption = (f"[{prefix}] {status} Pred: {pred_val} | GT: {gt_val}\n"
                       f"{display_text}")

            # 4. 发送 SwanLab (使用 prefix 分组)
            swanlab.log({
                f"Visual/{prefix}_RGB": swanlab.Image(rgb_img, caption=caption),
                f"Visual/{prefix}_Depth": swanlab.Image(depth_img, caption="Depth Map")
            })
            
        except Exception as e:
            print(f"SwanLab Visual Error: {e}")

    def _tensor_to_pil(self, tensor, is_depth=False):
        """反归一化并转 PIL"""
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = img - img.min()
        img = img / (img.max() + 1e-6)
        img = (img * 255).astype(np.uint8)
        return img



def main():
    # =================Configuration=================
    model_name_or_path = "./instructblip-vicuna-7b" 
    # Weight: Fusion, Q-Former, Depth
    stage1_checkpoint = "checkpoint/latest_checkpoint.pth"
    data_path = "dataset_waypoint/rgb_images_r2r_train_processed.json"
    output_dir = "./output/rvln_sft_llm"
    # 训练参数
    batch_size = 2
    grad_accumulation = 8 # 稍微加大累积，模拟更大 batch
    learning_rate = 5e-5  # SFT LLM 学习率
    num_epochs = 3
    lora_rank = 32
    lora_alpha = 64
    
    # ================= [SwanLab] 2. 初始化 SwanLab =================
    # 在这里定义实验名称和需要记录的配置信息
    swanlab.init(
        project="Rvln-LoRA-SFT",
        experiment_name="vicuna-7b-lora-stage2",
        description="Rvln Stage 2 SFT with LoRA monitoring",
        config={
            "model_name": model_name_or_path,
            "stage1_checkpoint": stage1_checkpoint,
            "data_path": data_path,
            "batch_size": batch_size,
            "grad_accumulation": grad_accumulation,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.05,
            "modules_to_save": ["embed_tokens", "lm_head","score_head"]
        }
    )
    
    # =================1. Processor & Tokenizer=================
    print("Loading Processor...")
    processor = InstructBlipProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    qformer_tokenizer = processor.qformer_tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 添加特殊 Token
    special_tokens_dict = {'additional_special_tokens': ["<history>", "<current>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    history_token_id = tokenizer.convert_tokens_to_ids("<history>")
    current_token_id = tokenizer.convert_tokens_to_ids("<current>")

    # =================2. Model Initialization=================
    print("Loading Base Model...")
    config = InstructBlipConfig.from_pretrained(model_name_or_path)
    config.history_token_id = history_token_id
    config.current_token_id = current_token_id

    # 加载基础模型
    model = RvlnMultiTask.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.float16
    )

    # 调整 Embedding 大小
    model.language_model.resize_token_embeddings(len(tokenizer))

    # =================3. [关键] 加载 Stage 1 训练好的权重=================
    if os.path.exists(stage1_checkpoint):
        print(f"📥 Loading Stage 1 Checkpoint from: {stage1_checkpoint}")
        ckpt = torch.load(stage1_checkpoint, map_location="cpu")
        
        msg = model.load_state_dict(ckpt, strict=False) 
        print(f"Checkpoint Load Status: {msg}")
        
        if 'visual_fusion' in ckpt: print(" - Visual Fusion Loaded ✅")
        if 'qformer' in ckpt: print(" - Q-Former Loaded ✅")
        if 'depth_backbone' in ckpt: print(" - Depth Backbone Loaded ✅")
    else:
        print("❌ Warning: Stage 1 checkpoint not found! Training from scratch (Not Recommended).")

    # =================4. Freeze & LoRA Setup=================
    
    # 4.1 全局冻结
    for param in model.parameters():
        param.requires_grad = False
        
    # 4.2 配置 LoRA (针对 LLM)
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head","score_head"] 
    )
    
    print("Applying LoRA to LLM...")
    model.language_model = get_peft_model(model.language_model, peft_config)
    
    print_trainable_parameters(model)

# ================= 5. Data Setup (关键修改：划分验证集) =================
    print("Loading Full Dataset...")
    full_dataset = RvlnLoRADataset(
        data_path=data_path,
        processor=processor,
        tokenizer=tokenizer,
        image_root="", 
        history_len=4,
        current_len=1
    )
    
    val_ratio = 0.01  # 1% 做验证，99% 训练
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    print(f"Splitting Dataset: Total={len(full_dataset)} | Train={train_size} | Val={val_size}")
    
    # generator用于固定随机种子，保证每次切分一样，方便复现
    train_dataset, eval_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )

    collator = DataCollatorForRvln(
        processor=processor,
        tokenizer=tokenizer,
        qformer_tokenizer=qformer_tokenizer
    )

    # ================= 6. Trainer Setup (关键修改：添加 Eval 配置) =================
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        fp16=True,
        deepspeed="./ds_config_zero2.json",
        remove_unused_columns=False,
        report_to="none",
        evaluation_strategy="steps",   # 按步数评估 (也可以选 "epoch")
        eval_steps=1000,                # 每 100 步评估一次验证集 (根据你总步数调整)
        per_device_eval_batch_size=batch_size, # 验证集的 Batch Size
        save_strategy="steps",         # 必须和 evaluation_strategy 一致
        save_steps=2000,                # 每 2000 步尝试保存
        save_total_limit=2,            # 最多保留 2 个 checkpoint，省硬盘
        load_best_model_at_end=True,   # 训练结束时，自动加载验证集效果最好的模型
        metric_for_best_model="loss",  # 以 loss 为标准 (loss 越小越好)
        greater_is_better=False,       # loss 是越小越好，所以是 False
        logging_steps=4,
    )

    # 使用自定义 Trainer
    trainer = ClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[SwanLabCallback()]
    )

    trainer.train()
    trainer.accelerator.wait_for_everyone()
    
    # 仅由主进程触发保存逻辑（或者 trainer.save_model 内部会处理，但加上 wait 更安全）
    if trainer.is_world_process_zero():
        trainer.save_model(output_dir)
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()