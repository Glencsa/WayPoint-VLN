import os
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np 
import torch.nn.functional as F
from torch.utils.data import random_split
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
        
        # --- 初始化目标 Token 集合 ---
        self.target_token_ids = set()
        
        # 定义你需要加权的数字（字符形式）
        # 包括 -1 和 0-8
        target_strings = [str(i) for i in range(9)] + ["-1", "-"] 
        
        # 遍历词表，找到所有可能的编码形式
        vocab = self.tokenizer.get_vocab()
        
        # 推荐：直接精准添加 ID (以 Llama/Qwen 等常用 Tokenizer 为例)
        # 1. 纯数字
        for i in range(9):
            # 尝试添加 "1", " 1" 等形式
            self.target_token_ids.add(self.tokenizer.convert_tokens_to_ids(str(i)))
            # 有些 tokenizer 会把空格后的数字单独作为一个 token
            self.target_token_ids.add(self.tokenizer.convert_tokens_to_ids(" " + str(i)))
        
        # 2. 处理负号 (对于 -1)
        self.target_token_ids.add(self.tokenizer.convert_tokens_to_ids("-"))
        self.target_token_ids.add(self.tokenizer.convert_tokens_to_ids(" -"))

        # 移除可能存在的 Unknown token ID
        if self.tokenizer.unk_token_id in self.target_token_ids:
            self.target_token_ids.remove(self.tokenizer.unk_token_id)
            
        print(f"WeightedTrainer: 已激活加权 Token IDs: {self.target_token_ids}")

        # 权重倍数
        self.key_token_weight = 10.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        自定义 Loss 计算，对特定 Token 进行加权
        """
        # 1. 获取 Labels 并确保 device 正确
        labels = inputs.get("labels")
        
        # 2. 前向传播
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 3. 移位 (Shift) 操作 - 核心步骤
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 4. 展平 (Flatten) 以便计算 CrossEntropy
        batch_size, seq_len, vocab_size = shift_logits.shape
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        # 5. 计算未缩减 (Reduction='none') 的 Loss
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_losses = loss_fct(flat_logits, flat_labels)

        # 6. 构建权重矩阵
        weights = torch.ones_like(token_losses)
        
        # 7. 识别目标 Token 并加权
        for target_id in self.target_token_ids:
            weights[flat_labels == target_id] = self.key_token_weight
            
        # 8. 应用权重
        weighted_loss = token_losses * weights

        # 9. 计算最终平均 Loss
        active_elements = (flat_labels != -100).sum()
        
        if active_elements > 0:
            final_loss = weighted_loss.sum() / active_elements
        else:
            final_loss = weighted_loss.sum() # 防止除以 0

        return (final_loss, outputs) if return_outputs else final_loss


class ClassificationTrainer(CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_buffer = []
        # [已移除] self.last_eval_visual_step = -1

    def generate_gaussian_target(self, labels, num_classes, sigma=1.0):
        """
        生成高斯软标签
        """
        device = labels.device
        batch_size = labels.size(0)
        
        range_tensor = torch.arange(num_classes, device=device).unsqueeze(0).expand(batch_size, -1)
        target_tensor = labels.unsqueeze(1)
        
        distance = torch.abs(range_tensor - target_tensor)
        scores = torch.exp(- (distance.float() ** 2) / (2 * sigma ** 2))
        
        is_stop_token = (labels == 0) # [Batch]
        scores[:, 0] = 0.0
        
        probs = scores / (scores.sum(dim=1, keepdim=True) + 1e-9)
        
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, target_tensor, 1.0)
        
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
            num_classes = logits.size(-1)
            soft_targets = self.generate_gaussian_target(labels, num_classes, sigma=1.5)
            
            log_probs = F.log_softmax(logits, dim=-1)
            
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

        # [已移除] 可视化相关调用 self._handle_visualization(model, inputs, preds, labels)
        return (loss, outputs) if return_outputs else loss

    # [已移除] def _handle_visualization(self, model, inputs, preds, labels): ...
    # [已移除] def _log_visuals(self, inputs, preds, labels, prefix="Train"): ...
    # [已移除] def _tensor_to_pil(self, tensor, is_depth=False): ...


def main():
    # =================Configuration=================
    model_name_or_path = "./instructblip-vicuna-7b" 
    # Weight: Fusion, Q-Former, Depth
    stage1_checkpoint = "checkpoints/latest_checkpoint.pth"
    data_path = "/home/guanbin/scratch/dataset/r2r_dataset/rgb_images_r2r_train.json"
    output_dir = "./output/rvln_sft_llm"
    # 训练参数
    batch_size = 4 
    grad_accumulation = 8 # 稍微加大累积，模拟更大 batch
    learning_rate = 5e-5  # SFT LLM 学习率
    num_epochs = 3
    lora_rank = 32
    lora_alpha = 64
    
    # [已移除] SwanLab 初始化代码块
    # swanlab.login(...)
    # swanlab.init(...)
    
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
        torch_dtype=torch.bfloat16
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
        modules_to_save=["embed_tokens", "lm_head"]# "score_head" 
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

    # ================= 6. Trainer Setup =================
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        fp16=False,
        bf16=True,
        deepspeed="./ds_config_zero2_1.json",
        remove_unused_columns=False,
        report_to="none", 
        evaluation_strategy="steps",   
        eval_steps=1000,                
        per_device_eval_batch_size=batch_size, 
        save_strategy="steps",         
        save_steps=2000,                
        save_total_limit=2,            
        load_best_model_at_end=True,   
        metric_for_best_model="loss",  
        greater_is_better=False,       
        logging_steps=4,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        tf32=True,
        gradient_checkpointing=True,   
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )

    # 使用自定义 Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[], # [已移除] 移除了 SwanLabCallback
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
    trainer.accelerator.wait_for_everyone()
    
    # 仅由主进程触发保存逻辑
    if trainer.is_world_process_zero():
        trainer.save_model(output_dir)
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
