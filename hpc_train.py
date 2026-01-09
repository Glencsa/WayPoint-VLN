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
from utils.data_utils import RvlnLoRADataset, DataCollatorForRvln
from utils.utils import *

class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # ==================== 1. 初始化 Token 映射与超参数 ====================
        self.target_token_ids = set() # 用于基础加权 Mask (包含 -1)
        self.id_to_value = {}         # 用于距离计算 (只包含 0-8)
        self.digit_canonical_ids = [] # 存储 0-8 的标准 Token ID，用于提取 Soft Logits
        
        # --- 超参数设置 ---
        self.key_token_weight = 1.0  # 硬标签权重 (做对了奖励大)
        self.soft_loss_weight = 5.0   # 软标签权重 (控制距离惩罚的力度)
        self.sigma = 2.0              # 高斯分布标准差 (越大越宽容)
        # --- A. 注册数字 0-8 (参与高斯计算) ---
        for i in range(9):
            s = str(i)
            # 获取该数字的所有可能 Token ID (例如 "1", " 1")
            ids = [
                self.tokenizer.convert_tokens_to_ids(s),
                self.tokenizer.convert_tokens_to_ids(" " + s)
            ]
            
            # 记录第一个有效的 ID 作为该数字的"代表"，用于提取 Logits 计算 Soft Loss
            # (通常 tokenizer 的第一个结果就是最常用的)
            canonical_added = False
            
            for tid in ids:
                if tid != self.tokenizer.unk_token_id:
                    self.target_token_ids.add(tid)
                    self.id_to_value[tid] = i  # 建立 ID -> 整数值 的映射
                    
                    if not canonical_added:
                        self.digit_canonical_ids.append(tid)
                        canonical_added = True
        
        # 确保我们收集齐了 0-8 的代表 ID，否则无法进行 Softmax 计算
        if len(self.digit_canonical_ids) != 9:
            print("⚠️ Warning: 无法找到 0-8 的完整 Token ID，软标签逻辑可能受损。")

        # --- B. 注册负号/-1 (只加权，不参与高斯) ---
        # -1 代表 Stop，它在空间上没有"邻居"，所以只做硬分类
        neg_ids = [
            self.tokenizer.convert_tokens_to_ids("-"),
            self.tokenizer.convert_tokens_to_ids(" -"),
            self.tokenizer.convert_tokens_to_ids("-1"),
            self.tokenizer.convert_tokens_to_ids(" -1")
        ]
        for tid in neg_ids:
            if tid != self.tokenizer.unk_token_id:
                self.target_token_ids.add(tid)
                # 注意：不在 id_to_value 中注册

        # 打印日志（只在主进程）
        if self.is_world_process_zero():
            print(f"WeightedTrainer Ready:")
            print(f"  - Hard Weighted Tokens: {len(self.target_token_ids)}")
            print(f"  - Distance Aware Tokens: 0-8 (Sigma={self.sigma})")

    def generate_gaussian_target(self, gt_values, num_classes=9):
        """
        生成高斯分布目标
        gt_values: [Batch] 真实的数字值 (0-8)
        """
        device = gt_values.device
        # 创建 [Batch, 9] 的矩阵，每一行都是 0,1,2...8
        target_indices = torch.arange(num_classes, device=device).expand(len(gt_values), -1)
        # 扩展 GT: [Batch, 1] -> [Batch, 9]
        gt_expand = gt_values.unsqueeze(1).expand(-1, num_classes)
        
        # 计算距离平方
        distance = (target_indices - gt_expand).float() ** 2
        
        # 高斯公式: exp(-dist / 2*sigma^2)
        scores = torch.exp(-distance / (2 * self.sigma ** 2))
        
        # 归一化 (Sum = 1)，这就变成了一个概率分布
        probs = scores / scores.sum(dim=1, keepdim=True)
        return probs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Loss = Hard_Weighted_CE + Alpha * Soft_Gaussian_KL
        """
        # 1. 获取 Labels
        labels = inputs.get("labels")
        
        # 2. 前向传播
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 3. Shift 操作 (对齐 Logits 和 Labels)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 4. 展平
        batch_size, seq_len, vocab_size = shift_logits.shape
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        # ==================== Part 1: 基础加权 Loss (Hard Target) ====================
        # 计算所有 Token 的 CrossEntropy
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_losses = loss_fct(flat_logits, flat_labels)

        # 构建权重矩阵
        weights = torch.ones_like(token_losses)
        
        # 标记哪些位置是需要计算距离的数字 (0-8)
        ordinal_mask = torch.zeros_like(token_losses, dtype=torch.bool)
        # 存储这些位置对应的真实整数值
        ordinal_gt_values = torch.zeros_like(flat_labels, dtype=torch.long)

        # 应用权重并识别数字
        # (这里为了代码清晰使用了循环，Token 只有十几个，开销可忽略)
        for target_id in self.target_token_ids:
            is_target = (flat_labels == target_id)
            # 加权
            weights[is_target] = self.key_token_weight
            
            # 如果是 0-8，加入 Soft Loss 计算队列
            if target_id in self.id_to_value:
                ordinal_mask |= is_target
                # 记录该 Token ID 对应的整数值 (例如 ID 299 -> Value 8)
                ordinal_gt_values[is_target] = self.id_to_value[target_id]

        weighted_loss = token_losses * weights
        
        # 计算平均 Hard Loss
        active_elements = (flat_labels != -100).sum()
        base_loss = weighted_loss.sum() / (active_elements + 1e-6)

        # ==================== Part 2: 距离感知 Loss (Soft Target) ====================
        soft_loss = torch.tensor(0.0, device=flat_logits.device)
        
        if ordinal_mask.any():
            # 1. 取出属于数字的样本的 Logits
            # 我们只关心模型在 0-8 这 9 个 Token 上的表现
            # digit_canonical_ids 是我们预先存好的 [id_0, id_1, ..., id_8]
            digit_ids_tensor = torch.tensor(self.digit_canonical_ids, device=flat_logits.device)
            
            # 提取 Mask 对应的 Logits 行，且只提取 9 个数字列 -> [N_ordinal, 9]
            subset_logits = flat_logits[ordinal_mask][:, digit_ids_tensor]
            
            # 2. 计算 Log Softmax (模型预测分布)
            subset_log_probs = F.log_softmax(subset_logits, dim=-1)
            
            # 3. 生成高斯目标分布 (Target分布) -> [N_ordinal, 9]
            subset_gt = ordinal_gt_values[ordinal_mask]
            soft_targets = self.generate_gaussian_target(subset_gt, num_classes=9)
            
            # 4. 计算 KL 散度 (KLDiv = -Sum(P_target * log P_pred))
            # 衡量模型分布与高斯分布的差异
            kl_loss = F.kl_div(subset_log_probs, soft_targets, reduction='batchmean')
            
            soft_loss = kl_loss

        # ==================== Part 3: 总 Loss ====================
        final_loss = base_loss + self.soft_loss_weight * soft_loss

        return (final_loss, outputs) if return_outputs else final_loss

    def save_model(self, output_dir=None, _internal_call=False):
            """
            自定义保存逻辑：针对嵌套 LoRA 结构 (Rvln -> LLM -> LoRA)
            """
            if output_dir is None:
                output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # --- 关键步骤：获取被 Unwrap 的模型 ---
            # 如果使用了 DeepSpeed 或 DDP，最外层会被 wrap，需要先剥离
            model_to_save = self.model
            if hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module
                
            # --- 关键步骤：定位 LoRA 核心 ---
            # 你的 LoRA 是加在 model.language_model 上的
            # 这里的 peft_model 就是那个被 get_peft_model 包裹的对象
            peft_model = model_to_save.language_model
            
            # 仅在主进程执行保存操作
            if self.is_world_process_zero():
                print(f"💾 Saving LoRA adapters and trained modules to {output_dir}...")
                
                # 1. 保存 LoRA 权重 + modules_to_save (embed_tokens, lm_head)
                # PEFT 库会自动处理 modules_to_save，将它们和 adapter 一起存下来
                peft_model.save_pretrained(output_dir)
                
                # 2. 保存 Tokenizer
                saver = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
                if saver:
                    saver.save_pretrained(output_dir)
                
                # 3. 保存 Config (可选，方便查看)
                peft_model.config.save_pretrained(output_dir)

                print(f"✅ Model components saved successfully.")
def main():
    # =================Configuration=================
    model_name_or_path = "./instructblip-vicuna-7b" 
    # Weight: Fusion, Q-Former, Depth
    stage1_checkpoint = "checkpoints/latest_checkpoint.pth"
    data_path = "/home/guanbin/scratch/dataset/r2r_dataset/rgb_images_r2r_train.json"
    output_dir = "./output/rvln_sft_llm_new"
    # 训练参数
    batch_size = 4 
    grad_accumulation = 8 # 稍微加大累积，模拟更大 batch
    learning_rate = 2e-4  # SFT LLM 学习率
    num_epochs = 10
    lora_rank = 32
    lora_alpha = 64
    
    
    # =================1. Processor & Tokenizer=================
    print("Loading Processor...")
    processor = InstructBlipProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
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
    
    val_ratio = 0.1  # 10% 做验证，90% 训练
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
        tokenizer=tokenizer
    )

    # ================= 6. Trainer Setup =================
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        num_train_epochs=num_epochs,
        fp16=False,
        bf16=True,
        deepspeed="./config/ds_config_zero2_1.json",
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
        callbacks=[],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
    trainer.accelerator.wait_for_everyone()
    
    # ================= 7. Save Adapter Only (常规保存 LoRA，不合并) =================
    # 仅主进程执行保存，避免多进程写入冲突
    if trainer.is_world_process_zero():
        print("⏳ Starting Save process (Adapter Only)...")
        
        # 1. 定义保存路径 (建议单独一个子文件夹，清晰明了)
        final_adapter_dir = os.path.join(output_dir, "final_adapter")
        os.makedirs(final_adapter_dir, exist_ok=True)

        # 2. 获取模型本体 (剥离 DeepSpeed/DDP 的封装)
        model_to_save = trainer.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        # 3. 关键步骤：定位 LoRA 模块
        # 你的 LoRA 是加在 model.language_model 上的，它是一个 PeftModel 对象
        peft_model = model_to_save.language_model
        
        # 4. 保存 LoRA 权重
        # PEFT 库会自动检测 config 中的 modules_to_save (embed_tokens, lm_head)
        # 并将它们与 lora 权重一起保存到 adapter_model.safetensors 中
        print(f"   - Saving LoRA adapters and trainable modules to {final_adapter_dir}...")
        peft_model.save_pretrained(final_adapter_dir)
        
        # 5. 保存 Tokenizer
        # 确保推理时使用的 tokenizer 与训练时一致
        print("   - Saving Tokenizer...")
        tokenizer.save_pretrained(final_adapter_dir)
        
        # 6. 保存 LoRA Config (包含 rank, alpha, base_model_path 等信息)
        peft_model.config.save_pretrained(final_adapter_dir)

        print(f"✅ Adapter saved successfully! Path: {final_adapter_dir}")
        print("   (You can load this with PeftModel.from_pretrained over the base model)")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
