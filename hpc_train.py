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
        self.id_to_value = {}        # 仅存 0-8，用于 Soft Loss
        self.digit_canonical_ids = [] # 存储 0-8 的标准 Token ID
        self.minus_token_ids = set()  #专门存储负号相关的 Token ID
        self.key_token_weight = 1.0    # 普通数字 (0-8) 的权重
        self.minus_token_weight = 20.0 
        self.soft_loss_weight = 5.0    # 软标签权重
        self.sigma = 2.0               # 高斯分布标准差

        # --- A. 注册负号 (Stop Signal) ---
        # 只要包含负号，就认为是停止意图的开始，给予重罚
        minus_candidates = ["-", " -", "-1", " -1"]
        for s in minus_candidates:
            tid = self.tokenizer.convert_tokens_to_ids(s)
            if tid != self.tokenizer.unk_token_id:
                self.minus_token_ids.add(tid)
        
        # 打印日志确保加载成功
        if self.is_world_process_zero():
            print(f"🛑 Stop/Minus Tokens Registered: {self.minus_token_ids} (Weight: {self.minus_token_weight})")

        # --- B. 注册数字 0-8 (参与高斯计算) ---
        for i in range(9):
            s = str(i)
            ids = [
                self.tokenizer.convert_tokens_to_ids(s),
                self.tokenizer.convert_tokens_to_ids(" " + s)
            ]
            
            canonical_added = False
            for tid in ids:
                if tid != self.tokenizer.unk_token_id:
                    # [关键] 只有当它不是负号集合里的 ID 时，才注册为普通数字
                    # 防止 "-1" 这个 token 被同时注册
                    if tid not in self.minus_token_ids:
                        self.id_to_value[tid] = i
                        
                        if not canonical_added:
                            self.digit_canonical_ids.append(tid)
                            canonical_added = True
        
        # 检查完整性
        if len(self.digit_canonical_ids) != 9:
            print("⚠️ Warning: 无法找到 0-8 的完整 Token ID，软标签逻辑可能受损。")
        else:
            if self.is_world_process_zero():
                print(f"✅ Navigation Tokens Registered: 0-8 (Sigma={self.sigma})")

    def generate_gaussian_target(self, gt_values, num_classes=9):
        """
        生成高斯分布目标
        """
        device = gt_values.device
        target_indices = torch.arange(num_classes, device=device).expand(len(gt_values), -1)
        gt_expand = gt_values.unsqueeze(1).expand(-1, num_classes)
        distance = (target_indices - gt_expand).float() ** 2
        scores = torch.exp(-distance / (2 * self.sigma ** 2))
        probs = scores / scores.sum(dim=1, keepdim=True)
        return probs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Loss = Hard_Weighted_CE + Alpha * Soft_Gaussian_KL
        """
        # 1-4. 前向传播与展平
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        batch_size, seq_len, vocab_size = shift_logits.shape
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        # ==================== Part 1: 基础加权 Loss (Hard Target) ====================
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_losses = loss_fct(flat_logits, flat_labels)

        # 初始化权重为 1.0
        weights = torch.ones_like(token_losses)

        for mid in self.minus_token_ids:
            weights[flat_labels == mid] = self.minus_token_weight

        # 准备 Soft Loss 变量
        ordinal_mask = torch.zeros_like(token_losses, dtype=torch.bool)
        ordinal_gt_values = torch.zeros_like(flat_labels, dtype=torch.long)

        # [修改点 3] 处理数字 0-8
        # 注意：这里的 id_to_value 已经被我们在 __init__ 里清洗过，不包含负号
        for tid, val in self.id_to_value.items():
            is_digit = (flat_labels == tid)
            
            # 如果是普通数字，我们可以给它 key_token_weight (1.0)，也可以给更高，这里保持 1.0
            # 这里的 is_digit 会和上面的负号逻辑天然互斥 (ID 不会重复)
            if is_digit.any():
                # 只有 0-8 才开启 Soft Loss
                ordinal_mask |= is_digit
                ordinal_gt_values[is_digit] = val

        # 计算最终加权的 Hard Loss
        weighted_loss = token_losses * weights
        active_elements = (flat_labels != -100).sum()
        base_loss = weighted_loss.sum() / (active_elements + 1e-6)

        # ==================== Part 2: 距离感知 Loss (Soft Target) ====================
        soft_loss = torch.tensor(0.0, device=flat_logits.device)
        
        if ordinal_mask.any():
            # 1. 提取 Logits
            digit_ids_tensor = torch.tensor(self.digit_canonical_ids, device=flat_logits.device)
            subset_logits = flat_logits[ordinal_mask][:, digit_ids_tensor]
            
            # 2. 计算预测分布
            subset_log_probs = F.log_softmax(subset_logits, dim=-1)
            
            # 3. 生成高斯目标
            subset_gt = ordinal_gt_values[ordinal_mask]
            soft_targets = self.generate_gaussian_target(subset_gt, num_classes=9)
            
            # 4. KL 散度
            kl_loss = F.kl_div(subset_log_probs, soft_targets, reduction='batchmean')
            soft_loss = kl_loss

        # ==================== Part 3: 总 Loss ====================
        final_loss = base_loss + self.soft_loss_weight * soft_loss

        return (final_loss, outputs) if return_outputs else final_loss

    def save_model(self, output_dir=None, _internal_call=False):

        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.is_world_process_zero():
            print(f"💾 Saving Checkpoint to {output_dir}...")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            peft_model = unwrapped_model.language_model
            peft_model.save_pretrained(output_dir)
            
            stage1_weights = {}
            for name, param in unwrapped_model.named_parameters():
                if "language_model" not in name:
                    stage1_weights[name] = param.cpu()
            
            torch.save(stage1_weights, os.path.join(output_dir, "stage1_visual_weights.pth"))

            saver = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            if saver:
                saver.save_pretrained(output_dir)
            
            peft_model.config.save_pretrained(output_dir)
            print(f"✅ Checkpoint saved: LoRA + Stage1 Weights included.")
def main():
    # =================Configuration=================
    model_name_or_path = "./instructblip-vicuna-7b" 
    depth_encoder_path = "./vit-base-patch16-224"
    # Weight: Fusion, Q-Former, Depth
    stage1_checkpoint = "checkpoints/latest_checkpoint.pth"
    data_path = "/home/guanbin/scratch/dataset/r2r_dataset/rgb_images_r2r_train.json"
    output_dir = "./output/rvln_sft_llm_new"
    # 训练参数
    batch_size = 4 
    grad_accumulation = 8 # 稍微加大累积，模拟更大 batch
    learning_rate = 2e-4  # SFT LLM 学习率
    num_epochs = 50
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
    config.depth_model_name_or_path = depth_encoder_path
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
        modules_to_save=["embed_tokens", "lm_head"]
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
    
# ================= 7. Save Adapter & Dependencies =================
    # 仅主进程执行保存
    if trainer.is_world_process_zero():
        print("⏳ Starting Save process...")
        
        final_adapter_dir = os.path.join(output_dir, "final_adapter")
        os.makedirs(final_adapter_dir, exist_ok=True)

        # 1. [优化] 使用 Accelerator 解包模型 (兼容 DeepSpeed)
        # 这会剥离 DeepSpeed/DDP 壳子，拿到原始的 RvlnMultiTask
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        
        # 2. 保存 LoRA 权重 (包含 Embeddings/Head)
        print(f"   - Saving LoRA adapters to {final_adapter_dir}...")
        peft_model = unwrapped_model.language_model
        peft_model.save_pretrained(final_adapter_dir)

        # 3. 手动保存 Stage 1 权重 (Fusion & Depth)
        # 这样你的 output 文件夹就是独立的，不再依赖外部的 stage1_checkpoint
        print(f"   - Saving Stage 1 frozen weights (Safety Backup)...")
        stage1_weights = {}
        for name, param in unwrapped_model.named_parameters():
            # 筛选出不属于 LLM 的参数 (即 Visual, Depth, Fusion 部分)
            if "language_model" not in name:
                stage1_weights[name] = param.cpu()
        
        torch.save(stage1_weights, os.path.join(final_adapter_dir, "stage1_visual_weights.pth"))

        # 4. 保存完整的 Processor (不仅仅是 Tokenizer)
        print("   - Saving Processor (Tokenizer + Image Config)...")
        if processor:
            processor.save_pretrained(final_adapter_dir)
        else:
            tokenizer.save_pretrained(final_adapter_dir)
        
        print(f"✅ Save Complete! Output Checkpoint is self-contained in: {final_adapter_dir}")

if __name__ == "__main__":
    main()
