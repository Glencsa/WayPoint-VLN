import os
import torch
import torch.nn as nn
import torch.distributed as dist
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

# ==========================================
# [SwanLab] 1. 引入 SwanLab 和 HF 回调
# ==========================================
import swanlab
from swanlab.integration.huggingface import SwanLabCallback

# 引入你的自定义模块
from models.rvln import InstructBlipMultiTask 
# 引入你上面提供的 Dataset 和 Collator 类
from data_utils import InstructBlipLoRADataset, DataCollatorForInstructBlip 

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


# ==========================================
# 2. 修正 Data Collator 以匹配模型输入
# ==========================================
class DataCollatorWrapper(DataCollatorForInstructBlip):
    """
    包装你原本的 Collator，将输出的键名修改为模型 forward 函数需要的名字
    pixel_values_rgb -> pixel_values
    pixel_values_depth -> depth_pixel_values
    """
    def __call__(self, batch):
        outputs = super().__call__(batch)
        
        # 重命名键值以匹配 InstructBlipMultiTask.forward 的参数
        if "pixel_values_rgb" in outputs:
            outputs["pixel_values"] = outputs.pop("pixel_values_rgb")
        
        if "pixel_values_depth" in outputs:
            outputs["depth_pixel_values"] = outputs.pop("pixel_values_depth")
            
        return outputs

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
            
            print(f"✅ Model (LoRA + Embeddings) saved to {output_dir}")

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
def main():
    # =================Configuration=================
    model_name_or_path = "./instructblip-vicuna-7b" 
    # 之前训练好的 Stage 1 权重路径 (包含 Fusion, Q-Former, Depth 等)
    stage1_checkpoint = "checkpoint/latest_checkpoint.pth"
    
    data_path = "dataset_waypoint/rgb_images_r2r_train_processed.json"
    output_dir = "./output/rvln_sft_llm"
    
    # 训练参数
    batch_size = 2
    grad_accumulation = 4 # 稍微加大累积，模拟更大 batch
    learning_rate = 5e-5  # SFT LLM 学习率
    num_epochs = 3
    lora_rank = 32
    lora_alpha = 64
    
    # ================= [SwanLab] 2. 初始化 SwanLab =================
    # 在这里定义实验名称和需要记录的配置信息
    swanlab.init(
        project="InstructBlip-LoRA-SFT",
        experiment_name="vicuna-7b-lora-stage2",
        description="InstructBlip Stage 2 SFT with LoRA monitoring",
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
            "modules_to_save": ["embed_tokens", "lm_head"]
        }
    )
    
    # =================1. Processor & Tokenizer=================
    print("Loading Processor...")
    processor = InstructBlipProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    qformer_tokenizer = processor.qformer_tokenizer

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
    model = InstructBlipMultiTask.from_pretrained(
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
        # 针对 Vicuna/Llama 的所有线性层
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # ⚠️ 关键：因为加了新 token，必须训练 Embedding 层和 Head
        modules_to_save=["embed_tokens", "lm_head"] 
    )
    
    print("Applying LoRA to LLM...")
    model.language_model = get_peft_model(model.language_model, peft_config)
    
    print_trainable_parameters(model)

# ================= 5. Data Setup (关键修改：划分验证集) =================
    print("Loading Full Dataset...")
    full_dataset = InstructBlipLoRADataset(
        data_path=data_path,
        processor=processor,
        tokenizer=tokenizer,
        image_root="", 
        history_len=4,
        current_len=1
    )
    
    # [新增] 计算划分数量
    val_ratio = 0.01  # 1% 做验证，99% 训练
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    print(f"Splitting Dataset: Total={len(full_dataset)} | Train={train_size} | Val={val_size}")
    
    # [新增] 随机切分
    # generator用于固定随机种子，保证每次切分一样，方便复现
    train_dataset, eval_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )
    
    collator = DataCollatorWrapper(
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
        
        # --- [新增] 验证集相关配置 ---
        evaluation_strategy="steps",   # 按步数评估 (也可以选 "epoch")
        eval_steps=1000,                # 每 100 步评估一次验证集 (根据你总步数调整)
        per_device_eval_batch_size=batch_size, # 验证集的 Batch Size
        
        # --- [新增] 模型保存策略 (Save Best) ---
        save_strategy="steps",         # 必须和 evaluation_strategy 一致
        save_steps=2000,                # 每 2000 步尝试保存
        save_total_limit=2,            # 最多保留 2 个 checkpoint，省硬盘
        load_best_model_at_end=True,   # 训练结束时，自动加载验证集效果最好的模型
        metric_for_best_model="loss",  # 以 loss 为标准 (loss 越小越好)
        greater_is_better=False,       # loss 是越小越好，所以是 False
        logging_steps=5,
    )

    # 使用自定义 Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        # ================= [SwanLab] 3. 添加 Callback =================
        # SwanLabCallback 会自动记录 Loss, LR, Epoch 等信息
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