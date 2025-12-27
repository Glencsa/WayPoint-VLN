import os
import torch
import torch.nn as nn
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
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model (LoRA + Embeddings) saved to {output_dir}")

def main():
    # =================Configuration=================
    model_name_or_path = "./instructblip-vicuna-7b" 
    # 之前训练好的 Stage 1 权重路径 (包含 Fusion, Q-Former, Depth 等)
    stage1_checkpoint = "checkpoint/latest_checkpoint.pth"
    
    data_path = "datasets/filtered_traj_3279.json"
    output_dir = "./output/instructblip_sft_llm"
    
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

    # =================5. Data Setup=================
    print("Loading Dataset...")
    train_dataset = InstructBlipLoRADataset(
        data_path=data_path,
        processor=processor,
        tokenizer=tokenizer,
        image_root="", 
        history_len=4,
        current_len=1
    )
    
    # 使用 Wrapper 后的 Collator
    collator = DataCollatorWrapper(
        processor=processor,
        tokenizer=tokenizer,
        qformer_tokenizer=qformer_tokenizer
    )

    # =================6. Trainer Setup=================
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        num_train_epochs=num_epochs,
        fp16=True,
        deepspeed="./ds_config_zero2.json",
        remove_unused_columns=False,
        save_total_limit=1,
        report_to="none", # ⚠️ 建议设为 none，完全通过下方的 callback 控制 swanlab
    )

    # 使用自定义 Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        # ================= [SwanLab] 3. 添加 Callback =================
        # SwanLabCallback 会自动记录 Loss, LR, Epoch 等信息
        callbacks=[SwanLabCallback()]
    )

    trainer.train()
    
    trainer.save_model(output_dir)
    
    # 训练结束，结束 swanlab 记录 (可选，脚本结束会自动调用)
    # swanlab.finish()

if __name__ == "__main__":
    main()