import os
import torch
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
    prepare_model_for_kbit_training,
    TaskType
)

# 假设你的代码保存在这些文件中
from models.InstructBlip import InstructBlipMultiTask  # 你提供的模型类
from data_utils import InstructBlipLoRADataset, DataCollatorForInstructBlip # 你提供的数据集类

def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量
    """
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

def main():
    # =================Configuration=================
    # 基础模型路径 (可以是本地路径或 huggingface hub id)
    # 例如: "Salesforce/instructblip-vicuna-7b"
    model_name_or_path = "./instructblip-vicuna-7b" 
    data_path = "/home/isvl/guan_code/RVLN/dataset_instructblip.json"
    output_dir = "./output/instructblip_depth_lora"
    
    # 训练超参数
    batch_size = 2 # 根据显存调整 (InstructBlip 显存占用较大)
    grad_accumulation = 4
    learning_rate = 1e-4 # LoRA 学习率通常比全量微调大一点
    num_epochs = 3
    
    # =================1. Processor & Tokenizer=================
    print("Loading Processor and Tokenizer...")
    processor = InstructBlipProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    qformer_tokenizer = processor.qformer_tokenizer

    special_tokens_dict = {'additional_special_tokens': ["<history>", "<current>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens.")
    
    # 获取 Token ID 以便传入模型 Config
    history_token_id = tokenizer.convert_tokens_to_ids("<history>")
    current_token_id = tokenizer.convert_tokens_to_ids("<current>")

    # =================2. Model Initialization=================
    print("Loading Model Config...")
    config = InstructBlipConfig.from_pretrained(model_name_or_path)
    
    # 将特殊 Token ID 注入 Config (你的模型类中做了检查)
    config.history_token_id = history_token_id
    config.current_token_id = current_token_id

    print("Loading Model (this may take a while)...")
    # 如果显存不够，可以考虑添加 load_in_8bit=True 或 load_in_4bit=True
    # 这里演示加载 float16/bfloat16
    model = InstructBlipMultiTask.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16, # 建议使用 bfloat16
        device_map="auto"
    )

    # 极其重要：因为添加了新 Token，必须调整 Embedding 层大小
    model.language_model.resize_token_embeddings(len(tokenizer))

    # =================3. Freeze & LoRA Setup=================
    # 策略：
    # 1. 冻结所有参数
    # 2. 解冻新加入的模块 (visual_fusion, itm_head)
    # 3. 对 LLM 应用 LoRA
    
    # 3.1 冻结所有
    for param in model.parameters():
        param.requires_grad = False
        
    # 3.2 解冻自定义模块 (必须全量训练，因为是随机初始化的)
    # 注意：depth_backbone 在你的 init 代码里已经设为 False 了
    for name, param in model.named_parameters():
        if "visual_fusion" in name or "itm_head" in name:
            param.requires_grad = True
            # 确保这些层是 float32 (可选，为了数值稳定性) 或者跟随模型 dtype
            # param.data = param.data.to(torch.float32) 

    # 3.3 配置 LoRA 针对 LLM
    # InstructBlip-Vicuna 的 LLM 是 Llama 架构
    # Target modules 通常是 q_proj, v_proj (attention)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 针对 LLM 的 Attention 层
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"] # 如果词表扩充了，通常建议训练 embedding
    )
    
    # 将 LoRA 应用于 language_model
    # 注意：InstructBlipMultiTask -> language_model (Llama)
    # 我们需要手动包装 language_model，或者直接对整个 model 调 get_peft_model
    # 但直接对整个 model 调可能会影响到不需要 LoRA 的部分。
    # 标准做法：只对 LLM 加 LoRA。
    
    print("Applying LoRA to LLM...")
    model.language_model = get_peft_model(model.language_model, peft_config)
    
    # 再次确保自定义模块是可训练的 (get_peft_model 可能会重置部分状态)
    for name, param in model.named_parameters():
        if "visual_fusion" in name or "itm_head" in name:
            param.requires_grad = True

    # 打印可训练参数情况
    print_trainable_parameters(model)

    # =================4. Dataset Loading=================
    print("Loading Dataset...")
    train_dataset = InstructBlipLoRADataset(
        data_path=data_path,
        processor=processor,
        tokenizer=tokenizer,
        image_root="",
        history_len=4,
        current_len=1
    )
    
    collator = DataCollatorForInstructBlip(
            processor=processor,
            tokenizer=tokenizer,
            qformer_tokenizer=qformer_tokenizer # <--- 传入这里
        )

    # =================5. Trainer Setup=================
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        bf16=True, # 推荐开启 BF16
        remove_unused_columns=False, # 必须设为 False，否则 Trainer 会过滤掉 pixel_values 等非标准参数
        report_to="tensorboard",
        ddp_find_unused_parameters=False, # 如果有多余的参数未参与计算设为 True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # =================6. Start Training=================
    print("Starting Training...")
    trainer.train()
    
    # =================7. Save Model=================
    # 保存 LoRA 权重
    trainer.model.language_model.save_pretrained(os.path.join(output_dir, "llm_lora"))
    
    # 保存自定义模块的权重 (因为它们不是 LoRA 的一部分，需手动保存)
    custom_modules_path = os.path.join(output_dir, "custom_modules.pth")
    custom_state_dict = {
        k: v.cpu() for k, v in model.named_parameters() 
        if ("visual_fusion" in k or "itm_head" in k)
    }
    torch.save(custom_state_dict, custom_modules_path)
    print(f"Custom modules saved to {custom_modules_path}")
    
    # 保存 Tokenizer (因为添加了新 token)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()