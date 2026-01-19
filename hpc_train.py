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
from models.WayPointVLN import RvlnMultiTask 
from utils.data_utils import RvlnLoRADataset, DataCollatorForRvln
from utils.utils import *

class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_to_value = {}       
        self.digit_canonical_ids = [] 
        self.minus_token_ids = set()  
        self.key_token_weight = 1.0   
        self.minus_token_weight = 20.0 
        self.soft_loss_weight = 5.0    
        self.sigma = 2.0              


        minus_candidates = ["-", " -", "-1", " -1"]
        for s in minus_candidates:
            tid = self.tokenizer.convert_tokens_to_ids(s)
            if tid != self.tokenizer.unk_token_id:
                self.minus_token_ids.add(tid)
        
        if self.is_world_process_zero():
            print(f"Stop/Minus Tokens Registered: {self.minus_token_ids} (Weight: {self.minus_token_weight})")

        for i in range(9):
            s = str(i)
            ids = [
                self.tokenizer.convert_tokens_to_ids(s),
                self.tokenizer.convert_tokens_to_ids(" " + s)
            ]
            
            canonical_added = False
            for tid in ids:
                if tid != self.tokenizer.unk_token_id:
                    if tid not in self.minus_token_ids:
                        self.id_to_value[tid] = i
                        
                        if not canonical_added:
                            self.digit_canonical_ids.append(tid)
                            canonical_added = True
        
        if len(self.digit_canonical_ids) != 9:
            print("Warning: Can not find all digit tokens 0-8 in tokenizer vocab.")
        else:
            if self.is_world_process_zero():
                print(f"Navigation Tokens Registered: 0-8 (Sigma={self.sigma})")

    def generate_gaussian_target(self, gt_values, num_classes=9):
        """
        Generate Gaussian-distributed soft targets for ordinal regression.
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
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        batch_size, seq_len, vocab_size = shift_logits.shape
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_losses = loss_fct(flat_logits, flat_labels)

        weights = torch.ones_like(token_losses)

        for mid in self.minus_token_ids:
            weights[flat_labels == mid] = self.minus_token_weight

        ordinal_mask = torch.zeros_like(token_losses, dtype=torch.bool)
        ordinal_gt_values = torch.zeros_like(flat_labels, dtype=torch.long)
        for tid, val in self.id_to_value.items():
            is_digit = (flat_labels == tid)
            if is_digit.any():
                ordinal_mask |= is_digit
                ordinal_gt_values[is_digit] = val
        weighted_loss = token_losses * weights
        active_elements = (flat_labels != -100).sum()
        base_loss = weighted_loss.sum() / (active_elements + 1e-6)
        soft_loss = torch.tensor(0.0, device=flat_logits.device)
        
        if ordinal_mask.any():
            digit_ids_tensor = torch.tensor(self.digit_canonical_ids, device=flat_logits.device)
            subset_logits = flat_logits[ordinal_mask][:, digit_ids_tensor]
            subset_log_probs = F.log_softmax(subset_logits, dim=-1)
            subset_gt = ordinal_gt_values[ordinal_mask]
            soft_targets = self.generate_gaussian_target(subset_gt, num_classes=9)
            kl_loss = F.kl_div(subset_log_probs, soft_targets, reduction='batchmean')
            soft_loss = kl_loss

        final_loss = base_loss + self.soft_loss_weight * soft_loss

        return (final_loss, outputs) if return_outputs else final_loss

    def save_model(self, output_dir=None, _internal_call=False):

        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.is_world_process_zero():
            print(f"Saving Checkpoint to {output_dir}...")
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
            print(f"Checkpoint saved: LoRA + Stage1 Weights included.")
def main():
    model_name_or_path = "./instructblip-vicuna-7b" 
    depth_encoder_path = "./vit-base-patch16-224"
    # Weight: Fusion, Q-Former, Depth
    stage1_checkpoint = "checkpoints/latest_checkpoint.pth"
    data_path = "/home/guanbin/scratch/dataset/r2r_dataset/rgb_images_r2r_train.json"
    output_dir = "./output/rvln_sft_llm_new"
    batch_size = 4 
    grad_accumulation = 4 
    learning_rate = 2e-4 
    num_epochs = 50
    lora_rank = 32
    lora_alpha = 64
    
    
    print("Loading Processor...")
    processor = InstructBlipProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {'additional_special_tokens': ["<history>", "<current>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    history_token_id = tokenizer.convert_tokens_to_ids("<history>")
    current_token_id = tokenizer.convert_tokens_to_ids("<current>")

    print("Loading Base Model...")
    config = InstructBlipConfig.from_pretrained(model_name_or_path)
    config.history_token_id = history_token_id
    config.current_token_id = current_token_id
    config.depth_model_name_or_path = depth_encoder_path
    model = RvlnMultiTask.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16
    )

    model.language_model.resize_token_embeddings(len(tokenizer))

    if os.path.exists(stage1_checkpoint):
        print(f"📥 Loading Stage 1 Checkpoint from: {stage1_checkpoint}")
        ckpt = torch.load(stage1_checkpoint, map_location="cpu")
        
        msg = model.load_state_dict(ckpt, strict=False) 
        print(f"Checkpoint Load Status: {msg}")
        
        if 'visual_fusion' in ckpt: print(" - Visual Fusion Loaded ✅")
        if 'qformer' in ckpt: print(" - Q-Former Loaded ")
        if 'depth_backbone' in ckpt: print(" - Depth Backbone Loaded ✅")
    else:
        print(" Warning: Stage 1 checkpoint not found! Training from scratch (Not Recommended).")

    for param in model.parameters():
        param.requires_grad = False
        
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

    print("Loading Full Dataset...")
    full_dataset = RvlnLoRADataset(
        data_path=data_path,
        processor=processor,
        tokenizer=tokenizer,
        image_root="", 
        history_len=4,
        current_len=1
    )
    
    val_ratio = 0.1  
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    print(f"Splitting Dataset: Total={len(full_dataset)} | Train={train_size} | Val={val_size}")

    train_dataset, eval_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )

    collator = DataCollatorForRvln(
        processor=processor,
        tokenizer=tokenizer
    )

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

    if trainer.is_world_process_zero():
        print("Starting Save process...")
        
        final_adapter_dir = os.path.join(output_dir, "final_adapter")
        os.makedirs(final_adapter_dir, exist_ok=True)
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        print(f"   - Saving LoRA adapters to {final_adapter_dir}...")
        peft_model = unwrapped_model.language_model
        peft_model.save_pretrained(final_adapter_dir)

        print(f"   - Saving Stage 1 frozen weights (Safety Backup)...")
        stage1_weights = {}
        for name, param in unwrapped_model.named_parameters():
            if "language_model" not in name:
                stage1_weights[name] = param.cpu()
        
        torch.save(stage1_weights, os.path.join(final_adapter_dir, "stage1_visual_weights.pth"))

        print("   - Saving Processor (Tokenizer + Image Config)...")
        if processor:
            processor.save_pretrained(final_adapter_dir)
        else:
            tokenizer.save_pretrained(final_adapter_dir)
        
        print(f"✅ Save Complete! Output Checkpoint is self-contained in: {final_adapter_dir}")

if __name__ == "__main__":
    main()
