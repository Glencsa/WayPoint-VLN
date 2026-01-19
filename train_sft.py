import os
import json
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

from transformers import (
    InstructBlipProcessor,
    InstructBlipConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

from models.WayPointVLN import RvlnMultiTask
from utils.data_utils import RvlnLoRADataset, DataCollatorForRvln
from utils.utils import print_trainable_parameters, compute_metrics, preprocess_logits_for_metrics


# Load configuration from JSON
def _load_config(config_path: str = "config_train.json") -> Dict[str, Any]:
    """Load training configuration from JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return json.load(f)


CONFIG = _load_config("config/config_train.json")


class WeightedTrainer(Trainer):
    """Custom trainer with weighted loss and ordinal regression support."""

    MINUS_TOKEN_WEIGHT = 20.0
    SOFT_LOSS_WEIGHT = 5.0
    SIGMA = 2.0
    NUM_CLASSES = 9
    MINUS_CANDIDATES = ["-", " -", "-1", " -1"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_to_value: Dict[int, int] = {}
        self.digit_canonical_ids: list = []
        self.minus_token_ids: set = set()

        self._register_minus_tokens()
        self._register_digit_tokens()

    def _register_minus_tokens(self) -> None:
        """Register minus/stop tokens and their weights."""
        for candidate in self.MINUS_CANDIDATES:
            token_id = self.tokenizer.convert_tokens_to_ids(candidate)
            if token_id != self.tokenizer.unk_token_id:
                self.minus_token_ids.add(token_id)

        if self.is_world_process_zero():
            print(
                f"Stop/Minus Tokens Registered: {self.minus_token_ids} "
                f"(Weight: {self.MINUS_TOKEN_WEIGHT})"
            )

    def _register_digit_tokens(self) -> None:
        """Register digit tokens (0-8) for ordinal regression."""
        for digit in range(self.NUM_CLASSES):
            token_strings = [str(digit), " " + str(digit)]
            canonical_added = False

            for token_str in token_strings:
                token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                if (
                    token_id != self.tokenizer.unk_token_id
                    and token_id not in self.minus_token_ids
                ):
                    self.id_to_value[token_id] = digit
                    if not canonical_added:
                        self.digit_canonical_ids.append(token_id)
                        canonical_added = True

        if len(self.digit_canonical_ids) != self.NUM_CLASSES:
            print(
                f"Warning: Could not find all digit tokens 0-{self.NUM_CLASSES - 1}."
            )
        elif self.is_world_process_zero():
            print(
                f"Navigation Tokens Registered: 0-{self.NUM_CLASSES - 1} "
                f"(Sigma={self.SIGMA})"
            )

    def _generate_gaussian_target(
        self, gt_values: torch.Tensor, num_classes: int = 9
    ) -> torch.Tensor:
        """Generate Gaussian-distributed soft targets for ordinal regression.

        Args:
            gt_values: Ground truth class values (batch_size,)
            num_classes: Number of ordinal classes

        Returns:
            Soft target probabilities (batch_size, num_classes)
        """
        device = gt_values.device
        target_indices = torch.arange(num_classes, device=device).expand(
            len(gt_values), -1
        )
        gt_expand = gt_values.unsqueeze(1).expand(-1, num_classes)
        distance = (target_indices - gt_expand).float() ** 2
        scores = torch.exp(-distance / (2 * self.SIGMA ** 2))
        probs = scores / scores.sum(dim=1, keepdim=True)
        return probs

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int = None,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy + Gaussian soft target KL loss.

        Loss = WeightedCE + SOFT_LOSS_WEIGHT * KL_Gaussian

        Args:
            model: The language model
            inputs: Input batch with labels
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch

        Returns:
            Loss tensor or (loss, outputs) tuple
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        batch_size, seq_len, vocab_size = shift_logits.shape
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_losses = loss_fct(flat_logits, flat_labels)

        weights = self._compute_token_weights(flat_labels)
        weighted_loss = token_losses * weights

        active_elements = (flat_labels != -100).sum()
        base_loss = weighted_loss.sum() / (active_elements + 1e-6)

        soft_loss = self._compute_soft_ordinal_loss(flat_logits, flat_labels)
        final_loss = base_loss + self.SOFT_LOSS_WEIGHT * soft_loss

        return (final_loss, outputs) if return_outputs else final_loss

    def _compute_token_weights(self, flat_labels: torch.Tensor) -> torch.Tensor:
        """Compute weights for each token (high weight for minus tokens)."""
        weights = torch.ones_like(flat_labels, dtype=torch.float32)
        for token_id in self.minus_token_ids:
            weights[flat_labels == token_id] = self.MINUS_TOKEN_WEIGHT
        return weights

    def _compute_soft_ordinal_loss(
        self,
        flat_logits: torch.Tensor,
        flat_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss with Gaussian soft targets for ordinal tokens."""
        ordinal_mask = torch.zeros_like(flat_labels, dtype=torch.bool)
        ordinal_gt_values = torch.zeros_like(flat_labels, dtype=torch.long)

        for token_id, value in self.id_to_value.items():
            is_digit = flat_labels == token_id
            if is_digit.any():
                ordinal_mask |= is_digit
                ordinal_gt_values[is_digit] = value

        if not ordinal_mask.any():
            return torch.tensor(0.0, device=flat_logits.device)

        digit_ids_tensor = torch.tensor(
            self.digit_canonical_ids, device=flat_logits.device
        )
        subset_logits = flat_logits[ordinal_mask][:, digit_ids_tensor]
        subset_log_probs = F.log_softmax(subset_logits, dim=-1)
        subset_gt = ordinal_gt_values[ordinal_mask]

        soft_targets = self._generate_gaussian_target(
            subset_gt, num_classes=self.NUM_CLASSES
        )
        kl_loss = F.kl_div(subset_log_probs, soft_targets, reduction="batchmean")
        return kl_loss

    def save_model(
        self, output_dir: str = None, _internal_call: bool = False
    ) -> None:
        """Save LoRA adapters and frozen visual weights."""
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.is_world_process_zero():
            print(f"Saving checkpoint to {output_dir}...")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            peft_model = unwrapped_model.language_model
            peft_model.save_pretrained(output_dir)

            self._save_stage1_weights(unwrapped_model, output_dir)
            self._save_tokenizer(output_dir)
            peft_model.config.save_pretrained(output_dir)
            print("Checkpoint saved: LoRA + Stage1 weights included.")

    def _save_stage1_weights(self, unwrapped_model: nn.Module, output_dir: str) -> None:
        """Save frozen visual backbone weights."""
        stage1_weights = {}
        for name, param in unwrapped_model.named_parameters():
            if "language_model" not in name:
                stage1_weights[name] = param.cpu()
        torch.save(
            stage1_weights, os.path.join(output_dir, "stage1_visual_weights.pth")
        )

    def _save_tokenizer(self, output_dir: str) -> None:
        """Save processor or tokenizer."""
        saver = getattr(self, "processing_class", None) or getattr(
            self, "tokenizer", None
        )
        if saver:
            saver.save_pretrained(output_dir)

def load_model_and_processor(
    cfg: Dict[str, Any],
) -> Tuple[RvlnMultiTask, InstructBlipProcessor]:
    """Load base model and processor with stage 1 weights.

    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple of (model, processor)
    """
    print("Loading processor...")
    processor = InstructBlipProcessor.from_pretrained(cfg["model_name_or_path"])
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens_dict = {"additional_special_tokens": ["<history>", "<current>"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    history_token_id = tokenizer.convert_tokens_to_ids("<history>")
    current_token_id = tokenizer.convert_tokens_to_ids("<current>")

    print("Loading base model...")
    config = InstructBlipConfig.from_pretrained(cfg["model_name_or_path"])
    config.history_token_id = history_token_id
    config.current_token_id = current_token_id
    config.depth_model_name_or_path = cfg["depth_encoder_path"]

    model = RvlnMultiTask.from_pretrained(
        cfg["model_name_or_path"],
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model.language_model.resize_token_embeddings(len(tokenizer))

    _load_stage1_checkpoint(model, cfg["stage1_checkpoint"])
    return model, processor


def _load_stage1_checkpoint(model: RvlnMultiTask, checkpoint_path: str) -> None:
    """Load frozen stage 1 visual weights."""
    if not os.path.exists(checkpoint_path):
        print("Warning: Stage 1 checkpoint not found! Training from scratch.")
        return

    print(f"Loading stage 1 checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    msg = model.load_state_dict(ckpt, strict=False)
    print(f"Checkpoint load status: {msg}")


def setup_lora_model(
    model: RvlnMultiTask, cfg: Dict[str, Any]
) -> RvlnMultiTask:
    """Freeze base model and apply LoRA to LLM."""
    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"],
    )

    print("Applying LoRA to LLM...")
    model.language_model = get_peft_model(model.language_model, peft_config)
    print_trainable_parameters(model)
    return model


def create_training_arguments(cfg: Dict[str, Any]) -> TrainingArguments:
    """Create training arguments with distributed settings."""
    return TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=0.03,
        num_train_epochs=cfg["num_epochs"],
        fp16=False,
        bf16=True,
        deepspeed="./config/ds_config_zero2_1.json",
        remove_unused_columns=False,
        report_to="none",
        evaluation_strategy="steps",
        eval_steps=cfg["eval_steps"],
        per_device_eval_batch_size=cfg["batch_size"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=cfg["logging_steps"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        dataloader_pin_memory=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def main() -> None:
    """Main training loop."""
    model, processor = load_model_and_processor(CONFIG)
    tokenizer = processor.tokenizer
    model = setup_lora_model(model, CONFIG)

    print("Loading full dataset...")
    full_dataset = RvlnLoRADataset(
        data_path=CONFIG["data_path"],
        processor=processor,
        tokenizer=tokenizer,
        image_root="",
        history_len=4,
        current_len=1,
    )

    val_size = int(len(full_dataset) * CONFIG["val_ratio"])
    train_size = len(full_dataset) - val_size

    print(
        f"Splitting dataset: Total={len(full_dataset)} | "
        f"Train={train_size} | Val={val_size}"
    )
    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    collator = DataCollatorForRvln(processor=processor, tokenizer=tokenizer)
    training_args = create_training_arguments(CONFIG)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print("Starting training...")
    trainer.train()
    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        _save_final_checkpoint(trainer, CONFIG["output_dir"])


def _save_final_checkpoint(trainer: WeightedTrainer, output_dir: str) -> None:
    """Save final merged checkpoint with LoRA adapters and visual weights."""
    print("Saving final checkpoint...")
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)

    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    peft_model = unwrapped_model.language_model

    print(f"   - Saving LoRA adapters to {final_adapter_dir}...")
    peft_model.save_pretrained(final_adapter_dir)

    print("   - Saving stage 1 frozen weights (backup)...")
    stage1_weights = {}
    for name, param in unwrapped_model.named_parameters():
        if "language_model" not in name:
            stage1_weights[name] = param.cpu()
    torch.save(
        stage1_weights, os.path.join(final_adapter_dir, "stage1_visual_weights.pth")
    )

    print("   - Saving processor (tokenizer + image config)...")
    processor = trainer.processing_class or trainer.tokenizer
    if processor:
        processor.save_pretrained(final_adapter_dir)

    print(f"Save complete! Checkpoint saved to: {final_adapter_dir}")

if __name__ == "__main__":
    main()
