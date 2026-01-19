import os
import torch
from transformers import InstructBlipProcessor, InstructBlipConfig
from peft import PeftModel
from models.WayPointVLN import RvlnMultiTask


def merge_lora():
    base_model_path = "./instructblip-vicuna-7b"
    adapter_path = "./output_116/final_adapter"
    stage1_weights_path = "output/stage1_checkpoint/latest_checkpoint.pth"
    depth_encoder_path = "./vit-base-patch16-224"
    output_path = "./output/rvln_merged_final_116"

    print("Starting merge process...")
    print(f" -> Base: {base_model_path}")
    print(f" -> Adapter: {adapter_path}")
    print(f" -> Output: {output_path}")

    print("Loading Processor & Config...")
    processor = InstructBlipProcessor.from_pretrained(base_model_path)
    tokenizer = processor.tokenizer

    special_tokens_dict = {"additional_special_tokens": ["<history>", "<current>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f" -> Added {num_added_toks} special tokens. Vocab size: {len(tokenizer)}")

    config = InstructBlipConfig.from_pretrained(base_model_path)
    config.history_token_id = tokenizer.convert_tokens_to_ids("<history>")
    config.current_token_id = tokenizer.convert_tokens_to_ids("<current>")
    config.depth_model_name_or_path = depth_encoder_path

    print("Loading Base Model (RvlnMultiTask)...")
    model = RvlnMultiTask.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.float16,
    )
    model.language_model.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)

    if os.path.exists(stage1_weights_path):
        print(f"Loading Stage 1 Visual Weights from: {stage1_weights_path}")
        stage1_state_dict = torch.load(stage1_weights_path, map_location="cpu")
        msg = model.load_state_dict(stage1_state_dict, strict=False)
        print(f"   Load Status: {msg}")
    else:
        print("Warning: Stage 1 weights not found! The visual part will remain original/random.")

    print("Loading LoRA Adapter...")
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        adapter_path,
        torch_dtype=torch.float16,
    )

    print("Merging LoRA into Base Model...")
    model.language_model = model.language_model.merge_and_unload()

    print(f"Saving Merged Model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print("Merge Complete! You can now use the model directly without loading adapters.")

if __name__ == "__main__":
    merge_lora()