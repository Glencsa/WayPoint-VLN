import torch
from peft import PeftModel
from models.rvln import RvlnMultiTask
from transformers import InstructBlipProcessor

# 1. 路径
base_model_path = "lora_weight/rvln_sft_llm" # 原始底座
lora_model_path = "lora_weight/rvln_sft_llm"    # 你训练出的 Output (包含 adapter)

# 2. 加载底座
print("Loading Base Model...")
base_model = RvlnMultiTask.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. 加载 LoRA
print("Loading LoRA Adapter...")
# 你的情况特殊：如果你的 output 已经是全量模型，这里可能会报错。
# 只有当你按我之前教的方法，重新保存出只有几百 MB 的 adapter 文件夹后，才能用这步。
model = PeftModel.from_pretrained(base_model.language_model, lora_model_path)

# 4. 【关键步骤】合并并卸载 LoRA
print("Merging weights...")
model = model.merge_and_unload()

# 此时 model 已经是一个普通的 torch.nn.Module，没有 peft 结构了
# 所有的 lora_A, lora_B 都消失了，它们的值被加到了 base_model.weight 里

# 5. 保存最终的完整模型
save_path = "./lora_weight/rvln_final_merged"
model.save_pretrained(save_path)
# 别忘了保存 processor/tokenizer
processor = InstructBlipProcessor.from_pretrained(base_model_path)
processor.save_pretrained(save_path)

print(f"✅ 模型合并完成！已保存至 {save_path}")