import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from data_utils import InstructBlipLoRADataset, DataCollatorForInstructBlip
# ==========================================
# 1. Mock 组件 (用于模拟 Processor 和 Tokenizer)
# ==========================================
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
    
    def __call__(self, text, padding=True, return_tensors="pt", truncation=True, max_length=512, add_special_tokens=True):
        if isinstance(text, str): text = [text]
        batch_size = len(text)
        # 模拟生成 input_ids
        return type('obj', (object,), {
            'input_ids': torch.randint(1, 1000, (batch_size, 30)), 
            'attention_mask': torch.ones((batch_size, 30), dtype=torch.long)
        })

class MockProcessor:
    def __call__(self, images, return_tensors="pt"):
        # images 是 PIL Image 列表
        n_images = len(images)
        # 模拟返回 [N, 3, 224, 224]
        return type('obj', (object,), {
            'pixel_values': torch.randn(n_images, 3, 224, 224)
        })


def main():
    # 配置你的真实文件路径
    json_path = "dataset_waypoint/rgb_images_r2r_train_processed.json"
    
    # ⚠️ 关键：如果 json 里的路径已经是 "datasets/test/..."
    # 且你的脚本在根目录，那么 image_root 应该是 "." (当前目录)
    image_root = "." 

    if not os.path.exists(json_path):
        print(f"❌ 错误：找不到文件 {json_path}")
        return

    # 初始化 Mock
    mock_processor = MockProcessor()
    mock_tokenizer = MockTokenizer()

    # 初始化 Dataset
    print("-" * 30)
    dataset = InstructBlipLoRADataset(
        data_path=json_path,
        processor=mock_processor,
        tokenizer=mock_tokenizer,
        image_root=image_root
    )
    print(f"✅ Dataset 加载成功，共有 {len(dataset)} 条数据")

    # --- Test 1: 读取第一条数据，检查图片路径 ---
    print("\n[Test 1] 正在读取第 1 条数据...")
    try:
        item = dataset[0]
        print("   RGB Tensor Shape:", item['pixel_values_rgb'].shape)
        print("   Depth Tensor Shape:", item['pixel_values_depth'].shape)
        
        # 检查形状是否为 [5, 3, 224, 224]
        if list(item['pixel_values_rgb'].shape) == [5, 3, 224, 224]:
            print("   ✅ Tensor 形状正确 (5帧, 3通道)")
        else:
            print("   ❌ Tensor 形状异常！")

    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()

    # --- Test 2: DataLoader 批处理 ---
    print("\n[Test 2] 正在测试 DataLoader (Batch Size=2)...")
    collator = DataCollatorForInstructBlip(mock_processor, mock_tokenizer, mock_tokenizer)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

    try:
        batch = next(iter(loader))
        print("   Batch RGB Shape:", batch['pixel_values_rgb'].shape)
        print("   Batch Depth Shape:", batch['pixel_values_depth'].shape)
        print("   Batch Labels Shape:", batch['labels'].shape)
        print("   ✅ DataLoader 测试通过！")
    except Exception as e:
        print(f"   ❌ DataLoader 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()