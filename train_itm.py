import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import InstructBlipProcessor, BertTokenizer
from InstructBlip import InstructBlipMultiTask
# ==============================================================================
# 2. Dataset 定义 (修改为返回 PIL)
# ==============================================================================
class Flickr30kDataset(Dataset):
    def __init__(self, image_root, caption_file):
        self.image_root = image_root
        self.samples = []
        
        print("正在加载数据集索引...")
        self.image_to_captions = {}
        with open(caption_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) < 2: continue # 跳过坏行
                name_and_id, caption = parts
                image_name = name_and_id.split("#")[0]
                
                if image_name not in self.image_to_captions:
                    self.image_to_captions[image_name] = []
                self.image_to_captions[image_name].append(caption)
        
        for image_name, captions in self.image_to_captions.items():
            for caption in captions:
                self.samples.append((image_name, caption))
        
        self.image_names = list(self.image_to_captions.keys())
        print(f"数据集加载完成，共 {len(self.samples)} 个样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]
        image_path = os.path.join(self.image_root, image_name)
        
        # --- 改动点：直接返回 PIL Image ---
        # InstructBlipProcessor 会处理 Resize 和 Normalize，不要在这里转 Tensor
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个纯黑图片防止崩溃
            img = Image.new('RGB', (224, 224), color='black')
            
        return img, caption, image_name

def collate_fn(batch):
    # 简单的 list 打包，不做 tensor 转换，因为后面要做负采样
    images, captions, image_names = zip(*batch)
    return list(images), list(captions), list(image_names)

# ==============================================================================
# 3. 负采样逻辑 (Batch 构造)
# ==============================================================================
def create_itm_batch(images, captions, image_names, dataset):
    """
    修正版：构造真正的负样本 (Image A + Text B)
    """
    batch_size = len(images)
    
    # --- 1. 正样本 (Image A + Text A) ---
    positive_images = list(images) 
    positive_texts = list(captions)
    positive_labels = [1] * batch_size
    
    # --- 2. 负样本 (Image A + Text B) ---
    # 策略：图片还是这批图片，但是文字换成别人的
    negative_images = list(images) # 图片不变 (Image A)
    negative_texts = [] # 准备填入错误的文字 (Text B)
    
    for i in range(batch_size):
        current_image_name = image_names[i]
        
        # 死循环直到找到一个“别人的”文字
        while True:
            # 随机从数据集里抽一个索引
            random_idx = random.randint(0, len(dataset.samples) - 1)
            other_image_name, other_caption = dataset.samples[random_idx]
            
            # 只要这张图的名字和当前图不一样，那它的文字就是“错误的”
            if other_image_name != current_image_name:
                negative_texts.append(other_caption)
                break
    
    negative_labels = [0] * batch_size
    
    # --- 3. 合并 ---
    all_images = positive_images + negative_images
    all_texts = positive_texts + negative_texts
    all_labels = positive_labels + negative_labels
    
    # --- 4. 打乱 ---
    combined = list(zip(all_images, all_texts, all_labels))
    random.shuffle(combined)
    
    all_images, all_texts, all_labels = zip(*combined)
    
    return list(all_images), list(all_texts), torch.tensor(all_labels, dtype=torch.long)
# ==============================================================================
# 4. 主训练循环
# ==============================================================================
if __name__ == "__main__":
    # 配置路径
    MODEL_NAME = "./instructblip-vicuna-7b" # 或者你的本地路径
    DATA_ROOT = "./flickr_30k"
    IMAGE_ROOT = os.path.join(DATA_ROOT, "flickr30k-images")
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions_clean.token")
    CHECKPOINT_DIR = "./checkpoints_itm_instructblip"
    
    # 显存优化参数
    # 如果显存不够 (如 < 24G)，建议开启 load_in_8bit=True (需要安装 bitsandbytes)
    LOAD_IN_8BIT = False 
    BATCH_SIZE = 64 # 根据显存调整，InstructBlip 比 Qwen2VL 稍大
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading Processor...")
    processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)

    print("Loading Model...")
    
    print("Loading Q-Former Tokenizer (BERT)...")
    qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # 加载自定义模型
    model = InstructBlipMultiTask.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, # 强制 fp16 节省显存
        load_in_8bit=LOAD_IN_8BIT,
        device_map="auto" if LOAD_IN_8BIT else None
    )
    
    if not LOAD_IN_8BIT:
        model.to(device)

    # --- 冻结参数 ---
    print("Freezing parameters...")
    for name, param in model.named_parameters():
        if "itm_head" in name:
            param.requires_grad = True
            # Head 建议用 fp32 训练以保证稳定，或者保持 fp16
            param.data = param.data.to(torch.float32) 
        else:
            param.requires_grad = False
            
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters (ITM Head): {trainable_params}")

    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # 数据加载
    dataset = Flickr30kDataset(IMAGE_ROOT, CAPTION_FILE)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=16,
        collate_fn=collate_fn # 使用新的 collate
    )

    print("Start ITM training...")
    
    criterion = nn.CrossEntropyLoss()
    num_epochs = 100
    save_every_steps = 500
    global_step = 0
    
    loss_list = []
    
    model.train() # 开启 Dropout 等

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        steps_in_epoch = 0
        
        for step, (images, captions, image_names) in enumerate(dataloader):
            # 1. 构造正负样本
            itm_images_pil, itm_texts, itm_labels = create_itm_batch(
                images, captions, image_names, dataset
            )
            itm_labels = itm_labels.to(device)
            
            # 2. 【修改部分】分开处理图片和文本
            
            # A. 处理图片 (使用原版 Processor，只传 images)
            image_inputs = processor(
                images=itm_images_pil,
                return_tensors="pt"
            ).to(device)
            
            # B. 处理文本 (使用 BERT Tokenizer)
            # 这生成的才是 Q-Former 能看懂的 IDs (0-30522)
            text_inputs = qformer_tokenizer(
                itm_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32 # ITM 文本通常不长，Q-Former 也不需要太长
            ).to(device)
            
            # 3. 前向传播
            optimizer.zero_grad()
            
            # 这里的 input_ids 来自 BERT tokenizer，绝对安全
            logits = model.forward_itm(
                pixel_values=image_inputs.pixel_values.to(dtype=torch.float16),
                input_ids=text_inputs.input_ids,         # <--- 使用 BERT 的 ID
                attention_mask=text_inputs.attention_mask # <--- 使用 BERT 的 Mask
            )
            
            loss = criterion(logits, itm_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 4. 统计
            acc = (logits.argmax(dim=1) == itm_labels).float().mean().item()
            loss_val = loss.item()
            
            epoch_loss += loss_val
            epoch_acc += acc
            steps_in_epoch += 1
            global_step += 1
            loss_list.append(loss_val)

            if step % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(dataloader)}], "
                      f"Loss: {loss_val:.4f}, Acc: {acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            if global_step % save_every_steps == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{global_step}.pth")
                # 只保存 head 的权重，极大节省空间
                torch.save(model.itm_head.state_dict(), ckpt_path)
                print(f"Checkpoint saved -> {ckpt_path}")

        scheduler.step()
        
        avg_loss = epoch_loss / steps_in_epoch
        avg_acc = epoch_acc / steps_in_epoch
        print(f"=== Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f} ===")

    # 保存最终结果
    final_path = os.path.join(CHECKPOINT_DIR, "final_itm_head.pth")
    torch.save(model.itm_head.state_dict(), final_path)
    print(f"Training Done. Final head saved to {final_path}")