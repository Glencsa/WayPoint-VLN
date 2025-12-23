import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import InstructBlipProcessor, BertTokenizer,InstructBlipConfig
from transformers import AutoTokenizer
from models.InstructBlip import InstructBlipMultiTask 
import swanlab

# ==============================================================================
# 2. Dataset 定义 (保持不变)
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
                if len(parts) < 2: continue 
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
        
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            img = Image.new('RGB', (224, 224), color='black')
            
        return img, caption, image_name

def collate_fn(batch):
    images, captions, image_names = zip(*batch)
    return list(images), list(captions), list(image_names)

# ==============================================================================
# 3. 负采样逻辑 (保持不变)
# ==============================================================================
def create_itm_batch(images, captions, image_names, dataset):
    """
    dataset: 这里必须传入原始的 Flickr30kDataset 对象，
    因为 random_split 产生的 Subset 对象没有 image_names 属性。
    """
    batch_size = len(images)
    positive_images = list(images) 
    positive_texts = list(captions)
    positive_labels = [1] * batch_size
    
    negative_images = list(images) 
    negative_texts = [] 
    
    for i in range(batch_size):
        current_image_name = image_names[i]
        while True:
            # 从全局数据集中随机取负样本
            random_idx = random.randint(0, len(dataset.samples) - 1)
            other_image_name, other_caption = dataset.samples[random_idx]
            if other_image_name != current_image_name:
                negative_texts.append(other_caption)
                break
    
    negative_labels = [0] * batch_size
    all_images = positive_images + negative_images
    all_texts = positive_texts + negative_texts
    all_labels = positive_labels + negative_labels
    
    combined = list(zip(all_images, all_texts, all_labels))
    random.shuffle(combined)
    
    all_images, all_texts, all_labels = zip(*combined)
    return list(all_images), list(all_texts), torch.tensor(all_labels, dtype=torch.long)

# ==============================================================================
# [新增] 验证函数
# ==============================================================================
def validate(model, dataloader, full_dataset, device, processor, tokenizer, criterion):
    model.eval() # 切换到评估模式
    total_loss = 0
    total_acc = 0
    steps = 0
    
    print("\n[Validation] Starting evaluation on validation set...")
    with torch.no_grad(): # 禁用梯度计算
        for step, (images, captions, image_names) in enumerate(dataloader):
            # 1. 构造验证用的正负样本
            # 注意：传入 full_dataset 引用以获取负样本池
            itm_images_pil, itm_texts, itm_labels = create_itm_batch(
                images, captions, image_names, full_dataset
            )
            itm_labels = itm_labels.to(device)
            
            # 2. 预处理
            image_inputs = processor(images=itm_images_pil, return_tensors="pt").to(device)
            text_inputs = tokenizer(
                itm_texts, return_tensors="pt", padding=True, truncation=True, max_length=32
            ).to(device)
            
            # 3. 前向传播
            logits = model.forward_itm(
                pixel_values=image_inputs.pixel_values.to(dtype=torch.bfloat16),
                input_ids=text_inputs.input_ids,         
                attention_mask=text_inputs.attention_mask 
            )
            
            # 4. 计算指标
            loss = criterion(logits, itm_labels)
            acc = (logits.argmax(dim=1) == itm_labels).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            steps += 1
            
    avg_loss = total_loss / steps
    avg_acc = total_acc / steps
    print(f"[Validation] Done. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}\n")
    return avg_loss, avg_acc

# ==============================================================================
# 4. 主训练循环
# ==============================================================================
if __name__ == "__main__":
    args = {
        "model_name": "./instructblip-vicuna-7b",
        "data_root": "./flickr_30k",
        "batch_size": 32,
        "lr_head": 5e-5,
        "lr_fusion": 1e-4, 
        "epochs": 10,
        "load_in_8bit": False,
        "fusion_type": "CrossAttention"
    }
    
    swanlab.init(
        project="InstructBlip-DualTower", 
        experiment_name="full-finetune-with-val-v1", 
        config=args, 
        description="Training ITM with Validation Split"
    )

    MODEL_NAME = "./instructblip-vicuna-7b" 
    DATA_ROOT = "./flickr_30k"
    IMAGE_ROOT = os.path.join(DATA_ROOT, "flickr30k-images")
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions_clean.token")
    CHECKPOINT_DIR = "./checkpoints_itm_cross_attn" 
    
    LOAD_IN_8BIT = False 
    BATCH_SIZE = 32     
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 模型与处理器加载 ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens = {"additional_special_tokens": ["<history>", "<current>"]}
    tokenizer.add_special_tokens(special_tokens)

    hist_id = tokenizer.convert_tokens_to_ids("<history>")
    curr_id = tokenizer.convert_tokens_to_ids("<current>")
    print(f"Token IDs injected: History={hist_id}, Current={curr_id}")

    # 3. 加载 Config 对象
    config = InstructBlipConfig.from_pretrained(MODEL_NAME)

    # 4. 【关键步骤】将 ID 手动写入 Config 对象
    # 这一步必须在 model 初始化之前完成！
    config.history_token_id = hist_id
    config.current_token_id = curr_id
    print(f"Loading Processor from {MODEL_NAME}...")
    processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)
    qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("Loading Dual-Tower Model...")
    model = InstructBlipMultiTask.from_pretrained(
        MODEL_NAME,
        config=config,              
        torch_dtype=torch.bfloat16, 
        load_in_8bit=LOAD_IN_8BIT,
        device_map="auto" if LOAD_IN_8BIT else None
    )

    model.language_model.resize_token_embeddings(len(tokenizer))
    if not LOAD_IN_8BIT:
        model.to(device)

    # --- 冻结/解冻参数 ---
    trainable_modules = ["itm_head", "visual_fusion"] 
    for name, param in model.named_parameters():
        if any(m in name for m in trainable_modules):
            param.requires_grad = True
            param.data = param.data.to(torch.bfloat16) 
        else:
            param.requires_grad = False
            
    # --- 优化器 ---
    fusion_params = list(map(id, model.visual_fusion.parameters()))
    base_params = filter(lambda p: id(p) not in fusion_params and p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args['lr_head']}, 
        {'params': model.visual_fusion.parameters(), 'lr': args['lr_fusion']} 
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # --- 【核心修改】数据划分 ---
    full_dataset = Flickr30kDataset(IMAGE_ROOT, CAPTION_FILE)
    
    # 90% 训练, 10% 验证
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    
    # 固定种子，保证每次划分一致
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset Split -> Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn, drop_last=False)

    print("Start Training...")
    
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    save_every_steps = 500
    global_step = 0
    best_val_acc = 0.0 # 记录历史最佳
    
    model.train() 

    for epoch in range(num_epochs):
        # 1. 训练阶段
        model.train()
        for step, (images, captions, image_names) in enumerate(train_loader):
            # 这里的 full_dataset 用于提供负样本池
            itm_images_pil, itm_texts, itm_labels = create_itm_batch(
                images, captions, image_names, full_dataset
            )
            itm_labels = itm_labels.to(device)
            
            image_inputs = processor(images=itm_images_pil, return_tensors="pt").to(device)
            text_inputs = qformer_tokenizer(
                itm_texts, return_tensors="pt", padding=True, truncation=True, max_length=32 
            ).to(device)
            
            optimizer.zero_grad()
            logits = model.forward_itm(
                pixel_values=image_inputs.pixel_values.to(dtype=torch.bfloat16),
                input_ids=text_inputs.input_ids,         
                attention_mask=text_inputs.attention_mask 
            )
            
            loss = criterion(logits, itm_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 计算梯度范数
            fusion_grad_norm = 0.0
            for p in model.visual_fusion.parameters():
                if p.grad is not None:
                    fusion_grad_norm += p.grad.data.norm(2).item() ** 2
            fusion_grad_norm = fusion_grad_norm ** 0.5

            optimizer.step()
            
            # 记录日志
            acc = (logits.argmax(dim=1) == itm_labels).float().mean().item()
            loss_val = loss.item()
            global_step += 1
            
            # SwanLab Log (Train)
            swanlab.log({
                "train/loss": loss_val,
                "train/acc": acc,
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/fusion_grad_norm": fusion_grad_norm
            })
            
            if step % 50 == 0:
                print(f"[Epoch {epoch+1}][Step {step}] Train Loss: {loss_val:.4f}, Acc: {acc:.4f}")

            # 定期保存普通 Checkpoint
            if global_step % save_every_steps == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
                torch.save({
                    "visual_fusion": model.visual_fusion.state_dict(),
                    "itm_head": model.itm_head.state_dict()
                }, ckpt_path)

        scheduler.step()
        
        # 2. 验证阶段 (每个 Epoch 结束后)
        val_loss, val_acc = validate(model, val_loader, full_dataset, device, processor, qformer_tokenizer, criterion)
        
        # SwanLab Log (Validation)
        swanlab.log({
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/epoch": epoch + 1
        })
        
        # 3. 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")
            torch.save({
                "epoch": epoch,
                "val_acc": val_acc,
                "visual_fusion": model.visual_fusion.state_dict(),
                "itm_head": model.itm_head.state_dict()
            }, best_path)
            print(f"🏆 New Best Model Saved! Val Acc: {val_acc:.4f}")

    print(f"Training Done. Best Validation Accuracy: {best_val_acc:.4f}")