import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import InstructBlipProcessor, BertTokenizer
from InstructBlip import InstructBlipMultiTask
import swanlab
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
    args = {
        "model_name": "./instructblip-vicuna-7b",
        "data_root": "./flickr_30k",
        "batch_size": 32,
        "lr": 5e-5,
        "epochs": 10,
        "load_in_8bit": False,
        "fusion_bias": -3.0 # 记录一下你的特殊初始化参数
    }
    
    # <--- 【SwanLab 新增】2. 初始化实验 ---
    swanlab.init(
        project="InstructBlip-DualTower", # 项目名
        experiment_name="full-finetune-v1", # 实验名
        config=args, # 记录超参数
        description="Training ITM head + Visual Fusion module with Depth Anything V2"
    )
    # --- 1. 配置路径 ---
    MODEL_NAME = "./instructblip-vicuna-7b" 
    DATA_ROOT = "./flickr_30k"
    IMAGE_ROOT = os.path.join(DATA_ROOT, "flickr30k-images")
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions_clean.token")
    CHECKPOINT_DIR = "./checkpoints_itm_fusion" # 改个名区分一下
    # RESUME_PATH = "./checkpoints_itm_fusion/checkpoint_step_10500.pth" 
    RESUME_PATH = ""  # 不加载，重新训练
    # --- 2. 显存与精度设置 ---
    LOAD_IN_8BIT = False  # 显存<24G 时建议开启
    BATCH_SIZE = 32      # 融合层增加了计算量，可能需要稍微调小 Batch Size
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Loading Processor from {MODEL_NAME}...")
    processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)
    # Q-Former 必须使用 BERT Tokenizer (这是 InstructBLIP 的硬性要求)
    qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("Loading Dual-Tower Model...")
    # 加载我们自定义的双塔模型
    model = InstructBlipMultiTask.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16, 
        load_in_8bit=LOAD_IN_8BIT,
        device_map="auto" if LOAD_IN_8BIT else None
    )
    
    if not LOAD_IN_8BIT:
        model.to(device)

    # --- 3. 【核心修改】参数冻结与解冻 ---
    print("Configuring trainable parameters...")
    
    trainable_modules = ["itm_head", "visual_fusion"] # 我们要训练的两个模块
    
    for name, param in model.named_parameters():
        # 检查参数名是否包含我們要训练的模块名
        is_trainable = any(module_name in name for module_name in trainable_modules)
        
        if is_trainable:
            param.requires_grad = True
            # 【重要】训练的层建议转回 FP32，防止 Loss NaN 或梯度下溢
            param.data = param.data.to(torch.bfloat16) 
            print(f"  -> Unfrozen: {name}") 
        else:
            param.requires_grad = False
            
    # 计算参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {trainable_params / 1e6:.2f} M")
    
    if os.path.exists(RESUME_PATH):
        print(f"🔄正在加载权重: {RESUME_PATH} ...")
        
        # 1. 读取文件
        checkpoint = torch.load(RESUME_PATH, map_location=device)
        
        # 2. 分别加载 visual_fusion 和 itm_head
        # 注意：因为我们保存的是个字典 {'visual_fusion': ..., 'itm_head': ...}
        # 所以不能直接 model.load_state_dict(checkpoint)
        
        try:
            model.visual_fusion.load_state_dict(checkpoint['visual_fusion'])
            print("  ✅ Visual Fusion 权重加载成功")
        except KeyError:
            print("  ⚠️ 警告: Checkpoint 中未找到 visual_fusion")
            
        try:
            model.itm_head.load_state_dict(checkpoint['itm_head'])
            print("  ✅ ITM Head 权重加载成功")
        except KeyError:
            print("  ⚠️ 警告: Checkpoint 中未找到 itm_head")
            
        print("🚀 权重加载完毕，准备继续训练！")
    else:
        print(f"⚠️ 未找到路径 {RESUME_PATH}，将从头开始训练！")
    # --- 5. 优化器 ---
    # 只传入 requires_grad=True 的参数
    # 修改优化器定义
    fusion_params = list(map(id, model.visual_fusion.parameters()))
    base_params = filter(lambda p: id(p) not in fusion_params and p.requires_grad, model.parameters())

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 5e-5}, # Head 保持小 LR
        {'params': model.visual_fusion.parameters(), 'lr': 5e-4} # Fusion 层大 LR (放大10倍)
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # 数据加载
    dataset = Flickr30kDataset(IMAGE_ROOT, CAPTION_FILE)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, # 适当降低 worker 防止内存爆炸
        collate_fn=collate_fn 
    )

    print("Start Dual-Tower ITM training...")
    
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    save_every_steps = 500
    global_step = 0
    
    model.train() 

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        steps_in_epoch = 0
        
        for step, (images, captions, image_names) in enumerate(dataloader):
            # A. 构造正负样本 (Batch Size * 2)
            itm_images_pil, itm_texts, itm_labels = create_itm_batch(
                images, captions, image_names, dataset
            )
            itm_labels = itm_labels.to(device)
            
            # B. 数据预处理
            # 图片 -> RGB Tensor
            image_inputs = processor(
                images=itm_images_pil,
                return_tensors="pt"
            ).to(device)
            
            # 文本 -> Q-Former Token IDs
            text_inputs = qformer_tokenizer(
                itm_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32 
            ).to(device)
            
            # C. 前向传播
            optimizer.zero_grad()
            
            # 调用 forward_itm (内部会自动调用 Depth backbone 和 Fusion)
            logits = model.forward_itm(
                pixel_values=image_inputs.pixel_values.to(dtype=torch.bfloat16),
                input_ids=text_inputs.input_ids,         
                attention_mask=text_inputs.attention_mask 
            )
            
            loss = criterion(logits, itm_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # print(f"Gate Bias Grad: {model.visual_fusion.gate_net[-2].bias.grad}")
            optimizer.step()
            
            # D. 统计与日志
            preds = logits.argmax(dim=1)
            acc = (logits.argmax(dim=1) == itm_labels).float().mean().item()
            loss_val = loss.item()
            
            epoch_loss += loss_val
            epoch_acc += acc
            steps_in_epoch += 1
            global_step += 1
            swanlab.log({
                            "train/loss": loss_val,
                            "train/acc": acc,
                            "train/lr": optimizer.param_groups[0]['lr']
                        })
            if step % 100 == 0:
                # 取 Batch 里的第一张图做展示
                # 记录：原始图片 + 文本 + 真实标签 + 预测标签
                log_image = swanlab.Image(
                    itm_images_pil[0], 
                    caption=f"Text: {itm_texts[0]} | GT: {itm_labels[0]} | Pred: {preds[0].item()}"
                )
                swanlab.log({"val/visualization": log_image})
            if step % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(dataloader)}], "
                      f"Loss: {loss_val:.4f}, Acc: {acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # 监控 Gate 的值 (可选，调试用)
                # 我们可以看看 Gate 是否从 0 开始逐渐变大
                with torch.no_grad():
                   print(f"  Sample Gate Value: {model.visual_fusion.gate_net[-2].bias.data[0]:.4f} (Bias)")

            # --- 6. 【核心修改】保存逻辑 ---
            if global_step % save_every_steps == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{global_step}.pth")
                
                # 我们需要保存两个部分：Fusion Layer 和 ITM Head
                save_dict = {
                    "visual_fusion": model.visual_fusion.state_dict(),
                    "itm_head": model.itm_head.state_dict()
                }
                torch.save(save_dict, ckpt_path)
                print(f"Checkpoint saved -> {ckpt_path}")

        scheduler.step()
        
        avg_loss = epoch_loss / steps_in_epoch
        avg_acc = epoch_acc / steps_in_epoch
        print(f"=== Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f} ===")

    # 保存最终结果
    final_path = os.path.join(CHECKPOINT_DIR, "final_dual_tower.pth")
    save_dict = {
        "visual_fusion": model.visual_fusion.state_dict(),
        "itm_head": model.itm_head.state_dict()
    }
    torch.save(save_dict, final_path)
    print(f"Training Done. Final weights saved to {final_path}")