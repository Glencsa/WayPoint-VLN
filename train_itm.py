import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import InstructBlipProcessor, BertTokenizer,InstructBlipConfig
from transformers import AutoTokenizer
from models.WayPointVLN import RvlnMultiTask 
from models.depth_estimate import DepthEstimator
import swanlab


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
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.estimator = DepthEstimator(model_id="./Depth-Anything-V2-Small-hf", device="cpu")
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
        depth_image_jet = self.estimator.predict_depth(img, return_type="pil")
        return img, depth_image_jet, caption, image_name

def collate_fn(batch):
    images, depth_images, captions, image_names = zip(*batch)
    return list(images), list(depth_images), list(captions), list(image_names)

def create_itm_batch(images, depth_images, captions, image_names, dataset):
    """
    dataset:  Flickr30kDataset | torch.utils.data.Subset
    """
    batch_size = len(images)
    positive_images = list(images)
    positive_depth_images = list(depth_images)
    positive_texts = list(captions)
    positive_labels = [1] * batch_size

    negative_images = list(images)
    negative_depth_images = list(depth_images)
    negative_texts = []

    if isinstance(dataset, torch.utils.data.Subset):
        source_dataset = dataset.dataset  
        valid_indices = dataset.indices   
    else:
        source_dataset = dataset
        valid_indices = range(len(dataset)) 

    for i in range(batch_size):
        current_image_name = image_names[i]
        while True:
            random_idx = random.choice(valid_indices)
            other_image_name, other_caption = source_dataset.samples[random_idx]
            if other_image_name != current_image_name:
                negative_texts.append(other_caption)
                break
    
    negative_labels = [0] * batch_size
    all_images = positive_images + negative_images
    all_depth_images = positive_depth_images + negative_depth_images
    all_texts = positive_texts + negative_texts
    all_labels = positive_labels + negative_labels

    combined = list(zip(all_images, all_depth_images, all_texts, all_labels))
    random.shuffle(combined)

    all_images, all_depth_images, all_texts, all_labels = zip(*combined)
    return list(all_images), list(all_depth_images), list(all_texts), torch.tensor(all_labels, dtype=torch.long)

def validate(model, dataloader, val_dataset, device, processor, tokenizer, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0
    steps = 0
    
    print("\n[Validation] Starting evaluation on validation set...")
    with torch.no_grad():
        for step, (images, depth_images, captions, image_names) in enumerate(dataloader):
            itm_images_pil, itm_depth_images, itm_texts, itm_labels = create_itm_batch(
                images, depth_images, captions, image_names, val_dataset
            )
            itm_labels = itm_labels.to(device)
            image_inputs = processor(images=itm_images_pil, return_tensors="pt").to(device)
            depth_inputs = processor(images=itm_depth_images, return_tensors="pt").to(device)
            text_inputs = tokenizer(
                itm_texts, return_tensors="pt", padding=True, truncation=True, max_length=32
            ).to(device)
            
            logits = model.forward_itm(
                pixel_values=image_inputs.pixel_values.to(dtype=torch.bfloat16),
                depth_pixel_values=depth_inputs.pixel_values.to(dtype=torch.bfloat16),
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
            
            loss = criterion(logits, itm_labels)
            acc = (logits.argmax(dim=1) == itm_labels).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            steps += 1
            
    avg_loss = total_loss / steps if steps > 0 else 0
    avg_acc = total_acc / steps if steps > 0 else 0
    print(f"[Validation] Done. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}\n")
    return avg_loss, avg_acc

if __name__ == "__main__":
    args = {
        "model_name": "./instructblip-vicuna-7b",
        "data_root": "./flickr_30k",
        "batch_size": 32,
        "lr_head": 5e-6,
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
    CHECKPOINT_DIR = "./checkpoints_itm_cross_attn_with_depth_qformer_vit_v1" 
    
    LOAD_IN_8BIT = False 
    BATCH_SIZE = 16     
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens = {"additional_special_tokens": ["<history>", "<current>"]}
    tokenizer.add_special_tokens(special_tokens)

    hist_id = tokenizer.convert_tokens_to_ids("<history>")
    curr_id = tokenizer.convert_tokens_to_ids("<current>")
    print(f"Token IDs injected: History={hist_id}, Current={curr_id}")

    config = InstructBlipConfig.from_pretrained(MODEL_NAME)


    config.history_token_id = hist_id
    config.current_token_id = curr_id
    print(f"Loading Processor from {MODEL_NAME}...")
    processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)
    qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("Loading Dual-Tower Model...")
    model = RvlnMultiTask.from_pretrained(
        MODEL_NAME,
        config=config,              
        torch_dtype=torch.bfloat16, 
        load_in_8bit=LOAD_IN_8BIT,
        device_map="auto" if LOAD_IN_8BIT else None
    )

    model.language_model.resize_token_embeddings(len(tokenizer))
    if not LOAD_IN_8BIT:
        model.to(device)

    trainable_modules = ["itm_head", "visual_fusion","qformer", "query_tokens", "depth_backbone"] 
    for name, param in model.named_parameters():
        if any(m in name for m in trainable_modules):
            param.requires_grad = True
            param.data = param.data.to(torch.bfloat16) 
        else:
            param.requires_grad = False
            
    fusion_params = list(map(id, model.visual_fusion.parameters()))
    base_params = filter(lambda p: id(p) not in fusion_params and p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args['lr_head']}, 
        {'params': model.visual_fusion.parameters(), 'lr': args['lr_fusion']} 
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    full_dataset = Flickr30kDataset(IMAGE_ROOT, CAPTION_FILE)
    
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    
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
    best_val_acc = 0.0 
    
    model.train() 

    for epoch in range(num_epochs):
        model.train()
        for step, (images, depth_images, captions, image_names) in enumerate(train_loader):
            itm_images_pil, itm_depth_images, itm_texts, itm_labels = create_itm_batch(
                images, depth_images, captions, image_names, train_dataset
            )
            itm_labels = itm_labels.to(device)
            
            image_inputs = processor(images=itm_images_pil, return_tensors="pt").to(device)
            depth_inputs = processor(images=itm_depth_images, return_tensors="pt").to(device)
            text_inputs = qformer_tokenizer(
                itm_texts, return_tensors="pt", padding=True, truncation=True, max_length=32 
            ).to(device)
            
            optimizer.zero_grad()
            logits = model.forward_itm(
                pixel_values=image_inputs.pixel_values.to(dtype=torch.bfloat16),
                depth_pixel_values=depth_inputs.pixel_values.to(dtype=torch.bfloat16),
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
            
            loss = criterion(logits, itm_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            

            fusion_grad_norm = 0.0
            for p in model.visual_fusion.parameters():
                if p.grad is not None:
                    fusion_grad_norm += p.grad.data.norm(2).item() ** 2
            fusion_grad_norm = fusion_grad_norm ** 0.5

            depth_grad_norm = 0.0
            for p in model.depth_backbone.parameters():
                if p.grad is not None:
                    depth_grad_norm += p.grad.data.norm(2).item() ** 2
            depth_grad_norm = depth_grad_norm ** 0.5


            qformer_grad_norm = 0.0
            for p in model.qformer.parameters():
                if p.grad is not None:
                    qformer_grad_norm += p.grad.data.norm(2).item() ** 2
            qformer_grad_norm = qformer_grad_norm ** 0.5


            if model.query_tokens.grad is not None:
                tokens_grad_norm = model.query_tokens.grad.data.norm(2).item()
            else:
                tokens_grad_norm = 0.0
            optimizer.step()
            
            acc = (logits.argmax(dim=1) == itm_labels).float().mean().item()
            loss_val = loss.item()
            global_step += 1
            
            # SwanLab Log (Train)
            swanlab.log({
                "train/loss": loss_val,
                "train/acc": acc,
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/fusion_grad_norm": fusion_grad_norm,
                "train/depth_grad_norm": depth_grad_norm,
                "train/qformer_grad_norm": qformer_grad_norm,
                "train/tokens_grad_norm": tokens_grad_norm
            })
            
            if step % 5 == 0:
                print(f"[Epoch {epoch+1}][Step {step}] Train Loss: {loss_val:.4f}, Acc: {acc:.4f}")

            if global_step % save_every_steps == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
                torch.save({
                    "visual_fusion": model.visual_fusion.state_dict(),
                    "itm_head": model.itm_head.state_dict(),
                    "qformer": model.qformer.state_dict(),
                    "query_tokens": model.query_tokens,
                    "depth_backbone": model.depth_backbone.state_dict()
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        scheduler.step()
        
        val_loss, val_acc = validate(model, val_loader, val_dataset, device, processor, qformer_tokenizer, criterion)

        # SwanLab Log (Validation)
        swanlab.log({
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/epoch": epoch + 1
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")
            torch.save({
                "epoch": epoch,
                "val_acc": val_acc,
                "visual_fusion": model.visual_fusion.state_dict(),
                "itm_head": model.itm_head.state_dict(),
                "qformer": model.qformer.state_dict(),
                "query_tokens": model.query_tokens,
                "depth_backbone": model.depth_backbone.state_dict()
            }, best_path)
            print(f"🏆 New Best Model Saved! Val Acc: {val_acc:.4f}")

    print(f"Training Done. Best Validation Accuracy: {best_val_acc:.4f}")