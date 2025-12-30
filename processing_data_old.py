import json
import os
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================

INPUT_JSON_PATH = 'datasets/rgb_images_r2r_train.json'                # 原始 JSON 文件
OUTPUT_JSON_PATH = 'datasets/filtered_traj_3279.json' # 输出 JSON 文件
PADDING_IMAGE_NAME = 'datasets/black.jpg'              # 填充用的黑图文件名

# 目标轨迹 ID
TARGET_TRAJ = "traj_3279"

# 路径前缀配置 (上一轮的逻辑)
RGB_PREFIX_NEW = "datasets/test/rgb/ep_4991/traj_3279"
DEPTH_PREFIX_NEW = "datasets/test/depth/ep_4991/traj_3279"
# ===========================================

def create_black_padding_image(path):
    """创建填充用的全黑图片"""
    if not os.path.exists(path):
        print(f"正在创建填充用全黑图片: {path}")
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        img.save(path)

def normalize_list_length(item_list, padding_value):
    """
    通用列表处理：将列表转换为固定 5 个元素 [H, H, H, H, C]
    适用于 images 和 depth_images
    """
    if not item_list:
        return None
        
    current_item = item_list[-1]
    raw_history = item_list[:-1]
    
    # 目标历史帧数量 = 4
    target_hist_len = 4
    
    if len(raw_history) >= target_hist_len:
        # 截断：取最后4帧
        final_history = raw_history[-target_hist_len:]
    else:
        # 填充：前面补 padding_value
        num_padding = target_hist_len - len(raw_history)
        final_history = [padding_value] * num_padding + raw_history
        
    return final_history + [current_item]

def process_human_text(text):
    """
    处理 Human 输入：替换为指定的 Prompt 模板
    """
    new_suffix = (
        "You are provided with:\n"
        "- Historical observations(four images): <history> \n"
        "- Current observation: <current>, there are some routes on the current observation.\n\n"
        "Your task is to select the best route number based on these routes, or return zero to Stop. \n"
        " The format of the result is {'Route': number 0~3}"
    )

    # 如果原文本里有 split_marker，则保留前半部分；否则直接用新模板
    split_marker = "You are provided with:"
    
    if split_marker in text:
        prefix = text.split(split_marker)[0]
        return prefix + new_suffix
    else:
        return new_suffix

def process_gpt_response(text):
    """
    处理 GPT 输出：将 A/B/C/D 映射为 2/1/3/0
    """
    if not text:
        return text

    clean_text = text.strip()
    if not clean_text:
        return text
        
    first_char = clean_text[0].upper()

    # 你的代码中指定的映射关系
    mapping = {
        'A': 2,
        'B': 1,
        'C': 3,
        'D': 0
    }

    if first_char in mapping:
        route_num = mapping[first_char]
        return f"{{'Route': {route_num}}}"
    else:
        # 如果不是标准选项，保留原样或打印警告
        return text

def main():
    # 1. 准备填充图片
    create_black_padding_image(PADDING_IMAGE_NAME)
    
    # 构造填充图片的完整路径（用于放入 json 列表）
    # 这里假设填充图和数据集在同一层级，或者你可以写绝对路径
    padding_rgb_path = os.path.abspath(PADDING_IMAGE_NAME) 
    padding_depth_path = padding_rgb_path # 深度图填充通常也可以用这张黑图

    if not os.path.exists(INPUT_JSON_PATH):
        print(f"❌ 错误：找不到文件 {INPUT_JSON_PATH}")
        return

    print(f"正在读取 {INPUT_JSON_PATH} ...")
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []

    print("开始处理数据...")
    for idx, item in tqdm(enumerate(data), total=len(data)):
        
        # --- 0. 筛选逻辑：只处理 traj_3279 ---
        if 'images' not in item:
            continue
        
        # 只要有一张图片路径包含 traj_3279 就算
        is_target_traj = any(TARGET_TRAJ in img_path for img_path in item['images'])
        if not is_target_traj:
            continue

        # --- 1. 处理路径 (上一轮的逻辑) ---
        raw_rgb_list = []
        raw_depth_list = []
        
        for old_path in item['images']:
            file_name = os.path.basename(old_path)
            
            # 拼接 RGB
            new_rgb = os.path.join(RGB_PREFIX_NEW, file_name)
            raw_rgb_list.append(new_rgb)
            
            # 拼接 Depth
            new_depth = os.path.join(DEPTH_PREFIX_NEW, file_name)
            raw_depth_list.append(new_depth)

        # --- 2. 列表截断与填充 (本轮逻辑，包含深度图同步) ---
        # 变成 [H, H, H, H, C]
        final_rgb_list = normalize_list_length(raw_rgb_list, padding_rgb_path)
        final_depth_list = normalize_list_length(raw_depth_list, padding_depth_path)
        
        if final_rgb_list is None: 
            continue

        # --- 3. 处理对话文本 (本轮逻辑) ---
        new_conversations = []
        for turn in item['conversations']:
            new_turn = turn.copy()
            
            if turn['from'] == 'human':
                new_turn['value'] = process_human_text(turn['value'])
            elif turn['from'] == 'gpt':
                new_turn['value'] = process_gpt_response(turn['value'])
            
            new_conversations.append(new_turn)

        # --- 4. 构建新条目 ---
        new_item = item.copy()
        new_item['images'] = final_rgb_list
        new_item['depth_images'] = final_depth_list # 加入深度图
        new_item['conversations'] = new_conversations
        
        processed_data.append(new_item)

    # 保存结果
    print(f"正在保存结果到 {OUTPUT_JSON_PATH} ...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 处理完成！")
    print(f"成功筛选并转换: {len(processed_data)} 条数据")

if __name__ == "__main__":
    main()