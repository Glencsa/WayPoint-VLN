import json
import os
import random
import re
from collections import defaultdict
from tqdm import tqdm

# ================= 配置区域 =================
JSON_PATH = "/media/isvl/T7/navid/VLN_Waypoint/rgb_images_r2r_train.json" 
NEW_JSON_PATH = "datasets/filtered_traj_cleaned.json"
DELETE_RATIO = 1 / 3
DRY_RUN = True  # True: 试运行 (不删文件); False: 真的删文件
# ===========================================

def extract_traj_id(item):
    """
    从数据条目中提取唯一的轨迹 ID。
    假设路径包含 '.../ep_4991/traj_3279/step_0...'
    我们将 'ep_xxxx/traj_yyyy' 组合作为唯一 ID，防止不同 episode 有重名 traj。
    """
    # 获取第一张图片或深度图的路径
    path = ""
    if "images" in item and len(item["images"]) > 0:
        path = item["images"][0]
    elif "depth_images" in item and len(item["depth_images"]) > 0:
        path = item["depth_images"][0]
    else:
        return "unknown_traj"

    # 使用正则提取 ep_xxxx/traj_yyyy
    # 匹配类似: ep_4991/traj_3279
    match = re.search(r'(ep_\d+)[/\\](traj_\d+)', path)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    
    # 如果没匹配到，尝试只匹配 traj_xxxx (兼容性)
    match_traj = re.search(r'(traj_\d+)', path)
    if match_traj:
        return match_traj.group(1)
        
    return "unknown_traj"

def main():
    print(f"正在加载 JSON: {JSON_PATH} ...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"原始数据总样本数 (Steps): {total_samples}")

    # 1. 按轨迹 ID 分组数据
    print("正在按轨迹 (Trajectory) 对数据进行分组...")
    traj_groups = defaultdict(list)
    for item in data:
        traj_id = extract_traj_id(item)
        traj_groups[traj_id].append(item)
    
    all_traj_ids = list(traj_groups.keys())
    total_trajs = len(all_traj_ids)
    print(f"识别到独立轨迹数: {total_trajs}")
    
    # 2. 按轨迹 ID 随机切分
    num_traj_to_delete = int(total_trajs * DELETE_RATIO)
    print(f"计划删除轨迹数: {num_traj_to_delete}")
    print(f"计划保留轨迹数: {total_trajs - num_traj_to_delete}")

    random.seed(42) # 固定种子
    random.shuffle(all_traj_ids)
    
    # 确定要删除的轨迹 ID 集合
    deleted_traj_ids_set = set(all_traj_ids[:num_traj_to_delete])
    
    # 3. 将数据分配到 Keep 和 Delete 列表
    data_to_keep = []
    data_to_delete = []
    
    for traj_id in all_traj_ids:
        items = traj_groups[traj_id]
        if traj_id in deleted_traj_ids_set:
            data_to_delete.extend(items)
        else:
            data_to_keep.extend(items)
            
    print(f"-> 最终保留样本数: {len(data_to_keep)}")
    print(f"-> 最终删除样本数: {len(data_to_delete)}")

    # 4. 建立文件白名单 (Protected Set)
    # 只有在保留的轨迹中完全没用到的图片，才能删
    protected_files = set()
    print("\n正在构建文件白名单...")
    
    for item in tqdm(data_to_keep, desc="Scanning Keep Data"):
        if "images" in item:
            for p in item["images"]: protected_files.add(os.path.abspath(p))
        if "depth_images" in item:
            for p in item["depth_images"]: protected_files.add(os.path.abspath(p))
                
    print(f"白名单构建完成，保护文件数: {len(protected_files)}")

    # 5. 扫描待删除文件
    files_to_remove = set()
    print("正在扫描可删除的文件...")
    
    for item in tqdm(data_to_delete, desc="Scanning Delete Data"):
        # RGB
        if "images" in item:
            for img_path in item["images"]:
                abs_path = os.path.abspath(img_path)
                if abs_path not in protected_files:
                    files_to_remove.add(abs_path)
        # Depth
        if "depth_images" in item:
            for depth_path in item["depth_images"]:
                abs_path = os.path.abspath(depth_path)
                if abs_path not in protected_files:
                    files_to_remove.add(abs_path)

    # 6. 执行操作
    print(f"\n统计结果:")
    print(f"- 物理文件总数 (待删): {len(files_to_remove)}")
    
    if DRY_RUN:
        print("\n【试运行模式 (DRY_RUN=True)】")
        print("不会删除任何文件。")
        print(f"如果执行，将生成新的 JSON 包含 {len(data_to_keep)} 条数据。")
        if len(files_to_remove) > 0:
            print("示例将删除的文件:", list(files_to_remove)[:3])
    else:
        print("\n【正在执行删除...】")
        deleted_counter = 0
        for file_path in tqdm(files_to_remove, desc="Deleting Files"):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_counter += 1
            except Exception as e:
                print(f"删除失败: {file_path}, {e}")
        
        print(f"\n文件清理完毕。已删除 {deleted_counter} 个文件。")
        
        print(f"正在保存新 JSON: {NEW_JSON_PATH}")
        with open(NEW_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(data_to_keep, f, indent=2)
        print("全部完成！")

if __name__ == "__main__":
    main()