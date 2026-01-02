import json
import os
import random
import re
import shutil
from collections import defaultdict
from tqdm import tqdm

# ================= 配置区域 =================
# 原始 JSON 路径
JSON_PATH = "/media/isvl/T7/vln_dataset/our_dataset/rgb_images_r2r_train_processed.json"

# 数据集 A (保留在原地) 的新 JSON 名字
JSON_PATH_A = "/media/isvl/T7/vln_dataset/our_dataset/split_part1_kept.json"

# 数据集 B (将被移动) 的新 JSON 名字
JSON_PATH_B = "/media/isvl/T7/vln_dataset/our_dataset_part2/split_part2_moved.json"

# 【关键】定义路径根目录，用于文件移动和路径替换
# 原始数据的根目录 (目前图片所在的位置)
OLD_ROOT = "/media/isvl/T7/vln_dataset/our_dataset/"
# 新数据的根目录 (数据集 B 图片要移动到的位置)
NEW_ROOT = "/media/isvl/T7/vln_dataset/our_dataset_part2/"

# 切分比例 (例如 0.5 表示 50% 留在 A，50% 移动到 B)
SPLIT_RATIO = 0.5 

# True: 试运行 (不移动文件，只打印信息); False: 真的移动文件
DRY_RUN = False
# ===========================================

def extract_traj_id(item):
    """提取轨迹 ID"""
    path = ""
    if "images" in item and len(item["images"]) > 0:
        path = item["images"][0]
    elif "depth_images" in item and len(item["depth_images"]) > 0:
        path = item["depth_images"][0]
    else:
        return "unknown_traj"

    match = re.search(r'(ep_\d+)[/\\](traj_\d+)', path)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    
    match_traj = re.search(r'(traj_\d+)', path)
    if match_traj:
        return match_traj.group(1)
    return "unknown_traj"

def try_remove_empty_dir_recursive(path):
    """递归清理空文件夹 (仅用于清理 Dataset A 移走文件后留下的空壳)"""
    if not os.path.isdir(path): return
    try:
        if not os.listdir(path):
            os.rmdir(path)
            # 递归检查父目录
            try_remove_empty_dir_recursive(os.path.dirname(path))
    except: pass

def get_new_path(old_path):
    """将旧路径转换为新路径 (替换根目录)"""
    # 确保路径格式统一
    abs_old = os.path.abspath(old_path)
    abs_root_old = os.path.abspath(OLD_ROOT)
    abs_root_new = os.path.abspath(NEW_ROOT)
    
    if abs_old.startswith(abs_root_old):
        # 替换路径前缀
        rel_path = os.path.relpath(abs_old, abs_root_old)
        new_path = os.path.join(abs_root_new, rel_path)
        return new_path
    else:
        # 如果路径不在 OLD_ROOT 下，则不修改
        return old_path

def main():
    print(f"正在加载 JSON: {JSON_PATH} ...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. 按轨迹 ID 分组
    print("正在按轨迹分组...")
    traj_groups = defaultdict(list)
    for item in data:
        traj_id = extract_traj_id(item)
        traj_groups[traj_id].append(item)
    
    all_traj_ids = list(traj_groups.keys())
    total_trajs = len(all_traj_ids)
    print(f"总轨迹数: {total_trajs}")
    
    # 2. 随机切分
    num_traj_move = int(total_trajs * SPLIT_RATIO)
    print(f"计划保留轨迹数 (Set A): {total_trajs - num_traj_move}")
    print(f"计划移动轨迹数 (Set B): {num_traj_move}")

    random.seed(42)
    random.shuffle(all_traj_ids)
    
    ids_to_move = set(all_traj_ids[:num_traj_move]) # Set B
    
    data_keep = [] # Set A
    data_move = [] # Set B
    
    for traj_id in all_traj_ids:
        items = traj_groups[traj_id]
        if traj_id in ids_to_move:
            data_move.extend(items)
        else:
            data_keep.extend(items)

    # 3. 构建 Set A 的白名单 (这些文件绝对不能移走)
    files_in_A = set()
    print("正在扫描 Set A (保留组) 的文件引用...")
    for item in tqdm(data_keep):
        for key in ["images", "depth_images"]:
            if key in item:
                for p in item[key]: files_in_A.add(os.path.abspath(p))
    
    # 4. 扫描 Set B 需要移动的文件
    files_to_move = set() # 物理路径
    print("正在扫描 Set B (移动组) 的文件引用...")
    for item in tqdm(data_move):
        for key in ["images", "depth_images"]:
            if key in item:
                for p in item[key]: 
                    abs_p = os.path.abspath(p)
                    # 只有当文件不在 Set A 中时，才移动
                    # (如果在 Set A 中也用了，那就留在原地，Set B 指向旧位置或复制，这里选择不移动)
                    if abs_p not in files_in_A:
                        files_to_move.add(abs_p)

    print(f"\n统计:")
    print(f"- Set A 样本数: {len(data_keep)}")
    print(f"- Set B 样本数: {len(data_move)}")
    print(f"- 需要物理移动的文件数: {len(files_to_move)}")

    if DRY_RUN:
        print("\n【试运行模式 (DRY_RUN=True)】")
        print(f"源目录: {OLD_ROOT}")
        print(f"目标目录: {NEW_ROOT}")
        print("不会移动任何文件。")
        if len(files_to_move) > 0:
            sample = list(files_to_move)[0]
            print(f"示例移动: {sample} \n      -> {get_new_path(sample)}")
    else:
        print("\n【正在执行移动...】")
        
        # 创建目标根目录
        if not os.path.exists(NEW_ROOT):
            os.makedirs(NEW_ROOT)

        moved_count = 0
        affected_dirs = set() # 记录旧目录以便清理空文件夹

        # 5. 物理移动文件
        for old_abs_path in tqdm(files_to_move, desc="Moving Files"):
            if os.path.exists(old_abs_path):
                new_abs_path = get_new_path(old_abs_path)
                
                # 确保目标文件夹存在
                os.makedirs(os.path.dirname(new_abs_path), exist_ok=True)
                
                try:
                    # 使用 move 移动文件
                    shutil.move(old_abs_path, new_abs_path)
                    moved_count += 1
                    affected_dirs.add(os.path.dirname(old_abs_path))
                except Exception as e:
                    print(f"移动失败: {old_abs_path} -> {e}")

        print(f"文件移动完成: {moved_count} 个。")

        # 6. 更新 Set B JSON 中的路径
        print("正在更新 Set B 的 JSON 路径...")
        for item in tqdm(data_move, desc="Updating JSON paths"):
            # 更新 images
            if "images" in item:
                new_imgs = []
                for p in item["images"]:
                    abs_p = os.path.abspath(p)
                    # 如果这个文件被移动了，更新路径；否则保留原路径
                    if abs_p in files_to_move:
                        new_imgs.append(get_new_path(p))
                    else:
                        new_imgs.append(p)
                item["images"] = new_imgs

            # 更新 depth_images
            if "depth_images" in item:
                new_depths = []
                for p in item["depth_images"]:
                    abs_p = os.path.abspath(p)
                    if abs_p in files_to_move:
                        new_depths.append(get_new_path(p))
                    else:
                        new_depths.append(p)
                item["depth_images"] = new_depths

        # 7. 保存两个 JSON
        print(f"保存 Set A JSON: {JSON_PATH_A}")
        with open(JSON_PATH_A, 'w', encoding='utf-8') as f:
            json.dump(data_keep, f, indent=2)

        print(f"保存 Set B JSON: {JSON_PATH_B}")
        # 确保 B 的 JSON 目录存在
        os.makedirs(os.path.dirname(JSON_PATH_B), exist_ok=True)
        with open(JSON_PATH_B, 'w', encoding='utf-8') as f:
            json.dump(data_move, f, indent=2)

        # 8. 清理 Set A 中留下的空目录
        print("清理原目录中的空文件夹...")
        sorted_dirs = sorted(list(affected_dirs), key=len, reverse=True)
        for d in sorted_dirs:
            try_remove_empty_dir_recursive(d)

        print("全部完成！数据集已成功分离。")

if __name__ == "__main__":
    main()