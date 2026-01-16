import os
import json
import glob
import shutil  # 用于复制文件

def process_success_data(log_dir="log", video_dir="video", output_dir="success_clips"):
    """
    1. 读取 log_dir 下的 json，找出成功 (success=1) 的 ID。
    2. 在 video_dir 下找到对应的 .gif 文件。
    3. 将这些 gif 复制到 output_dir 文件夹中保存。
    """
    
    # --- 步骤 1: 找出成功的 ID ---
    pattern = os.path.join(log_dir, "*.json")
    files = glob.glob(pattern)
    success_ids = []
    
    print(f"🔍 正在分析 {log_dir} 文件夹下的 {len(files)} 个日志文件...")

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 检查 success 是否为 1
                if data.get("success") == 1:
                    item_id = data.get("id")
                    if item_id is not None:
                        success_ids.append(str(item_id)) # 转为字符串以确保文件名匹配
        except Exception as e:
            print(f"⚠️  跳过损坏文件 {file_path}: {e}")

    # 排序 ID
    try:
        success_ids.sort(key=lambda x: int(x))
    except ValueError:
        success_ids.sort()

    count = len(success_ids)
    print(f"✅ 共找到 {count} 个成功案例。ID列表: {success_ids}")

    if count == 0:
        print("没有需要复制的视频。")
        return

    # --- 步骤 2 & 3: 复制对应的 GIF 文件 ---
    
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 创建输出目录: {output_dir}")

    print(f"🚀 开始复制视频文件到 {output_dir} ...")
    
    copied_count = 0
    missing_count = 0

    for item_id in success_ids:
        # 假设视频文件名是 "ID.gif" (例如 "4.gif")
        # 如果你的文件名是 "episode_4.gif" 或其他格式，请修改下面这一行
        filename = f"{item_id}.gif" 
        
        src_path = os.path.join(video_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path) # copy2 会保留文件的时间戳信息
                # print(f"  [复制成功] {filename}") # 如果文件太多，可以注释掉这行
                copied_count += 1
            except Exception as e:
                print(f"  [复制失败] {filename}: {e}")
        else:
            print(f"  [文件缺失] 未找到视频: {src_path}")
            missing_count += 1

    # --- 总结报告 ---
    print("-" * 30)
    print(f"🎉 处理完成！")
    print(f"日志中成功ID数: {count}")
    print(f"实际复制视频数: {copied_count}")
    if missing_count > 0:
        print(f"缺失视频文件数: {missing_count}")
    print(f"视频已保存在: {os.path.abspath(output_dir)}")
    print("-" * 30)

if __name__ == "__main__":
    # 使用说明：
    # 1. 确保当前目录下有 log 文件夹（存放json）
    # 2. 确保当前目录下有 video 文件夹（存放gif）
    process_success_data(log_dir="/home/isvl/guan_code/WayPoint-VLN/VLN-CE/WayPointVLN-CE/tmp/WayPoint-VLN/log",
                         video_dir="/home/isvl/guan_code/WayPoint-VLN/VLN-CE/WayPointVLN-CE/tmp/WayPoint-VLN/video",
                         output_dir="/home/isvl/guan_code/WayPoint-VLN/VLN-CE/WayPointVLN-CE/tmp/WayPoint-VLN/success_clips")