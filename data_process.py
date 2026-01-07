import json
import os

def process_r2r_data(input_file, output_file):
    print(f"正在读取文件: {input_file} ...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return

    # 定义新的提示词文本
    new_prompt_suffix = (
        "You are provided with:\n"
        "- Historical observations(four images): <history> \n"
        "- Current observation: <current>, there are some routes on the current observation.\n\n"
        "Your task is to select the best route number based on these routes, you can also choose the 0 to turn left 30° ,choose the 8 to turn right 30°, or return -1 to Stop. \n"
        " The format of the result is {'Route': number -1~8}"
    )

    # 遍历每一条数据进行修改
    processed_count = 0
    for item in data:
        # 1. 修改深度图路径 (depth_images)
        # 目标: .../depth_images_r2r_train/ep_xxx -> dataset_waypoint/depth_images_r2r_train/ep_xxx
        new_depth_images = []
        for path in item.get('depth_images', []):
            if 'depth_images_r2r_train/' in path:
                # 分割路径，保留 'depth_images_r2r_train/' 之后的部分
                suffix = path.split('depth_images_r2r_train/')[-1]
                new_path = f"/home/guanbin/scratch/dataset/r2r_dataset/depth_images_r2r_train/{suffix}"
                new_depth_images.append(new_path)
            else:
                # 如果路径格式不匹配，保持原样或打印警告
                new_depth_images.append(path)
        item['depth_images'] = new_depth_images

        # 2. 修改RGB图路径 (images)
        # 目标: .../rgb_images_points/ep_xxx -> dataset_waypoint/rgb_images_points/ep_xxx
        new_rgb_images = []
        for path in item.get('images', []):
            if 'rgb_images_points/' in path:
                # 分割路径，保留 'rgb_images_points/' 之后的部分
                suffix = path.split('rgb_images_points/')[-1]
                new_path = f"/home/guanbin/scratch/dataset/r2r_dataset/rgb_images_points/{suffix}"
                new_rgb_images.append(new_path)
            else:
                new_rgb_images.append(path)
        item['images'] = new_rgb_images

        # 3. 修改 conversations 中的 value 文本
        if 'conversations' in item and len(item['conversations']) > 0:
            human_conv = item['conversations'][0]
            if human_conv['from'] == 'human':
                original_text = human_conv['value']
                split_marker = "You are provided with:"
                
                # 检查是否存在标记文本
                if split_marker in original_text:
                    # 获取标记之前的所有文本（即导航指令部分）
                    instruction_part = original_text.split(split_marker)[0]
                    # 拼接新的后缀
                    human_conv['value'] = instruction_part + new_prompt_suffix
        
        processed_count += 1

    print(f"处理完成，共处理了 {processed_count} 条数据。")
    print(f"正在保存到: {output_file} ...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("完成！")

# 执行脚本
if __name__ == "__main__":
    input_filename = "/media/isvl/T7/vln_dataset/our_dataset_part2/split_part2_moved.json"
    output_filename = "/media/isvl/T7/vln_dataset/our_dataset_part2/rgb_images_r2r_train.json" # 保存为新文件以防覆盖

    process_r2r_data(input_filename, output_filename)