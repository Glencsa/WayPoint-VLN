import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_loss_files(log_dir):
    """
    遍历文件夹读取所有txt文件，解析loss和epoch数据
    """
    loss_data = []
    
    # 查找所有txt文件
    files = glob.glob(os.path.join(log_dir, "*.txt"))
    print(f"📂 发现 {len(files)} 个日志文件...")

    # 正则表达式匹配字典格式: {'loss': 58.0566, ... 'epoch': 0.0}
    # 兼容浮点数和整数
    pattern = re.compile(r"\{'loss':\s*([\d\.]+),\s*'grad_norm':.*?,\s*'epoch':\s*([\d\.]+)\}")

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    loss = float(match.group(1))
                    epoch = float(match.group(2))
                    loss_data.append((epoch, loss))
    
    # 按 epoch 排序，防止读取文件顺序混乱导致曲线回折
    loss_data.sort(key=lambda x: x[0])
    
    return loss_data

def smooth_curve(points, factor=0.9):
    """
    使用指数移动平均 (EMA) 平滑曲线
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_loss(loss_data):
    if not loss_data:
        print("⚠️ 未找到任何 Loss 数据，请检查日志格式或路径。")
        return

    epochs = [x[0] for x in loss_data]
    losses = [x[1] for x in loss_data]
    
    # 计算平滑曲线
    smooth_losses = smooth_curve(losses, factor=0.85)

    # --- 开始绘图 ---
    # 设置风格 (需要 matplotlib 3.6+ 支持 seaborn-v0_8，旧版可用 seaborn)
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')

    plt.figure(figsize=(12, 6), dpi=100)
    
    # 1. 绘制原始 Loss (浅色、透明，作为背景)
    plt.plot(epochs, losses, color='dodgerblue', alpha=0.3, linewidth=1, label='Raw Loss')
    
    # 2. 绘制平滑 Loss (深色、醒目，作为主趋势)
    plt.plot(epochs, smooth_losses, color='navy', linewidth=2.5, label='Smoothed Trend (EMA)')

    # 3. 标注最低点
    min_loss = min(smooth_losses)
    min_idx = smooth_losses.index(min_loss)
    plt.scatter(epochs[min_idx], min_loss, color='red', s=50, zorder=5)
    plt.annotate(f'Min: {min_loss:.4f}', 
                 xy=(epochs[min_idx], min_loss), 
                 xytext=(epochs[min_idx], min_loss + (max(losses)-min(losses))*0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, fontweight='bold')

    # 设置标签和标题
    plt.title('Training Loss Convergence', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = 'loss_curve1.png'
    plt.savefig(output_path)
    print(f"✅ 绘图完成！已保存为: {output_path}")
    # plt.show()

if __name__ == "__main__":
    # 在这里修改你的日志文件夹路径
    # 默认为当前目录下的 txt_log 文件夹，或者你可以改成 '.' 表示当前目录
    log_directory = "/home/isvl/guan_code/WayPoint-VLN/log" 
    
    data = parse_loss_files(log_directory)
    plot_loss(data)
