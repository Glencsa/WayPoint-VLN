import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_loss_files(log_dir):
    """
    Parse log files in the specified directory to extract loss and epoch information.
    """
    loss_data = []
    files = glob.glob(os.path.join(log_dir, "*.txt"))
    print(f"📂 发现 {len(files)} 个日志文件...")
    pattern = re.compile(r"\{'loss':\s*([\d\.]+),\s*'grad_norm':.*?,\s*'epoch':\s*([\d\.]+)\}")

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    loss = float(match.group(1))
                    epoch = float(match.group(2))
                    loss_data.append((epoch, loss))
    
    loss_data.sort(key=lambda x: x[0])
    
    return loss_data

def smooth_curve(points, factor=0.9):
    """
    Use Exponential Moving Average (EMA) to smooth the curve.
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
        print("Cannot plot loss curve: No data available.")
        return

    epochs = [x[0] for x in loss_data]
    losses = [x[1] for x in loss_data]
    smooth_losses = smooth_curve(losses, factor=0.85)
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')

    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(epochs, losses, color='dodgerblue', alpha=0.3, linewidth=1, label='Raw Loss')
    plt.plot(epochs, smooth_losses, color='navy', linewidth=2.5, label='Smoothed Trend (EMA)')
    min_loss = min(smooth_losses)
    min_idx = smooth_losses.index(min_loss)
    plt.scatter(epochs[min_idx], min_loss, color='red', s=50, zorder=5)
    plt.annotate(f'Min: {min_loss:.4f}', 
                 xy=(epochs[min_idx], min_loss), 
                 xytext=(epochs[min_idx], min_loss + (max(losses)-min(losses))*0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, fontweight='bold')

    plt.title('Training Loss Convergence', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    output_path = 'loss_curve1.png'
    plt.savefig(output_path)
    print(f"picture saved at: {output_path}")

if __name__ == "__main__":
    log_directory = "/home/isvl/guan_code/WayPoint-VLN/log" 
    data = parse_loss_files(log_directory)
    plot_loss(data)
