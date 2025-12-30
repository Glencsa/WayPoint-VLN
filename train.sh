#!/bin/bash

# ================= 配置区域 =================
# 1. 定义训练脚本和参数
SCRIPT="train_sft.py"
NUM_GPUS=2
PORT=29500

# 2. 定义日志文件路径 (自动按时间命名)
# 创建 logs 文件夹防止目录混乱
mkdir -p logs
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

# ================= 启动逻辑 =================

echo "准备启动分布式训练 (GPUs: $NUM_GPUS)..."
echo "日志将输出到: $LOG_FILE"
echo " swanlab: 也会同步记录到云端"


export PYTHONUNBUFFERED=1

nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    $SCRIPT > "$LOG_FILE" 2>&1 &

# 获取刚刚启动的进程 ID
PID=$!

echo "任务已成功在后台启动!"
echo "🆔 进程 PID: $PID"
echo "---------------------------------------------------"
echo "   常用操作:"
echo "   查看实时日志:  tail -f $LOG_FILE"
echo "   停止训练:      kill $PID"
echo "---------------------------------------------------"