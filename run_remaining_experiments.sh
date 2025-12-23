#!/bin/bash

################################################################################
# 运行剩余实验 - Batch 2 和 Batch 3
# Batch 1 已经在运行，这个脚本会在Batch 1完成后自动启动剩余实验
################################################################################

set -e

BASE_DIR="/root/Perceptual-IQA-CS3324"
LOG_DIR="$BASE_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$BASE_DIR"

echo "========================================"
echo "等待 Batch 1 完成..."
echo "========================================"

# 等待Batch 1的两个进程完成
echo "检测到的Batch 1进程:"
ps aux | grep "train_swin.py.*lr 1e-6\|train_swin.py.*lr 5e-7" | grep -v grep

BATCH1_PIDS=$(ps aux | grep "train_swin.py.*lr 1e-6\|train_swin.py.*lr 5e-7" | grep -v grep | awk '{print $2}')

if [ -z "$BATCH1_PIDS" ]; then
    echo "Batch 1 已完成或未运行"
else
    echo "等待进程: $BATCH1_PIDS"
    for pid in $BATCH1_PIDS; do
        echo "等待 PID $pid..."
        while kill -0 $pid 2>/dev/null; do
            sleep 60
            echo "  $(date): Batch 1 仍在运行..."
        done
    done
    echo "Batch 1 完成！"
fi

echo ""
echo "========================================"
echo "启动 Batch 2: Ablation Studies"
echo "========================================"

# GPU 0: A1 - Remove Attention
echo "启动 GPU 0: A1 - Remove Attention"
cd "$BASE_DIR"
tmux send-keys -t iqa_experiments:gpu0 C-c  # 清理
sleep 2
tmux send-keys -t iqa_experiments:gpu0 "cd $BASE_DIR" C-m
tmux send-keys -t iqa_experiments:gpu0 "CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 10 --lr 1e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/batch2_gpu0_A1_no_attention_${TIMESTAMP}.log" C-m

# GPU 1: A2 - Remove Multi-scale
echo "启动 GPU 1: A2 - Remove Multi-scale"
tmux send-keys -t iqa_experiments:gpu1 C-c  # 清理
sleep 2
tmux send-keys -t iqa_experiments:gpu1 "cd $BASE_DIR" C-m
tmux send-keys -t iqa_experiments:gpu1 "CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 10 --lr 1e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --no_multi_scale --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/batch2_gpu1_A2_no_multiscale_${TIMESTAMP}.log" C-m

echo "Batch 2 已启动，等待完成..."
sleep 10

# 等待Batch 2完成
while ps aux | grep -E "train_swin.py.*(no_multi_scale|drop_path_rate 0.3.*ranking_loss_alpha 0.*no_spaq)" | grep -v grep | grep -q .; do
    sleep 60
    echo "  $(date): Batch 2 仍在运行..."
done

echo "Batch 2 完成！"

echo ""
echo "========================================"
echo "启动 Batch 3: Model Size Comparison"
echo "========================================"

# GPU 0: B1 - Swin-Tiny
echo "启动 GPU 0: B1 - Swin-Tiny"
tmux send-keys -t iqa_experiments:gpu0 "cd $BASE_DIR" C-m
tmux send-keys -t iqa_experiments:gpu0 "CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size tiny --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 10 --lr 1e-6 --weight_decay 2e-4 --drop_path_rate 0.2 --dropout_rate 0.3 --lr_scheduler cosine --attention_fusion --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/batch3_gpu0_B1_tiny_${TIMESTAMP}.log" C-m

# GPU 1: B2 - Swin-Small
echo "启动 GPU 1: B2 - Swin-Small"
tmux send-keys -t iqa_experiments:gpu1 "cd $BASE_DIR" C-m
tmux send-keys -t iqa_experiments:gpu1 "CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size small --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 10 --lr 1e-6 --weight_decay 2e-4 --drop_path_rate 0.25 --dropout_rate 0.35 --lr_scheduler cosine --attention_fusion --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/batch3_gpu1_B2_small_${TIMESTAMP}.log" C-m

echo "Batch 3 已启动，等待完成..."
sleep 10

# 等待Batch 3完成
while ps aux | grep "train_swin.py.*model_size" | grep -E "tiny|small" | grep -v grep | grep -q .; do
    sleep 60
    echo "  $(date): Batch 3 仍在运行..."
done

echo "Batch 3 完成！"

echo ""
echo "========================================"
echo "所有实验完成！"
echo "========================================"
echo "查看结果:"
echo "  Batch 1: logs/batch1_gpu*_20251223_002208.log"
echo "  Batch 2: logs/batch2_gpu*_${TIMESTAMP}.log"
echo "  Batch 3: logs/batch3_gpu*_${TIMESTAMP}.log"
echo ""
echo "提取结果: ./extract_all_results.sh"

