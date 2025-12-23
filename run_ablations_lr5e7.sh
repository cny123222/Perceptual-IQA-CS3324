#!/bin/bash

################################################################################
# 消融实验和模型大小实验 - 使用最佳学习率 5e-7
# 
# 实验配置:
# - Learning Rate: 5e-7 (已验证的最佳LR)
# - Epochs: 10
# - Patience: 3
# - train_test_num: 1 (单轮)
# - Batch Size: 32
#
# 实验列表:
# 1. Baseline (重新跑，确保一致性)
# 2. A1: Remove Attention Fusion
# 3. A2: Remove Multi-scale Fusion
# 4. B1: Tiny Model
# 5. B2: Small Model
################################################################################

set -e

BASE_DIR="/root/Perceptual-IQA-CS3324"
LOG_DIR="$BASE_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$BASE_DIR"

# 创建tmux session
echo "创建tmux session: iqa_ablations"
tmux kill-session -t iqa_ablations 2>/dev/null || true
tmux new-session -d -s iqa_ablations -n controller

# 创建窗口
tmux new-window -t iqa_ablations -n gpu0
tmux new-window -t iqa_ablations -n gpu1
tmux select-window -t iqa_ablations:controller

echo "========================================"
echo "使用最佳LR=5e-7进行所有消融实验"
echo "========================================"
echo ""
echo "实验计划:"
echo "  Batch 1: Baseline + A1 (Remove Attention)"
echo "  Batch 2: A2 (Remove Multi-scale) + B1 (Tiny)"
echo "  Batch 3: B2 (Small)"
echo ""
echo "预计总时间: ~1.5小时"
echo ""

sleep 3

echo "========================================"
echo "BATCH 1: Baseline (LR=5e-7) + A1"
echo "========================================"

# GPU 0: Baseline with LR=5e-7 (重新跑以确保一致性)
echo "启动 GPU 0: Baseline (LR=5e-7)"
tmux send-keys -t iqa_ablations:gpu0 "cd $BASE_DIR" C-m
tmux send-keys -t iqa_ablations:gpu0 "CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 5e-7 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --attention_fusion --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/baseline_lr5e7_${TIMESTAMP}.log" C-m

# GPU 1: A1 - Remove Attention
echo "启动 GPU 1: A1 - Remove Attention"
tmux send-keys -t iqa_ablations:gpu1 "cd $BASE_DIR" C-m
tmux send-keys -t iqa_ablations:gpu1 "CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 5e-7 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/A1_no_attention_lr5e7_${TIMESTAMP}.log" C-m

echo "Batch 1 已启动，等待完成..."
sleep 30

# 等待Batch 1完成
while ps aux | grep "train_swin.py" | grep -E "CUDA_VISIBLE_DEVICES=(0|1)" | grep -v grep | wc -l | grep -q "2"; do
    sleep 60
    echo "  $(date '+%H:%M:%S'): Batch 1 运行中..."
done

echo "Batch 1 完成！"

sleep 10

echo ""
echo "========================================"
echo "BATCH 2: A2 + B1"
echo "========================================"

# GPU 0: A2 - Remove Multi-scale
echo "启动 GPU 0: A2 - Remove Multi-scale"
tmux send-keys -t iqa_ablations:gpu0 "CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 5e-7 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --no_multiscale --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/A2_no_multiscale_lr5e7_${TIMESTAMP}.log" C-m

# GPU 1: B1 - Tiny Model
echo "启动 GPU 1: B1 - Tiny Model"
tmux send-keys -t iqa_ablations:gpu1 "CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size tiny --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 5e-7 --weight_decay 2e-4 --drop_path_rate 0.2 --dropout_rate 0.3 --lr_scheduler cosine --attention_fusion --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/B1_tiny_lr5e7_${TIMESTAMP}.log" C-m

sleep 30

# 等待Batch 2完成
while ps aux | grep "train_swin.py" | grep -E "CUDA_VISIBLE_DEVICES=(0|1)" | grep -v grep | wc -l | grep -q "2"; do
    sleep 60
    echo "  $(date '+%H:%M:%S'): Batch 2 运行中..."
done

echo "Batch 2 完成！"

sleep 10

echo ""
echo "========================================"
echo "BATCH 3: B2 (Small Model)"
echo "========================================"

# GPU 0: B2 - Small Model
echo "启动 GPU 0: B2 - Small Model"
tmux send-keys -t iqa_ablations:gpu0 "CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size small --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 5e-7 --weight_decay 2e-4 --drop_path_rate 0.25 --dropout_rate 0.35 --lr_scheduler cosine --attention_fusion --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/B2_small_lr5e7_${TIMESTAMP}.log" C-m

sleep 30

# 等待Batch 3完成
while ps aux | grep "train_swin.py" | grep "CUDA_VISIBLE_DEVICES=0" | grep -v grep | grep -q .; do
    sleep 60
    echo "  $(date '+%H:%M:%S'): Batch 3 运行中..."
done

echo "Batch 3 完成！"

echo ""
echo "========================================"
echo "所有实验完成！"
echo "========================================"
echo ""
echo "提取结果:"
echo "  grep 'Best test SRCC' logs/*_lr5e7_${TIMESTAMP}.log"
echo ""
echo "实验日志:"
echo "  baseline_lr5e7_${TIMESTAMP}.log"
echo "  A1_no_attention_lr5e7_${TIMESTAMP}.log"
echo "  A2_no_multiscale_lr5e7_${TIMESTAMP}.log"
echo "  B1_tiny_lr5e7_${TIMESTAMP}.log"
echo "  B2_small_lr5e7_${TIMESTAMP}.log"
echo ""
echo "tmux窗口:"
echo "  tmux attach -t iqa_ablations"

