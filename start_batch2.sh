#!/bin/bash

################################################################################
# 等待Batch 1完成后自动启动Batch 2
################################################################################

BASE_DIR="/root/Perceptual-IQA-CS3324"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$BASE_DIR"

echo "========================================"
echo "等待Batch 1完成..."
echo "========================================"

# 等待A1和A2进程完成
while ps -p 334764 334768 >/dev/null 2>&1; do
    sleep 30
    echo "$(date '+%H:%M:%S'): Batch 1 仍在运行..."
done

echo "Batch 1 完成！"
sleep 10

echo ""
echo "========================================"
echo "启动 BATCH 2: B1 + B2"
echo "========================================"

# GPU 0: B1 - Tiny Model
echo "启动 GPU 0: B1 - Tiny Model"
tmux send-keys -t iqa_ablations:gpu0 "cd $BASE_DIR" C-m
tmux send-keys -t iqa_ablations:gpu0 "CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size tiny --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 5e-7 --weight_decay 2e-4 --drop_path_rate 0.2 --dropout_rate 0.3 --lr_scheduler cosine --attention_fusion --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/B1_tiny_lr5e7_${TIMESTAMP}.log" C-m

# GPU 1: B2 - Small Model
echo "启动 GPU 1: B2 - Small Model"
tmux send-keys -t iqa_ablations:gpu1 "cd $BASE_DIR" C-m
tmux send-keys -t iqa_ablations:gpu1 "CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size small --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 5e-7 --weight_decay 2e-4 --drop_path_rate 0.25 --dropout_rate 0.35 --lr_scheduler cosine --attention_fusion --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/B2_small_lr5e7_${TIMESTAMP}.log" C-m

echo "Batch 2 已启动！"
echo ""
echo "日志:"
echo "  logs/B1_tiny_lr5e7_${TIMESTAMP}.log"
echo "  logs/B2_small_lr5e7_${TIMESTAMP}.log"

