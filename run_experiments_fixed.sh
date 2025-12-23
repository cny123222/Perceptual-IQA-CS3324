#!/bin/bash

################################################################################
# 修复版本：所有实验自动化脚本（tmux）
# 修复内容：
# 1. train_test_num 改为 1（只跑1轮）
# 2. --no_multi_scale 改为 --no_multiscale（正确参数名）
################################################################################

set -e

BASE_DIR="/root/Perceptual-IQA-CS3324"
LOG_DIR="$BASE_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$BASE_DIR"

# 创建tmux session
echo "创建tmux session: iqa_experiments"
tmux kill-session -t iqa_experiments 2>/dev/null || true
tmux new-session -d -s iqa_experiments -n controller

# 创建窗口
tmux new-window -t iqa_experiments -n gpu0
tmux new-window -t iqa_experiments -n gpu1
tmux select-window -t iqa_experiments:controller

echo "========================================"
echo "BATCH 1: Learning Rate Experiments"
echo "========================================"

# GPU 0: LR = 1e-6
echo "启动 GPU 0: LR=1e-6"
tmux send-keys -t iqa_experiments:gpu0 "cd $BASE_DIR" C-m
tmux send-keys -t iqa_experiments:gpu0 "CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 1e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --attention_fusion --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/batch1_gpu0_lr1e6_${TIMESTAMP}.log" C-m

# GPU 1: LR = 5e-7
echo "启动 GPU 1: LR=5e-7"
tmux send-keys -t iqa_experiments:gpu1 "cd $BASE_DIR" C-m
tmux send-keys -t iqa_experiments:gpu1 "CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 5e-7 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --attention_fusion --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/batch1_gpu1_lr5e7_${TIMESTAMP}.log" C-m

echo "Batch 1 已启动，等待完成..."
sleep 30

# 等待Batch 1完成
while ps aux | grep "train_swin.py.*lr 1e-6\|train_swin.py.*lr 5e-7" | grep -v grep | grep -q .; do
    sleep 60
done

echo "Batch 1 完成！"

echo ""
echo "========================================"
echo "BATCH 2: Ablation Studies"
echo "========================================"

# GPU 0: A1 - Remove Attention
echo "启动 GPU 0: A1 - Remove Attention"
tmux send-keys -t iqa_experiments:gpu0 "CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 1e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/batch2_gpu0_A1_no_attention_${TIMESTAMP}.log" C-m

# GPU 1: A2 - Remove Multi-scale (修正参数名)
echo "启动 GPU 1: A2 - Remove Multi-scale"
tmux send-keys -t iqa_experiments:gpu1 "CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 1e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --no_multiscale --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/batch2_gpu1_A2_no_multiscale_${TIMESTAMP}.log" C-m

sleep 30

# 等待Batch 2完成
while ps aux | grep "train_swin.py.*A1\|train_swin.py.*A2\|train_swin.py.*no_multiscale\|train_swin.py.*batch2" | grep -v grep | grep -q .; do
    sleep 60
done

echo "Batch 2 完成！"

echo ""
echo "========================================"
echo "BATCH 3: Model Size Comparison"
echo "========================================"

# GPU 0: B1 - Swin-Tiny
echo "启动 GPU 0: B1 - Swin-Tiny"
tmux send-keys -t iqa_experiments:gpu0 "CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size tiny --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 1e-6 --weight_decay 2e-4 --drop_path_rate 0.2 --dropout_rate 0.3 --lr_scheduler cosine --attention_fusion --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/batch3_gpu0_B1_tiny_${TIMESTAMP}.log" C-m

# GPU 1: B2 - Swin-Small
echo "启动 GPU 1: B2 - Swin-Small"
tmux send-keys -t iqa_experiments:gpu1 "CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size small --batch_size 32 --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --lr 1e-6 --weight_decay 2e-4 --drop_path_rate 0.25 --dropout_rate 0.35 --lr_scheduler cosine --attention_fusion --ranking_loss_alpha 0 --test_random_crop --no_spaq --no_color_jitter 2>&1 | tee logs/batch3_gpu1_B2_small_${TIMESTAMP}.log" C-m

sleep 30

# 等待Batch 3完成
while ps aux | grep "train_swin.py.*B1\|train_swin.py.*B2\|train_swin.py.*tiny\|train_swin.py.*small" | grep -v grep | grep -q .; do
    sleep 60
done

echo "Batch 3 完成！"

echo ""
echo "========================================"
echo "所有实验完成！"
echo "========================================"
echo "查看结果: ls -lh logs/batch*_${TIMESTAMP}.log"
echo ""
echo "提取结果:"
echo "  grep 'Best test SRCC' logs/batch*_${TIMESTAMP}.log"
echo ""
echo "tmux窗口:"
echo "  tmux attach -t iqa_experiments"
echo "  Ctrl+B 然后 1/2 切换窗口"

