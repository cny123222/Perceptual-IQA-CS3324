#!/bin/bash

# 快速测试实验 - 只跑1个Round
# 基于最佳Base配置，测试不同变体

echo "=========================================="
echo "实验1: Base + 更小的Ranking Loss (alpha=0.3)"
echo "=========================================="

python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --epochs 30 \
  --train_test_num 1 \
  --batch_size 32 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.3 \
  --ranking_loss_margin 0.1 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

echo ""
echo "=========================================="
echo "实验2: Base + Attention Fusion"
echo "=========================================="

python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --attention_fusion \
  --epochs 30 \
  --train_test_num 1 \
  --batch_size 32 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --ranking_loss_margin 0.1 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="

