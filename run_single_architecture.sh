#!/bin/bash
# 单个架构实验脚本
# 用法: ./run_single_architecture.sh [resnet|swin|swin-ranking]

set -e

ARCHITECTURE=${1:-"swin-ranking"}

# 配置参数
DATASET="koniq-10k"
EPOCHS=10
TRAIN_TEST_NUM=1
BATCH_SIZE=96
TRAIN_PATCH_NUM=20
TEST_PATCH_NUM=20
RANKING_LOSS_ALPHA=0.3
RANKING_LOSS_MARGIN=0.1

echo "========================================="
echo "运行架构: $ARCHITECTURE"
echo "========================================="

case "$ARCHITECTURE" in
    "resnet")
        echo "切换到 master 分支..."
        git checkout master
        echo "开始训练 ResNet-50..."
        python train_test_IQA.py \
            --dataset "$DATASET" \
            --epochs "$EPOCHS" \
            --train_test_num "$TRAIN_TEST_NUM" \
            --batch_size "$BATCH_SIZE" \
            --train_patch_num "$TRAIN_PATCH_NUM" \
            --test_patch_num "$TEST_PATCH_NUM"
        ;;
    "swin")
        echo "切换到 swin-transformer-backbone 分支..."
        git checkout swin-transformer-backbone
        echo "开始训练 Swin Transformer..."
        python train_swin.py \
            --dataset "$DATASET" \
            --epochs "$EPOCHS" \
            --train_test_num "$TRAIN_TEST_NUM" \
            --batch_size "$BATCH_SIZE" \
            --train_patch_num "$TRAIN_PATCH_NUM" \
            --test_patch_num "$TEST_PATCH_NUM" \
            --ranking_loss_alpha 0
        ;;
    "swin-ranking")
        echo "切换到 ranking-loss 分支..."
        git checkout ranking-loss
        echo "开始训练 Swin Transformer + Ranking Loss..."
        python train_swin.py \
            --dataset "$DATASET" \
            --epochs "$EPOCHS" \
            --train_test_num "$TRAIN_TEST_NUM" \
            --batch_size "$BATCH_SIZE" \
            --train_patch_num "$TRAIN_PATCH_NUM" \
            --test_patch_num "$TEST_PATCH_NUM" \
            --ranking_loss_alpha "$RANKING_LOSS_ALPHA" \
            --ranking_loss_margin "$RANKING_LOSS_MARGIN"
        ;;
    *)
        echo "错误: 未知的架构 '$ARCHITECTURE'"
        echo "用法: $0 [resnet|swin|swin-ranking]"
        exit 1
        ;;
esac

echo "✓ 训练完成！"

