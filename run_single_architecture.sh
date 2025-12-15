#!/bin/bash
# 单个架构实验脚本
# 全部在 ranking-loss 分支运行
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
echo "分支: ranking-loss"
echo "========================================="

case "$ARCHITECTURE" in
    "resnet")
        echo "实验: ResNet-50 (原始架构)"
        echo "训练脚本: train_test_IQA.py"
        echo ""
        python train_test_IQA.py \
            --dataset "$DATASET" \
            --epochs "$EPOCHS" \
            --train_test_num "$TRAIN_TEST_NUM" \
            --batch_size "$BATCH_SIZE" \
            --train_patch_num "$TRAIN_PATCH_NUM" \
            --test_patch_num "$TEST_PATCH_NUM"
        ;;
    "swin")
        echo "实验: Swin Transformer (无 Ranking Loss)"
        echo "训练脚本: train_swin.py"
        echo "Ranking Loss Alpha: 0 (禁用)"
        echo ""
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
        echo "实验: Swin Transformer + Ranking Loss"
        echo "训练脚本: train_swin.py"
        echo "Ranking Loss Alpha: $RANKING_LOSS_ALPHA"
        echo "Ranking Loss Margin: $RANKING_LOSS_MARGIN"
        echo ""
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

echo ""
echo "✓ 训练完成！"
