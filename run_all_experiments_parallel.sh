#!/bin/bash
# 并行运行三个实验，每个使用不同的GPU

# 配置参数
DATASET="koniq-10k"
EPOCHS=10
TRAIN_TEST_NUM=1
BATCH_SIZE=96
TRAIN_PATCH_NUM=20
TEST_PATCH_NUM=20
RANKING_LOSS_ALPHA=0.3
RANKING_LOSS_MARGIN=0.1

# 日志目录
mkdir -p logs_parallel
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================="
echo "并行运行三个实验"
echo "========================================="
echo "GPU 0: ResNet-50"
echo "GPU 1: Swin Transformer (无 Ranking Loss)"
echo "GPU 2: Swin Transformer + Ranking Loss"
echo "========================================="
echo ""

# 实验1: ResNet-50 (GPU 0)
echo "启动实验1 (GPU 0): ResNet-50..."
CUDA_VISIBLE_DEVICES=0 python train_test_IQA.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --train_test_num "$TRAIN_TEST_NUM" \
    --batch_size "$BATCH_SIZE" \
    --train_patch_num "$TRAIN_PATCH_NUM" \
    --test_patch_num "$TEST_PATCH_NUM" \
    > "logs_parallel/resnet50_${TIMESTAMP}.log" 2>&1 &
PID1=$!
echo "  PID: $PID1, 日志: logs_parallel/resnet50_${TIMESTAMP}.log"

# 实验2: Swin Transformer (GPU 1)
echo "启动实验2 (GPU 1): Swin Transformer..."
CUDA_VISIBLE_DEVICES=1 python train_swin.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --train_test_num "$TRAIN_TEST_NUM" \
    --batch_size "$BATCH_SIZE" \
    --train_patch_num "$TRAIN_PATCH_NUM" \
    --test_patch_num "$TEST_PATCH_NUM" \
    --ranking_loss_alpha 0 \
    > "logs_parallel/swin_${TIMESTAMP}.log" 2>&1 &
PID2=$!
echo "  PID: $PID2, 日志: logs_parallel/swin_${TIMESTAMP}.log"

# 实验3: Swin Transformer + Ranking Loss (GPU 2)
echo "启动实验3 (GPU 2): Swin Transformer + Ranking Loss..."
CUDA_VISIBLE_DEVICES=2 python train_swin.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --train_test_num "$TRAIN_TEST_NUM" \
    --batch_size "$BATCH_SIZE" \
    --train_patch_num "$TRAIN_PATCH_NUM" \
    --test_patch_num "$TEST_PATCH_NUM" \
    --ranking_loss_alpha "$RANKING_LOSS_ALPHA" \
    --ranking_loss_margin "$RANKING_LOSS_MARGIN" \
    > "logs_parallel/swin_ranking_${TIMESTAMP}.log" 2>&1 &
PID3=$!
echo "  PID: $PID3, 日志: logs_parallel/swin_ranking_${TIMESTAMP}.log"

echo ""
echo "========================================="
echo "所有实验已启动"
echo "========================================="
echo "查看运行状态:"
echo "  ps aux | grep python"
echo ""
echo "查看日志（实时）:"
echo "  tail -f logs_parallel/resnet50_${TIMESTAMP}.log"
echo "  tail -f logs_parallel/swin_${TIMESTAMP}.log"
echo "  tail -f logs_parallel/swin_ranking_${TIMESTAMP}.log"
echo ""
echo "等待所有实验完成:"
echo "  wait"
echo ""
echo "========================================="
