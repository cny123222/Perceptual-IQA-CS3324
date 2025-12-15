#!/bin/bash
# 架构对比实验脚本
# 全部在 ranking-loss 分支运行，无需切换分支
# 三个架构：
#   1. ResNet-50 (使用 train_test_IQA.py)
#   2. Swin Transformer (使用 train_swin.py, ranking_loss_alpha=0)
#   3. Swin Transformer + Ranking Loss (使用 train_swin.py, ranking_loss_alpha=0.3)

set -e  # 遇到错误立即退出

# 配置参数（可以根据需要修改）
DATASET="koniq-10k"
EPOCHS=10
TRAIN_TEST_NUM=1
BATCH_SIZE=96
TRAIN_PATCH_NUM=20
TEST_PATCH_NUM=20
RANKING_LOSS_ALPHA=0.3
RANKING_LOSS_MARGIN=0.1

# 结果保存目录
RESULTS_DIR="comparison_results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$RESULTS_DIR/comparison_${TIMESTAMP}.txt"

echo "========================================="
echo "架构对比实验（全部在 ranking-loss 分支）"
echo "========================================="
echo "参数配置:"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Train Patches: $TRAIN_PATCH_NUM"
echo "  Test Patches: $TEST_PATCH_NUM"
echo "  Ranking Loss Alpha: $RANKING_LOSS_ALPHA"
echo ""
echo "结果将保存到: $RESULTS_FILE"
echo "========================================="
echo ""

# 确认当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo "当前分支: $CURRENT_BRANCH"
if [ "$CURRENT_BRANCH" != "ranking-loss" ]; then
    echo "警告: 当前不在 ranking-loss 分支，建议切换到 ranking-loss 分支后再运行"
    read -p "继续运行？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# ============================================
# 实验1: ResNet-50 (原始架构)
# ============================================
echo "========================================="
echo "实验 1/3: ResNet-50 (原始架构)"
echo "========================================="
echo "使用训练脚本: train_test_IQA.py"
echo "开始训练 ResNet-50..."
echo ""

python train_test_IQA.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --train_test_num "$TRAIN_TEST_NUM" \
    --batch_size "$BATCH_SIZE" \
    --train_patch_num "$TRAIN_PATCH_NUM" \
    --test_patch_num "$TEST_PATCH_NUM" \
    2>&1 | tee "$RESULTS_DIR/resnet50_${TIMESTAMP}.log"

echo ""
echo "✓ ResNet-50 训练完成"
echo ""

# ============================================
# 实验2: Swin Transformer (无 Ranking Loss)
# ============================================
echo "========================================="
echo "实验 2/3: Swin Transformer (无 Ranking Loss)"
echo "========================================="
echo "使用训练脚本: train_swin.py"
echo "Ranking Loss Alpha: 0 (禁用)"
echo "开始训练 Swin Transformer..."
echo ""

python train_swin.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --train_test_num "$TRAIN_TEST_NUM" \
    --batch_size "$BATCH_SIZE" \
    --train_patch_num "$TRAIN_PATCH_NUM" \
    --test_patch_num "$TEST_PATCH_NUM" \
    --ranking_loss_alpha 0 \
    2>&1 | tee "$RESULTS_DIR/swin_${TIMESTAMP}.log"

echo ""
echo "✓ Swin Transformer 训练完成"
echo ""

# ============================================
# 实验3: Swin Transformer + Ranking Loss
# ============================================
echo "========================================="
echo "实验 3/3: Swin Transformer + Ranking Loss"
echo "========================================="
echo "使用训练脚本: train_swin.py"
echo "Ranking Loss Alpha: $RANKING_LOSS_ALPHA"
echo "Ranking Loss Margin: $RANKING_LOSS_MARGIN"
echo "开始训练 Swin Transformer + Ranking Loss..."
echo ""

python train_swin.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --train_test_num "$TRAIN_TEST_NUM" \
    --batch_size "$BATCH_SIZE" \
    --train_patch_num "$TRAIN_PATCH_NUM" \
    --test_patch_num "$TEST_PATCH_NUM" \
    --ranking_loss_alpha "$RANKING_LOSS_ALPHA" \
    --ranking_loss_margin "$RANKING_LOSS_MARGIN" \
    2>&1 | tee "$RESULTS_DIR/swin_ranking_${TIMESTAMP}.log"

echo ""
echo "✓ Swin Transformer + Ranking Loss 训练完成"
echo ""

# ============================================
# 生成对比报告
# ============================================
echo "========================================="
echo "生成对比报告..."
echo "========================================="

cat > "$RESULTS_FILE" << EOF
架构对比实验报告
生成时间: $(date)
分支: ranking-loss
参数配置:
  Dataset: $DATASET
  Epochs: $EPOCHS
  Batch Size: $BATCH_SIZE
  Train Patches: $TRAIN_PATCH_NUM
  Test Patches: $TEST_PATCH_NUM
  Ranking Loss Alpha: $RANKING_LOSS_ALPHA
  Ranking Loss Margin: $RANKING_LOSS_MARGIN

注意：
- 所有实验都在 ranking-loss 分支运行
- 所有实验都包含：
  * 每个epoch保存checkpoint（带时间戳的文件夹名）
  * SPAQ数据集上的跨数据集测试
  * 训练稳定性修复（filter bug, backbone LR decay, optimizer state preservation）

=========================================
实验1: ResNet-50 (原始架构)
训练脚本: train_test_IQA.py
=========================================
EOF

if [ -f "$RESULTS_DIR/resnet50_${TIMESTAMP}.log" ]; then
    echo "训练日志: $RESULTS_DIR/resnet50_${TIMESTAMP}.log" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    echo "最佳结果:" >> "$RESULTS_FILE"
    grep -E "Best test SRCC|Epoch.*Test_SRCC.*Test_PLCC|SPAQ_SRCC.*SPAQ_PLCC" "$RESULTS_DIR/resnet50_${TIMESTAMP}.log" | tail -10 >> "$RESULTS_FILE" || echo "未找到结果" >> "$RESULTS_FILE"
else
    echo "实验未完成或日志文件不存在" >> "$RESULTS_FILE"
fi

cat >> "$RESULTS_FILE" << EOF

=========================================
实验2: Swin Transformer (无 Ranking Loss)
训练脚本: train_swin.py --ranking_loss_alpha 0
=========================================
EOF

if [ -f "$RESULTS_DIR/swin_${TIMESTAMP}.log" ]; then
    echo "训练日志: $RESULTS_DIR/swin_${TIMESTAMP}.log" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    echo "最佳结果:" >> "$RESULTS_FILE"
    grep -E "Best test SRCC|Epoch.*Test_SRCC.*Test_PLCC|SPAQ_SRCC.*SPAQ_PLCC" "$RESULTS_DIR/swin_${TIMESTAMP}.log" | tail -10 >> "$RESULTS_FILE" || echo "未找到结果" >> "$RESULTS_FILE"
else
    echo "实验未完成或日志文件不存在" >> "$RESULTS_FILE"
fi

cat >> "$RESULTS_FILE" << EOF

=========================================
实验3: Swin Transformer + Ranking Loss
训练脚本: train_swin.py --ranking_loss_alpha $RANKING_LOSS_ALPHA --ranking_loss_margin $RANKING_LOSS_MARGIN
=========================================
EOF

if [ -f "$RESULTS_DIR/swin_ranking_${TIMESTAMP}.log" ]; then
    echo "训练日志: $RESULTS_DIR/swin_ranking_${TIMESTAMP}.log" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    echo "最佳结果:" >> "$RESULTS_FILE"
    grep -E "Best test SRCC|Epoch.*Test_SRCC.*Test_PLCC|SPAQ_SRCC.*SPAQ_PLCC" "$RESULTS_DIR/swin_ranking_${TIMESTAMP}.log" | tail -10 >> "$RESULTS_FILE" || echo "未找到结果" >> "$RESULTS_FILE"
else
    echo "实验未完成或日志文件不存在" >> "$RESULTS_FILE"
fi

echo ""
echo "========================================="
echo "对比实验完成！"
echo "========================================="
echo "结果文件: $RESULTS_FILE"
echo "详细日志保存在: $RESULTS_DIR/"
echo ""
echo "查看对比报告:"
echo "  cat $RESULTS_FILE"
echo ""
