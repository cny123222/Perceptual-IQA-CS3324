#!/bin/bash

################################################################################
# QualiCLIP Pre-training + Fine-tuning Auto Pipeline
# 
# 功能：
# 1. 监控预训练进程，等待完成
# 2. 验证预训练checkpoint
# 3. 自动启动微调训练（使用优化的学习率）
################################################################################

set -e  # Exit on error

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}QualiCLIP Auto Training Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 配置
PRETRAIN_LOG="/root/Perceptual-IQA-CS3324/logs/qualiclip_pretrain_run.log"
CHECKPOINT_DIR="/root/Perceptual-IQA-CS3324/checkpoints"
FINETUNE_LOG="/root/Perceptual-IQA-CS3324/logs/qualiclip_finetune_run.log"

################################################################################
# Step 1: 等待预训练完成
################################################################################

echo -e "${YELLOW}[Step 1] Waiting for pre-training to complete...${NC}"
echo ""

# 查找预训练进程
PRETRAIN_PID=$(ps aux | grep "[p]retrain_qualiclip.py" | awk '{print $2}')

if [ -z "$PRETRAIN_PID" ]; then
    echo -e "${YELLOW}⚠ No pre-training process found. Assuming already completed.${NC}"
else
    echo -e "Found pre-training process: PID ${PRETRAIN_PID}"
    echo "Monitoring progress..."
    echo ""
    
    # 监控进程
    while kill -0 $PRETRAIN_PID 2>/dev/null; do
        # 显示最新进度
        if [ -f "$PRETRAIN_LOG" ]; then
            LAST_LINE=$(tail -1 "$PRETRAIN_LOG" | grep -oP 'Epoch \d+/\d+' || echo "Training...")
            echo -ne "\r  Current: $LAST_LINE   "
        fi
        sleep 10
    done
    echo ""
    echo -e "${GREEN}✓ Pre-training process completed!${NC}"
fi

sleep 5

################################################################################
# Step 2: 验证预训练checkpoint
################################################################################

echo ""
echo -e "${YELLOW}[Step 2] Validating pre-training checkpoint...${NC}"
echo ""

# 查找最新的预训练checkpoint目录
PRETRAIN_DIR=$(ls -td ${CHECKPOINT_DIR}/qualiclip_pretrain_* 2>/dev/null | head -1)

if [ -z "$PRETRAIN_DIR" ]; then
    echo -e "${RED}✗ Error: No pre-training checkpoint directory found!${NC}"
    exit 1
fi

echo "Found checkpoint directory: $PRETRAIN_DIR"

# 查找epoch10的权重文件
PRETRAIN_WEIGHTS="$PRETRAIN_DIR/swin_base_epoch10.pkl"

if [ ! -f "$PRETRAIN_WEIGHTS" ]; then
    # 如果没有epoch10，尝试找最新的
    PRETRAIN_WEIGHTS=$(ls -t "$PRETRAIN_DIR"/swin_base_epoch*.pkl 2>/dev/null | head -1)
    if [ -z "$PRETRAIN_WEIGHTS" ]; then
        echo -e "${RED}✗ Error: No checkpoint file found in $PRETRAIN_DIR${NC}"
        exit 1
    fi
    echo -e "${YELLOW}⚠ epoch10 not found, using: $(basename $PRETRAIN_WEIGHTS)${NC}"
fi

echo -e "${GREEN}✓ Found pre-trained weights: $PRETRAIN_WEIGHTS${NC}"
FILE_SIZE=$(du -h "$PRETRAIN_WEIGHTS" | cut -f1)
echo "  File size: $FILE_SIZE"
echo ""

################################################################################
# Step 3: 启动微调训练
################################################################################

echo -e "${YELLOW}[Step 3] Starting fine-tuning with QualiCLIP pre-trained encoder...${NC}"
echo ""

# 训练参数（基于用户经验优化）
DATASET="koniq-10k"
MODEL_SIZE="base"
BATCH_SIZE=8
EPOCHS=50
LR_MAIN=1e-6              # HyperNet学习率（用户说1e-6效果好）
LR_ENCODER=5e-7           # Encoder学习率（更小，保护预训练特征）

echo "Training Configuration:"
echo "  Dataset: $DATASET"
echo "  Model Size: $MODEL_SIZE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Total Epochs: $EPOCHS"
echo "  HyperNet LR: $LR_MAIN"
echo "  Encoder LR: $LR_ENCODER"
echo "  Pre-trained Weights: $PRETRAIN_WEIGHTS"
echo ""
echo -e "${BLUE}Starting training in 5 seconds...${NC}"
sleep 5

# 进入项目目录
cd /root/Perceptual-IQA-CS3324

# 启动微调训练
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Fine-tuning...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

python train_swin.py \
    --dataset "$DATASET" \
    --model_size "$MODEL_SIZE" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR_MAIN \
    --pretrained_encoder "$PRETRAIN_WEIGHTS" \
    --lr_encoder_pretrained $LR_ENCODER \
    --no_color_jitter \
    --no_spaq \
    2>&1 | tee "$FINETUNE_LOG"

# 检查训练是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Fine-tuning completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Model saved to: checkpoints/"
    echo "Training log: $FINETUNE_LOG"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Fine-tuning failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check log file: $FINETUNE_LOG"
    exit 1
fi

################################################################################
# Step 4: 跨数据集测试（可选）
################################################################################

echo ""
echo -e "${YELLOW}[Step 4] Would you like to run cross-dataset evaluation?${NC}"
echo "Press Enter to skip, or type 'yes' to run tests:"
read -t 30 RUN_TESTS || RUN_TESTS=""

if [ "$RUN_TESTS" = "yes" ]; then
    echo ""
    echo -e "${BLUE}Running cross-dataset evaluation...${NC}"
    
    # Find the most recent checkpoint directory
    BEST_MODEL=$(find checkpoints/ -name "best_model.pkl" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
    
    if [ -f "$BEST_MODEL" ]; then
        python test_swin.py \
            --model_path "$BEST_MODEL" \
            --test_datasets spaq kadid agiqa \
            2>&1 | tee logs/qualiclip_cross_dataset_test.log
        
        echo -e "${GREEN}✓ Cross-dataset evaluation completed!${NC}"
    else
        echo -e "${RED}✗ Best model not found in checkpoints/${NC}"
    fi
else
    echo "Skipping cross-dataset evaluation."
fi

################################################################################
# Summary
################################################################################

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pipeline Completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  Pre-trained weights: $PRETRAIN_WEIGHTS"
echo "  Fine-tuned model: checkpoints/"
echo "  Training log: $FINETUNE_LOG"
echo ""
echo "Next steps:"
echo "  1. Check training metrics in: $FINETUNE_LOG"
echo "  2. Evaluate on test sets"
echo "  3. Compare with baseline results"
echo ""
echo -e "${GREEN}Done! 🎉${NC}"

