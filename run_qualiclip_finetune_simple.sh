#!/bin/bash

################################################################################
# QualiCLIP Fine-tuning (Simple Version)
# 直接运行微调，假设预训练已完成
################################################################################

# 查找预训练权重
PRETRAIN_WEIGHTS=$(ls -t /root/Perceptual-IQA-CS3324/checkpoints/qualiclip_pretrain_*/swin_base_epoch10.pkl 2>/dev/null | head -1)

if [ -z "$PRETRAIN_WEIGHTS" ]; then
    echo "Error: Pre-trained weights not found!"
    echo "Please run pre-training first: python pretrain_qualiclip.py"
    exit 1
fi

echo "Using pre-trained weights: $PRETRAIN_WEIGHTS"
echo ""

# 可自定义参数
DATASET="${1:-koniq-10k}"                    # 数据集
EPOCHS="${2:-50}"                            # Epochs
LR_MAIN="${3:-5e-7}"                         # 主学习率（根据实验优化）
LR_ENCODER="${4:-5e-7}"                      # Encoder学习率
BATCH_SIZE="${5:-32}"                        # Batch size（根据实验优化）

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  HyperNet LR: $LR_MAIN"
echo "  Encoder LR: $LR_ENCODER"
echo "  Batch Size: $BATCH_SIZE"
echo ""

cd /root/Perceptual-IQA-CS3324

python train_swin.py \
    --dataset "$DATASET" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR_MAIN \
    --pretrained_encoder "$PRETRAIN_WEIGHTS" \
    --lr_encoder_pretrained $LR_ENCODER \
    --model_size base \
    --train_patch_num 20 \
    --no_color_jitter \
    --no_spaq \
    2>&1 | tee logs/qualiclip_finetune_$(date +%Y%m%d_%H%M%S).log

