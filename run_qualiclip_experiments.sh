#!/bin/bash
# Script to run QualiCLIP pretraining experiments
# This script automates the two-stage training framework:
# Stage 1: Self-supervised pretraining with QualiCLIP
# Stage 2: Supervised fine-tuning on IQA task

set -e  # Exit on error

echo "================================"
echo "QualiCLIP Pretraining Pipeline"
echo "================================"
echo ""

# Configuration
DATA_ROOT="/root/Perceptual-IQA-CS3324/koniq-10k"
CHECKPOINT_DIR="/root/Perceptual-IQA-CS3324/checkpoints"
MODEL_SIZE="base"  # Options: tiny, small, base
PRETRAIN_EPOCHS=10
FINETUNE_EPOCHS=50

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# ============================================
# Stage 1: QualiCLIP Self-Supervised Pretraining
# ============================================
echo "[Stage 1/2] Self-Supervised Pretraining..."
echo "This will take approximately 3-4 hours for 10 epochs"
echo ""

python pretrain_qualiclip.py \
    --data_root "$DATA_ROOT" \
    --model_size "$MODEL_SIZE" \
    --epochs "$PRETRAIN_EPOCHS" \
    --batch_size 8 \
    --lr 5e-5 \
    --num_levels 5 \
    --distortion_types blur jpeg noise brightness \
    --loss_type simplified \
    --save_dir "$CHECKPOINT_DIR" \
    --save_freq 5

# Find the latest pretrained model
PRETRAINED_MODEL=$(ls -t "$CHECKPOINT_DIR"/qualiclip_pretrain_*/swin_${MODEL_SIZE}_qualiclip_pretrained.pkl | head -1)

if [ -z "$PRETRAINED_MODEL" ]; then
    echo "Error: Pretrained model not found!"
    exit 1
fi

echo ""
echo "✓ Pretraining complete!"
echo "Pretrained model: $PRETRAINED_MODEL"
echo ""

# ============================================
# Stage 2: Supervised Fine-tuning
# ============================================
echo "[Stage 2/2] Supervised Fine-tuning..."
echo ""

# Experiment A: Baseline (No Pretraining)
echo "Running Baseline experiment (no pretraining)..."
python train_swin.py \
    --dataset koniq-10k \
    --model_size "$MODEL_SIZE" \
    --epochs "$FINETUNE_EPOCHS" \
    --batch_size 96 \
    --lr 5e-6 \
    --lr_ratio 10 \
    --weight_decay 2e-4 \
    --train_patch_num 25 \
    --test_patch_num 25 \
    --patch_size 224 \
    --train_test_num 1 \
    --use_multiscale \
    --attention_fusion \
    --drop_path_rate 0.2 \
    --dropout_rate 0.3 \
    --lr_scheduler cosine \
    --patience 10 \
    --ranking_loss_alpha 0 \
    2>&1 | tee logs/baseline_no_pretrain.log

echo ""
echo "✓ Baseline experiment complete!"
echo ""

# Experiment B: With QualiCLIP Pretraining
echo "Running experiment with QualiCLIP pretraining..."
python train_swin.py \
    --dataset koniq-10k \
    --model_size "$MODEL_SIZE" \
    --epochs "$FINETUNE_EPOCHS" \
    --batch_size 96 \
    --lr 5e-6 \
    --lr_ratio 10 \
    --weight_decay 2e-4 \
    --train_patch_num 25 \
    --test_patch_num 25 \
    --patch_size 224 \
    --train_test_num 1 \
    --use_multiscale \
    --attention_fusion \
    --drop_path_rate 0.2 \
    --dropout_rate 0.3 \
    --lr_scheduler cosine \
    --patience 10 \
    --ranking_loss_alpha 0 \
    --pretrained_encoder "$PRETRAINED_MODEL" \
    --lr_encoder_pretrained 1e-6 \
    2>&1 | tee logs/pretrained_qualiclip.log

echo ""
echo "✓ Pretrained experiment complete!"
echo ""

# ============================================
# Results Summary
# ============================================
echo "================================"
echo "All experiments complete!"
echo "================================"
echo ""
echo "Results are saved in:"
echo "  Baseline:   logs/baseline_no_pretrain.log"
echo "  Pretrained: logs/pretrained_qualiclip.log"
echo ""
echo "Compare the final SRCC/PLCC values to see the effect of QualiCLIP pretraining."
echo ""

