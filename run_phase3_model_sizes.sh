#!/bin/bash

echo "═══════════════════════════════════════════════════════════════"
echo "  Starting Phase 3: Model Size Comparison (B1 + B2)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Using LR=1e-6 (best from Phase 1)"
echo ""

cd /root/Perceptual-IQA-CS3324

# GPU 0: B1 - Swin-Tiny
echo "🚀 Starting B1 on GPU 0: Swin-Tiny"
CUDA_VISIBLE_DEVICES=0 nohup python train_swin.py \
  --dataset koniq-10k \
  --model_size tiny \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter \
  > phase3_B1_tiny.out 2>&1 &

echo "  PID: $!"
echo ""

# GPU 1: B2 - Swin-Small
echo "🚀 Starting B2 on GPU 1: Swin-Small"
CUDA_VISIBLE_DEVICES=1 nohup python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.25 \
  --dropout_rate 0.35 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter \
  > phase3_B2_small.out 2>&1 &

echo "  PID: $!"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "✅ Phase 3 started successfully!"
echo "Estimated time: ~3.2 hours"
echo ""
echo "Monitor with: ./monitor_experiments.sh"
echo "Check logs:"
echo "  tail -f phase3_B1_tiny.out"
echo "  tail -f phase3_B2_small.out"
echo "═══════════════════════════════════════════════════════════════"

