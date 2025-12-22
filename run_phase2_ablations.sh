#!/bin/bash

echo "═══════════════════════════════════════════════════════════════"
echo "  Starting Phase 2: Ablation Studies (A1 + A2)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Using LR=1e-6 (best from Phase 1)"
echo ""

cd /root/Perceptual-IQA-CS3324

# GPU 0: A1 - Remove Attention
echo "🚀 Starting A1 on GPU 0: Remove Attention Fusion"
CUDA_VISIBLE_DEVICES=0 nohup python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter \
  > phase2_A1_no_attention.out 2>&1 &

echo "  PID: $!"
echo ""

# GPU 1: A2 - Remove Multi-scale
echo "🚀 Starting A2 on GPU 1: Remove Multi-scale Fusion"
CUDA_VISIBLE_DEVICES=1 nohup python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --no_multi_scale \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter \
  > phase2_A2_no_multiscale.out 2>&1 &

echo "  PID: $!"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "✅ Phase 2 started successfully!"
echo "Estimated time: ~3.4 hours"
echo ""
echo "Monitor with: ./monitor_experiments.sh"
echo "Check logs:"
echo "  tail -f phase2_A1_no_attention.out"
echo "  tail -f phase2_A2_no_multiscale.out"
echo "═══════════════════════════════════════════════════════════════"

