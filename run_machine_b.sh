#!/bin/bash
# Machine B Experiments Runner
# Runs: D1, D2, D4, E1, E3, E4

cd /root/Perceptual-IQA-CS3324

echo "=========================================="
echo "Starting Machine B Experiments"
echo "Total: 6 experiments"
echo "Estimated time: 30-60 minutes"
echo "=========================================="
echo ""

# D1: WD=5e-5
echo "========== [1/6] Starting D1: WD=5e-5 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --lr 5e-6 \
  --weight_decay 5e-5 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

echo "✅ D1 completed!"
echo ""

# D2: WD=1e-4
echo "========== [2/6] Starting D2: WD=1e-4 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --lr 5e-6 \
  --weight_decay 1e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

echo "✅ D2 completed!"
echo ""

# D4: WD=4e-4
echo "========== [3/6] Starting D4: WD=4e-4 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --lr 5e-6 \
  --weight_decay 4e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

echo "✅ D4 completed!"
echo ""

# E1: LR=2.5e-6
echo "========== [4/6] Starting E1: LR=2.5e-6 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --lr 2.5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

echo "✅ E1 completed!"
echo ""

# E3: LR=7.5e-6
echo "========== [5/6] Starting E3: LR=7.5e-6 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --lr 7.5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

echo "✅ E3 completed!"
echo ""

# E4: LR=1e-5
echo "========== [6/6] Starting E4: LR=1e-5 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --lr 1e-5 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

echo "✅ E4 completed!"
echo ""

echo "=========================================="
echo "🎉 All Machine B experiments completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check logs in logs/ directory"
echo "2. Commit results: git add logs/*.log && git commit -m 'feat: Machine B results' && git push"
echo "3. (Optional) Transfer best checkpoints to Machine A"

