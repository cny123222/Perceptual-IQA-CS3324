#!/bin/bash
# Standard training with Swin Transformer backbone
# 10 epochs, 2 rounds - for validation and comparison with baseline
# Expected training time: ~6-8 hours

export KMP_DUPLICATE_LIB_OK=TRUE

python train_swin.py \
  --dataset koniq-10k \
  --epochs 10 \
  --train_test_num 2 \
  --batch_size 96 \
  --train_patch_num 25 \
  --test_patch_num 25

