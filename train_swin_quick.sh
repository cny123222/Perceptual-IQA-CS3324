#!/bin/bash
# Quick test for Swin Transformer implementation
# 6 epochs, 1 round - for debugging and quick testing
# Expected training time: ~20-30 minutes

export KMP_DUPLICATE_LIB_OK=TRUE

python train_swin.py \
  --dataset koniq-10k \
  --epochs 6 \
  --train_test_num 1 \
  --batch_size 96 \
  --train_patch_num 10 \
  --test_patch_num 25

