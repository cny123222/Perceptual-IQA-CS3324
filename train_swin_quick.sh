#!/bin/bash
# Quick test for Swin Transformer implementation
# 6 epochs, 1 round - for debugging and quick testing
# Expected training time: ~2-3 hours (with batch_size=32 on Mac MPS)
# For faster testing, use batch_size=8 but expect lower performance

export KMP_DUPLICATE_LIB_OK=TRUE

# Recommended parameters for quick test
python train_swin.py \
  --dataset koniq-10k \
  --epochs 6 \
  --train_test_num 1 \
  --batch_size 32 \
  --train_patch_num 10 \
  --test_patch_num 25

# Note: If you get memory errors, reduce batch_size to 16 or 8
# But remember: smaller batch_size = lower performance and slower convergence

