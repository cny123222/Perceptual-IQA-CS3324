#!/bin/bash
# 快速测试配置（约3小时）

export KMP_DUPLICATE_LIB_OK=TRUE

python train_test_IQA.py \
  --dataset koniq-10k \
  --epochs 6 \
  --train_test_num 1 \
  --batch_size 96 \
  --train_patch_num 15 \
  --test_patch_num 15 \
  --lr 2e-5 \
  --weight_decay 5e-4

