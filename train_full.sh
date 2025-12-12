#!/bin/bash
# 完整训练配置（约32小时）- 用于最终论文

export KMP_DUPLICATE_LIB_OK=TRUE

python train_test_IQA.py \
  --dataset koniq-10k \
  --epochs 16 \
  --train_test_num 3 \
  --batch_size 96 \
  --train_patch_num 25 \
  --test_patch_num 25 \
  --lr 2e-5 \
  --weight_decay 5e-4

