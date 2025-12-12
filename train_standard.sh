#!/bin/bash
# 标准训练配置（约12小时）- 推荐

export KMP_DUPLICATE_LIB_OK=TRUE

python train_test_IQA.py \
  --dataset koniq-10k \
  --epochs 10 \
  --train_test_num 2 \
  --batch_size 96 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --lr 2e-5 \
  --weight_decay 5e-4

