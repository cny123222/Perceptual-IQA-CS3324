#!/bin/bash
# 原论文完整配置（epochs=16, train_test_num=10）
# 预计耗时：约120小时（5天）

export KMP_DUPLICATE_LIB_OK=TRUE

python train_test_IQA.py \
  --dataset koniq-10k \
  --epochs 16 \
  --train_test_num 10 \
  --batch_size 96 \
  --train_patch_num 25 \
  --test_patch_num 25 \
  --lr 2e-5 \
  --weight_decay 5e-4 \
  --lr_ratio 10 \
  --patch_size 224

