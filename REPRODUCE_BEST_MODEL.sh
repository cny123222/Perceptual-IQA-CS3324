#!/bin/bash
# 复现 Best Model (SRCC 0.9378, PLCC 0.9485)
# 配置: Swin-Base + Attention, LR 5e-7, Batch 32, Epochs 10

python scripts/train_smart_iqa.py \
    --dataset koniq-10k \
    --koniq_path ./koniq-10k \
    --model_size base \
    --use_attention \
    --epochs 10 \
    --lr 5e-7 \
    --lr_ratio 10 \
    --batch_size 32 \
    --weight_decay 0.0 \
    --drop_path_rate 0.2 \
    --test_patch_num 20 \
    --train_patch_num 1 \
    --patch_size 224 \
    --loss_type mae \
    --ranking_loss_alpha 0.0 \
    --lr_scheduler_type cosine \
    --use_lr_scheduler \
    --test_random_crop

# 注意：
# - 不加 --test_spaq 参数（默认就是不测试SPAQ）
# - 加 --test_random_crop 参数使用原始的随机裁剪方式
# - Batch size 32 (原实验配置)
# - 预期结果：Epoch 8达到 SRCC 0.9378, PLCC 0.9485
