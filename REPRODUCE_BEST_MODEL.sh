#!/bin/bash
# 复现 Best Model (SRCC 0.9378, PLCC 0.9485)
# 从日志文件提取的真实参数: logs/swin_multiscale_ranking_alpha0_20251223_002225.log
# Checkpoint: checkpoints/koniq-10k-swin_20251223_002226/best_model_srcc_0.9378_plcc_0.9485.pkl

python scripts/train_smart_iqa.py \
    --dataset koniq-10k \
    --koniq_path /root/Perceptual-IQA-CS3324/koniq-10k \
    --model_size base \
    --attention_fusion \
    --epochs 10 \
    --lr 5e-7 \
    --lr_ratio 10 \
    --batch_size 32 \
    --weight_decay 0.0002 \
    --drop_path_rate 0.3 \
    --dropout_rate 0.3 \
    --test_patch_num 20 \
    --train_patch_num 20 \
    --patch_size 224 \
    --loss_type l1 \
    --ranking_loss_alpha 0.0 \
    --lr_scheduler cosine \
    --test_random_crop \
    --no_color_jitter \
    --no_spaq

# 关键参数说明（从真实训练日志提取）:
# - Batch Size: 32
# - Learning Rate: 5e-07 (backbone), 5e-06 (other modules, 10x ratio)
# - Weight Decay: 0.0002 (不是0.0!)
# - Train Patch Num: 20 (不是1!)
# - Test Patch Num: 20
# - Drop Path Rate: 0.3 (不是0.2!)
# - Dropout Rate: 0.4 (在solver中自动设置)
# - Test Random Crop: True
# - LR Scheduler: cosine
# - Loss: L1 (MAE), Ranking Loss Alpha = 0.0

# 预期结果：Epoch 8达到 SRCC 0.9378, PLCC 0.9485
