#!/bin/bash
# 复现 Best Model (SRCC 0.9378, PLCC 0.9485)
# 日志文件: logs/swin_multiscale_ranking_alpha0_20251223_002225.log
# Checkpoint: checkpoints/koniq-10k-swin_20251223_002226/best_model_srcc_0.9378_plcc_0.9485.pkl
# 训练时间: 2025-12-23 00:22:25
# Epoch 8 达到 SRCC 0.9378

# 必须从项目根目录运行
cd /root/Perceptual-IQA-CS3324

# 从日志文件 logs/swin_multiscale_ranking_alpha0_20251223_002225.log 提取的完整配置:
# Dataset:                    koniq-10k
# Model Size:                 base
# Epochs:                     10
# Batch Size:                 32
# Learning Rate:              5e-07
# LR Ratio (backbone):        10
# Weight Decay:               0.0002
# Train Patch Num:            20
# Test Patch Num:             20
# Loss: L1 (MAE)
# Ranking Loss Alpha:         0.0
# Drop Path Rate:             0.3
# Dropout Rate:               0.4 (在solver中设置，日志显示0.3但实际用0.4)
# LR Scheduler:               cosine
# Test Random Crop:           True
# SPAQ Cross-Dataset Test:    False

# All parameters now use default values from train_smart_iqa.py
# Simply run the script without any arguments to reproduce the best model
python scripts/train_smart_iqa.py

# Or equivalently with all parameters explicitly specified:
# python scripts/train_smart_iqa.py \
#     --dataset koniq-10k \
#     --model_size base \
#     --attention_fusion \
#     --epochs 10 \
#     --lr 5e-7 \
#     --batch_size 32 \
#     --weight_decay 0.0002 \
#     --drop_path_rate 0.3 \
#     --dropout_rate 0.4 \
#     --test_patch_num 20 \
#     --train_patch_num 20 \
#     --patch_size 224 \
#     --loss_type l1 \
#     --ranking_loss_alpha 0.0 \
#     --lr_scheduler cosine \
#     --test_random_crop \
#     --no_color_jitter \
#     --no_spaq

# 说明:
# 1. 所有参数已设置为默认值，直接运行即可复现 best 模型
# 2. 数据集路径自动计算: scripts/../koniq-10k/ → /root/Perceptual-IQA-CS3324/koniq-10k/
# 3. lr_ratio 默认为 10，无需指定
# 4. 预期结果: Epoch 8 达到 SRCC 0.9378, PLCC 0.9485

# 简化运行方式（推荐）:
# bash REPRODUCE_BEST_MODEL.sh
# 
# 或者直接运行:
# cd /root/Perceptual-IQA-CS3324 && python scripts/train_smart_iqa.py
