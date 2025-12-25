#!/bin/bash
# 复现 Best Model (SRCC 0.9378, PLCC 0.9485)
# 从日志文件提取的真实参数: logs/swin_multiscale_ranking_alpha0_20251223_002225.log
# Checkpoint: checkpoints/koniq-10k-swin_20251223_002226/best_model_srcc_0.9378_plcc_0.9485.pkl

# 注意：脚本必须从项目根目录运行，因为train_smart_iqa.py会自动找koniq-10k数据集
# koniq-10k路径会自动设置为: scripts/../koniq-10k/

cd /root/Perceptual-IQA-CS3324

python scripts/train_smart_iqa.py \
    --dataset koniq-10k \
    --model_size base \
    --attention_fusion \
    --epochs 10 \
    --lr 5e-7 \
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
# - Learning Rate: 5e-07 (backbone会自动设置为lr/10)
# - Weight Decay: 0.0002
# - Train Patch Num: 20
# - Test Patch Num: 20
# - Drop Path Rate: 0.3
# - Dropout Rate: 0.3
# - Test Random Crop: True
# - LR Scheduler: cosine
# - Loss: L1 (MAE), Ranking Loss Alpha = 0.0
# - 数据集路径: 自动从scripts/目录向上查找 ../koniq-10k/

# 预期结果：Epoch 8达到 SRCC 0.9378, PLCC 0.9485
