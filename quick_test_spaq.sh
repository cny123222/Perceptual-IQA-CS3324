#!/bin/bash
# 快速测试 SPAQ 数据集（只测试少量图片和少量 patches）

# 测试参数：使用最小配置快速验证
# --epochs 1: 只跑1个epoch
# --test_patch_num 5: 每张图片只取5个patches（而不是20个）
# --batch_size 96: 保持原有batch size

python train_test_IQA.py \
    --dataset koniq-10k \
    --epochs 1 \
    --train_test_num 1 \
    --batch_size 96 \
    --train_patch_num 20 \
    --test_patch_num 5 \
    2>&1 | tee quick_test_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "快速测试完成！"
echo "注意：使用的是 test_patch_num=5 来加速测试"
echo "正式训练时请使用 test_patch_num=20"

