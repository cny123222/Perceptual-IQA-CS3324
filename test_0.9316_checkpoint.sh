#!/bin/bash
# 测试 SRCC 0.9316 (Round 1 最佳) 的跨数据集性能

cd /root/Perceptual-IQA-CS3324 && python cross_dataset_test.py \
  --checkpoint checkpoints/koniq-10k-swin-ranking-alpha0.5_20251221_003537/best_model_srcc_0.9316_plcc_0.9450.pkl \
  --model_size base \
  --test_patch_num 20 \
  --test_random_crop \
  2>&1 | tee logs/swin_base_0.9316_cross_dataset_test.log

echo ""
echo "============================================================"
echo "✅ 测试完成！结果已保存到:"
echo "   logs/swin_base_0.9316_cross_dataset_test.log"
echo "============================================================"
