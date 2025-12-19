#!/bin/bash
# 最终推荐实验 - 2025-12-19

echo "=================================================="
echo "🚀 最终推荐实验方案"
echo "=================================================="
echo ""

# ============================================================
# 实验 A：Swin-Base + 强正则化 + 低学习率（最高优先级）⭐⭐⭐⭐⭐
# ============================================================
echo "【实验 A】Swin-Base + 强正则化 + 低学习率"
echo "目标：解决 Base 模型的过拟合问题"
echo "预期：SRCC 0.930-0.933，稳定收敛"
echo ""
echo "命令："
cat << 'EOF'
python train_swin.py \
  --dataset koniq-10k \
  --epochs 30 \
  --patience 7 \
  --batch_size 32 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --model_size base \
  --ranking_loss_alpha 0.5 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
EOF
echo ""
echo "关键变化："
echo "  - lr: 1e-5 → 5e-6 (减半，更平稳收敛)"
echo "  - weight_decay: 1e-4 → 2e-4 (翻倍，更强L2正则化)"
echo "  - drop_path_rate: 0.2 → 0.3 (+50%，更强随机深度)"
echo "  - dropout_rate: 0.3 → 0.4 (+33%，更强dropout)"
echo "  - batch_size: 保持 32 (Base 模型显存限制)"
echo ""
echo "=================================================="
echo ""

# ============================================================
# 实验 B：Swin-Base + Pure L1 Loss (alpha=0)（探索性）⭐⭐⭐⭐
# ============================================================
echo "【实验 B】Swin-Base + Pure L1 Loss (alpha=0)"
echo "目标：验证 Ranking Loss 对 Base 模型是否必要"
echo "预期：与 alpha=0.5 性能接近，约 0.930-0.933"
echo ""
echo "命令："
cat << 'EOF'
python train_swin.py \
  --dataset koniq-10k \
  --epochs 30 \
  --patience 7 \
  --batch_size 32 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --model_size base \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
EOF
echo ""
echo "关键变化："
echo "  - ranking_loss_alpha: 0.5 → 0 (纯L1损失)"
echo "  - 在 Swin-Small 上 alpha=0 vs 0.5 差异仅 0.002"
echo "  - 验证这个规律在 Base 上是否成立"
echo "  - 其他正则化参数与实验A相同（强正则化）"
echo ""
echo "=================================================="
echo ""

# ============================================================
# 快速启动脚本
# ============================================================
echo "快速启动："
echo ""
echo "1. 启动实验 A (推荐优先运行)："
echo "   bash NEXT_EXPERIMENTS.sh run_a"
echo ""
echo "2. 启动实验 B (如有空余GPU)："
echo "   bash NEXT_EXPERIMENTS.sh run_b"
echo ""
echo "3. 同时启动两个实验 (不同GPU)："
echo "   CUDA_VISIBLE_DEVICES=0 bash NEXT_EXPERIMENTS.sh run_a &"
echo "   CUDA_VISIBLE_DEVICES=1 bash NEXT_EXPERIMENTS.sh run_b &"
echo ""
echo "=================================================="

# 处理命令行参数
if [ "$1" == "run_a" ]; then
    echo ""
    echo "🚀 启动实验 A：Swin-Base + 强正则化 + 低学习率"
    echo ""
    python train_swin.py \
      --dataset koniq-10k \
      --epochs 30 \
      --patience 7 \
      --batch_size 32 \
      --train_patch_num 20 \
      --test_patch_num 20 \
      --model_size base \
      --ranking_loss_alpha 0.5 \
      --lr 5e-6 \
      --weight_decay 2e-4 \
      --drop_path_rate 0.3 \
      --dropout_rate 0.4 \
      --lr_scheduler cosine \
      --test_random_crop \
      --no_spaq
elif [ "$1" == "run_b" ]; then
    echo ""
    echo "🚀 启动实验 B：Swin-Base + Pure L1 Loss (alpha=0)"
    echo ""
    python train_swin.py \
      --dataset koniq-10k \
      --epochs 30 \
      --patience 7 \
      --batch_size 32 \
      --train_patch_num 20 \
      --test_patch_num 20 \
      --model_size base \
      --ranking_loss_alpha 0 \
      --lr 5e-6 \
      --weight_decay 2e-4 \
      --drop_path_rate 0.3 \
      --dropout_rate 0.4 \
      --lr_scheduler cosine \
      --test_random_crop \
      --no_spaq
fi

