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
# 实验 B：Swin-Small + Attention Fusion（探索性）⭐⭐⭐⭐
# ============================================================
echo "【实验 B】Swin-Small + Attention Fusion"
echo "目标：验证注意力机制在更大模型上的效果"
echo "预期：成功则 0.932-0.935，失败则 0.925"
echo ""
echo "命令："
cat << 'EOF'
python train_swin.py \
  --dataset koniq-10k \
  --epochs 30 \
  --patience 7 \
  --batch_size 64 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --model_size small \
  --ranking_loss_alpha 0.5 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --attention_fusion
EOF
echo ""
echo "关键变化："
echo "  - 添加 --attention_fusion（注意力加权多尺度融合）"
echo "  - 在 Swin-Tiny 上失败了（容量不足，-0.28%）"
echo "  - 在 Swin-Small 上可能成功（容量充足）"
echo "  - batch_size=64（Small 显存充足）"
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
    echo "🚀 启动实验 B：Swin-Small + Attention Fusion"
    echo ""
    python train_swin.py \
      --dataset koniq-10k \
      --epochs 30 \
      --patience 7 \
      --batch_size 64 \
      --train_patch_num 20 \
      --test_patch_num 20 \
      --model_size small \
      --ranking_loss_alpha 0.5 \
      --lr 1e-5 \
      --weight_decay 1e-4 \
      --drop_path_rate 0.2 \
      --dropout_rate 0.3 \
      --lr_scheduler cosine \
      --test_random_crop \
      --no_spaq \
      --attention_fusion
fi

