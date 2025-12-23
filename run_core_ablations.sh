#!/bin/bash

# 核心消融实验：量化Swin Transformer、多尺度融合、注意力机制的独立贡献
# 
# C0: ResNet50 (baseline) - SRCC 0.907 ✅
# C1: Swin-Base only (no multiscale, no attention) - 预期 ~0.930
# C2: Swin-Base + Multiscale (no attention) - 预期 ~0.935
# C3: Swin-Base + Multiscale + Attention - SRCC 0.9378 ✅

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 核心消融实验 (Core Ablation Study)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "实验设计：正向消融（从简单到复杂）"
echo ""
echo "C1: Swin-Base only (单尺度, 无注意力)"
echo "C2: Swin-Base + 多尺度 (简单拼接, 无注意力)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查是否有正在运行的train_swin.py进程
RUNNING_PROCS=$(ps aux | grep "train_swin.py" | grep -v grep | wc -l)
if [ $RUNNING_PROCS -gt 0 ]; then
    echo "⚠️  检测到正在运行的实验："
    ps aux | grep "train_swin.py" | grep -v grep | awk '{print "  - PID " $2 ": " $NF}'
    echo ""
    echo "等待当前实验完成后再运行此脚本，或手动停止它们。"
    echo ""
    read -p "是否等待当前实验完成？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "等待中..."
        while [ $(ps aux | grep "train_swin.py" | grep -v grep | wc -l) -gt 0 ]; do
            sleep 30
            echo "  $(date '+%H:%M:%S') - 仍有 $(ps aux | grep "train_swin.py" | grep -v grep | wc -l) 个实验在运行..."
        done
        echo "✅ 所有实验已完成！"
    else
        echo "退出。请先停止当前实验。"
        exit 1
    fi
fi

# 创建tmux会话
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_NAME="core_ablations"

# 检查会话是否已存在
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Tmux会话 '$SESSION_NAME' 已存在"
    read -p "是否杀掉旧会话并创建新的？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t $SESSION_NAME
        echo "✅ 旧会话已删除"
    else
        echo "退出。请手动处理旧会话: tmux kill-session -t $SESSION_NAME"
        exit 1
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 开始运行核心消融实验"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 创建新会话
tmux new-session -d -s $SESSION_NAME -n "gpu0"
tmux new-window -t $SESSION_NAME -n "gpu1"

echo "✅ Tmux会话创建成功: $SESSION_NAME"
echo ""

# ============================================================================
# C1: Swin-Base only (GPU 0)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 C1: Swin-Base only (单尺度, 无注意力) - GPU 0"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

C1_CMD="CUDA_VISIBLE_DEVICES=0 python train_swin.py \
    --dataset koniq-10k \
    --model_size base \
    --batch_size 32 \
    --epochs 10 \
    --patience 3 \
    --train_patch_num 20 \
    --test_patch_num 20 \
    --train_test_num 1 \
    --lr 5e-7 \
    --weight_decay 2e-4 \
    --drop_path_rate 0.3 \
    --dropout_rate 0.4 \
    --lr_scheduler cosine \
    --no_multiscale \
    --ranking_loss_alpha 0 \
    --test_random_crop \
    --no_spaq \
    --no_color_jitter \
    --exp_name C1_swin_base_only \
    2>&1 | tee logs/C1_swin_base_only_${TIMESTAMP}.log"

echo "命令："
echo "$C1_CMD"
echo ""

tmux send-keys -t $SESSION_NAME:gpu0 "$C1_CMD" C-m

echo "✅ C1已启动（GPU 0）"
echo ""

# ============================================================================
# C2: Swin-Base + Multiscale (GPU 1)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 C2: Swin-Base + 多尺度 (简单拼接, 无注意力) - GPU 1"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

C2_CMD="CUDA_VISIBLE_DEVICES=1 python train_swin.py \
    --dataset koniq-10k \
    --model_size base \
    --batch_size 32 \
    --epochs 10 \
    --patience 3 \
    --train_patch_num 20 \
    --test_patch_num 20 \
    --train_test_num 1 \
    --lr 5e-7 \
    --weight_decay 2e-4 \
    --drop_path_rate 0.3 \
    --dropout_rate 0.4 \
    --lr_scheduler cosine \
    --ranking_loss_alpha 0 \
    --test_random_crop \
    --no_spaq \
    --no_color_jitter \
    --exp_name C2_swin_base_multiscale \
    2>&1 | tee logs/C2_swin_base_multiscale_${TIMESTAMP}.log"

echo "命令："
echo "$C2_CMD"
echo ""

tmux send-keys -t $SESSION_NAME:gpu1 "$C2_CMD" C-m

echo "✅ C2已启动（GPU 1）"
echo ""

# ============================================================================
# 总结
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 所有实验已启动！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 实验配置："
echo "  C1 (GPU 0): Swin-Base only"
echo "  C2 (GPU 1): Swin-Base + Multiscale"
echo ""
echo "⏱️  预计时间：每个实验约1-2小时"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 监控命令"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. 进入tmux会话："
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "2. 切换窗口："
echo "   Ctrl+B 0  # C1 (GPU 0)"
echo "   Ctrl+B 1  # C2 (GPU 1)"
echo ""
echo "3. 查看日志："
echo "   tail -f logs/C1_swin_base_only_${TIMESTAMP}.log"
echo "   tail -f logs/C2_swin_base_multiscale_${TIMESTAMP}.log"
echo ""
echo "4. 查看GPU使用："
echo "   watch -n 1 nvidia-smi"
echo ""
echo "5. 提取结果："
echo "   ./extract_core_ablation_results.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 保存命令到文件
cat > CORE_ABLATION_COMMANDS.txt << EOF
========================================
核心消融实验命令记录
时间: $(date)
========================================

C1: Swin-Base only (GPU 0)
----------------------------
$C1_CMD

C2: Swin-Base + Multiscale (GPU 1)
------------------------------------
$C2_CMD

日志位置:
---------
C1: logs/C1_swin_base_only_${TIMESTAMP}.log
C2: logs/C2_swin_base_multiscale_${TIMESTAMP}.log

监控:
-----
tmux attach -t $SESSION_NAME
watch -n 1 nvidia-smi

========================================
EOF

echo "💾 命令已保存到: CORE_ABLATION_COMMANDS.txt"
echo ""
echo "🎯 等待实验完成后，运行以下命令提取结果："
echo "   ./extract_core_ablation_results.sh"
echo ""

