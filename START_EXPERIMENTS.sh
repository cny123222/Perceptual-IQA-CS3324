#!/bin/bash

echo "========================================"
echo "🔍 实验前检查"
echo "========================================"

# 1. 检查磁盘空间
echo "1. 磁盘空间:"
df -h /root | tail -1
AVAIL=$(df /root | tail -1 | awk '{print $4}')
if [ $AVAIL -lt 20000000 ]; then
    echo "   ⚠️  警告: 可用空间不足20GB"
else
    echo "   ✅ 空间充足"
fi

# 2. 检查旧进程
echo ""
echo "2. 检查旧进程:"
OLD_PROCS=$(ps aux | grep "train_swin.py" | grep -v grep | wc -l)
if [ $OLD_PROCS -gt 0 ]; then
    echo "   ⚠️  发现 $OLD_PROCS 个旧进程:"
    ps aux | grep "train_swin.py" | grep -v grep
    echo ""
    read -p "   是否终止这些进程？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -9 -f train_swin.py
        echo "   ✅ 已终止"
    else
        echo "   ❌ 请手动处理后再运行"
        exit 1
    fi
else
    echo "   ✅ 无旧进程"
fi

# 3. 检查tmux
echo ""
echo "3. 检查tmux session:"
if tmux has-session -t iqa_experiments 2>/dev/null; then
    echo "   ⚠️  session 'iqa_experiments' 已存在"
    read -p "   是否清理？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t iqa_experiments
        echo "   ✅ 已清理"
    else
        echo "   ❌ 请手动处理后再运行"
        exit 1
    fi
else
    echo "   ✅ 无冲突session"
fi

# 4. 检查GPU
echo ""
echo "4. 检查GPU:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
    echo "   GPU $line"
done

echo ""
echo "========================================"
echo "✅ 检查完成！准备启动实验"
echo "========================================"
echo ""
read -p "按Enter键开始，或Ctrl+C取消..." 

cd /root/Perceptual-IQA-CS3324
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "启动实验脚本..."
nohup ./run_experiments_fixed.sh > experiments_${TIMESTAMP}.out 2>&1 &
SCRIPT_PID=$!

sleep 3

echo ""
echo "========================================"
echo "🚀 实验已启动！"
echo "========================================"
echo "脚本PID: $SCRIPT_PID"
echo "输出日志: experiments_${TIMESTAMP}.out"
echo ""
echo "监控方法:"
echo "  1. tmux attach -t iqa_experiments"
echo "  2. tail -f experiments_${TIMESTAMP}.out"
echo "  3. tail -f logs/batch1_gpu0_lr1e6_${TIMESTAMP}.log"
echo "  4. watch -n 10 nvidia-smi"
echo ""
echo "预计完成时间: 约1小时"
echo "========================================"
