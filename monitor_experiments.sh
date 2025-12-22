#!/bin/bash

echo "═══════════════════════════════════════════════════════════════"
echo "          Experiment Monitoring Dashboard"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check running processes
echo "📊 Running Experiments:"
echo "───────────────────────────────────────────────────────────────"
ps aux | grep "train_swin.py\|train_test_IQA.py" | grep -v grep | while read line; do
    echo "$line" | awk '{
        if ($14 == "--lr") lr = $15
        if ($12 == "--model_size") model = $13
        printf "  GPU: %s, Model: %s, LR: %s, CPU: %s%%, PID: %s\n", 
               (index($0, "CUDA_VISIBLE_DEVICES=0") ? "0" : "1"), 
               model, lr, $3, $2
    }'
done
echo ""

# GPU usage
echo "🖥️  GPU Status:"
echo "───────────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | while read line; do
    echo "  $line"
done
echo ""

# Check latest logs
echo "📝 Latest Log Updates:"
echo "───────────────────────────────────────────────────────────────"

# Phase 1 logs
if [ -f "phase1_lr1e6.out" ]; then
    echo "  GPU 0 (LR 1e-6):"
    tail -3 phase1_lr1e6.out | grep -E "Epoch|Round|SRCC|median" | tail -2 | sed 's/^/    /'
fi

if [ -f "phase1_lr5e7.out" ]; then
    echo "  GPU 1 (LR 5e-7):"
    tail -3 phase1_lr5e7.out | grep -E "Epoch|Round|SRCC|median" | tail -2 | sed 's/^/    /'
fi
echo ""

# Check latest checkpoints
echo "💾 Latest Checkpoints:"
echo "───────────────────────────────────────────────────────────────"
ls -lt checkpoints/ 2>/dev/null | head -5 | tail -4 | awk '{print "  " $9 " (" $6 " " $7 " " $8 ")"}'
echo ""

# Estimated time remaining
echo "⏱️  Estimated Progress:"
echo "───────────────────────────────────────────────────────────────"
if [ -f "phase1_lr1e6.out" ]; then
    rounds_done=$(grep -c "^Round" phase1_lr1e6.out)
    total_rounds=10
    percent=$((rounds_done * 100 / total_rounds))
    echo "  Phase 1: Round $rounds_done/10 ($percent% complete)"
    
    if [ $rounds_done -gt 0 ]; then
        # Estimate time remaining (assume ~20min per round)
        remaining=$((20 * (total_rounds - rounds_done)))
        echo "  Estimated time remaining: ~${remaining} minutes"
    fi
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "Tip: Run 'watch -n 30 ./monitor_experiments.sh' for auto-refresh"
echo "═══════════════════════════════════════════════════════════════"

