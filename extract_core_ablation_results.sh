#!/bin/bash

# 提取核心消融实验结果

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 核心消融实验结果提取"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# C0: ResNet50
echo "C0 (ResNet50 Baseline):"
echo "  SRCC: 0.907"
echo "  PLCC: ~0.918"
echo ""

# C1: Swin-Base only
echo "C1 (Swin-Base only, 单尺度, 无注意力):"
C1_LOG=$(ls -t logs/C1_swin_base_only_*.log 2>/dev/null | head -1)
if [ -n "$C1_LOG" ]; then
    C1_SRCC=$(grep "Best test SRCC" "$C1_LOG" | tail -1 | awk '{print $4}' | sed 's/,//')
    C1_PLCC=$(grep "Best test SRCC" "$C1_LOG" | tail -1 | awk '{print $6}')
    if [ -n "$C1_SRCC" ]; then
        echo "  SRCC: $C1_SRCC"
        echo "  PLCC: $C1_PLCC"
        echo "  日志: $C1_LOG"
    else
        echo "  ⏳ 实验尚未完成或结果未找到"
        echo "  日志: $C1_LOG"
    fi
else
    echo "  ❌ 日志文件未找到"
fi
echo ""

# C2: Swin-Base + Multiscale
echo "C2 (Swin-Base + 多尺度, 无注意力):"
C2_LOG=$(ls -t logs/C2_swin_base_multiscale_*.log 2>/dev/null | head -1)
if [ -n "$C2_LOG" ]; then
    C2_SRCC=$(grep "Best test SRCC" "$C2_LOG" | tail -1 | awk '{print $4}' | sed 's/,//')
    C2_PLCC=$(grep "Best test SRCC" "$C2_LOG" | tail -1 | awk '{print $6}')
    if [ -n "$C2_SRCC" ]; then
        echo "  SRCC: $C2_SRCC"
        echo "  PLCC: $C2_PLCC"
        echo "  日志: $C2_LOG"
    else
        echo "  ⏳ 实验尚未完成或结果未找到"
        echo "  日志: $C2_LOG"
    fi
else
    echo "  ❌ 日志文件未找到"
fi
echo ""

# C3: Full model (baseline)
echo "C3 (Swin-Base + 多尺度 + 注意力, 完整版本):"
echo "  SRCC: 0.9378"
echo "  PLCC: 0.9485"
echo "  Checkpoint: checkpoints/koniq-10k-swin_20251223_002226/"
echo ""

# Calculate contributions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 贡献分析"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -n "$C1_SRCC" ] && [ -n "$C2_SRCC" ]; then
    # Calculate improvements using bc for floating point
    C0_SRCC=0.907
    C3_SRCC=0.9378
    
    SWIN_CONTRIB=$(echo "scale=4; ($C1_SRCC - $C0_SRCC)" | bc)
    MULTISCALE_CONTRIB=$(echo "scale=4; ($C2_SRCC - $C1_SRCC)" | bc)
    ATTENTION_CONTRIB=$(echo "scale=4; ($C3_SRCC - $C2_SRCC)" | bc)
    TOTAL=$(echo "scale=4; ($C3_SRCC - $C0_SRCC)" | bc)
    
    SWIN_PCT=$(echo "scale=1; ($SWIN_CONTRIB / $TOTAL) * 100" | bc)
    MULTISCALE_PCT=$(echo "scale=1; ($MULTISCALE_CONTRIB / $TOTAL) * 100" | bc)
    ATTENTION_PCT=$(echo "scale=1; ($ATTENTION_CONTRIB / $TOTAL) * 100" | bc)
    
    echo "总提升: $TOTAL (+$(echo "scale=2; $TOTAL * 100" | bc)%)"
    echo ""
    echo "组件贡献："
    echo "  1. Swin Transformer:  $SWIN_CONTRIB (+$(echo "scale=2; $SWIN_CONTRIB * 100" | bc)%) - ${SWIN_PCT}%"
    echo "  2. 多尺度融合:        $MULTISCALE_CONTRIB (+$(echo "scale=2; $MULTISCALE_CONTRIB * 100" | bc)%) - ${MULTISCALE_PCT}%"
    echo "  3. 注意力机制:        $ATTENTION_CONTRIB (+$(echo "scale=2; $ATTENTION_CONTRIB * 100" | bc)%) - ${ATTENTION_PCT}%"
else
    echo "⏳ 等待C1和C2实验完成以计算贡献..."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

