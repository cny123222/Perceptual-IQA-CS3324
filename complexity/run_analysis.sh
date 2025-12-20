#!/bin/bash
# 一键运行复杂度分析脚本

echo "=================================="
echo "模型复杂度分析"
echo "=================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "complexity/compute_complexity.py" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    echo "   cd /root/Perceptual-IQA-CS3324"
    echo "   bash complexity/run_analysis.sh"
    exit 1
fi

# 检查测试图片
if [ ! -f "complexity/example.JPG" ]; then
    echo "❌ 错误：找不到测试图片 complexity/example.JPG"
    exit 1
fi

# 检查 checkpoint
CHECKPOINT="checkpoints/koniq-10k-swin-ranking-alpha0.5_20251220_091014/best_model_srcc_0.9336_plcc_0.9464.pkl"
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误：找不到模型 checkpoint"
    echo "   路径：$CHECKPOINT"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

# 询问用户选择
echo "请选择运行模式："
echo "  1. 快速测试（推荐，无需额外依赖）"
echo "  2. 完整分析（需要安装 ptflops, thop）"
echo ""
read -p "请输入选择 [1/2，默认 1]: " choice

choice=${choice:-1}

if [ "$choice" == "1" ]; then
    echo ""
    echo "=================================="
    echo "运行快速测试..."
    echo "=================================="
    echo ""
    python complexity/quick_test.py
    
elif [ "$choice" == "2" ]; then
    echo ""
    echo "=================================="
    echo "检查依赖..."
    echo "=================================="
    
    # 检查是否安装了必要的库
    python -c "import ptflops" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "⚠️  ptflops 未安装"
        read -p "是否立即安装？[y/N]: " install
        if [ "$install" == "y" ] || [ "$install" == "Y" ]; then
            pip install ptflops thop fvcore
        else
            echo "❌ 缺少依赖，退出"
            exit 1
        fi
    fi
    
    echo ""
    echo "=================================="
    echo "运行完整分析..."
    echo "=================================="
    echo ""
    python complexity/compute_complexity.py
    
else
    echo "❌ 无效的选择"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ 分析完成！"
echo "=================================="

if [ "$choice" == "2" ]; then
    if [ -f "complexity/complexity_results.md" ]; then
        echo ""
        echo "📄 结果已保存到："
        echo "   complexity/complexity_results.md"
        echo ""
        echo "查看结果："
        echo "   cat complexity/complexity_results.md"
    fi
fi

