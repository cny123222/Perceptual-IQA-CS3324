#!/bin/bash
# 完整的LaTeX编译脚本 - 包含参考文献处理

cd IEEE-conference-template-062824

echo "清理旧文件..."
rm -f *.aux *.log *.out *.bbl *.blg *.toc

echo "第一次编译 (生成aux文件)..."
pdflatex -interaction=nonstopmode IEEE-conference-template-062824.tex > /tmp/latex1.log 2>&1

echo "处理参考文献 (bibtex)..."
bibtex IEEE-conference-template-062824 > /tmp/bibtex.log 2>&1

echo "第二次编译 (插入引用)..."
pdflatex -interaction=nonstopmode IEEE-conference-template-062824.tex > /tmp/latex2.log 2>&1

echo "第三次编译 (解决交叉引用)..."
pdflatex -interaction=nonstopmode IEEE-conference-template-062824.tex > /tmp/latex3.log 2>&1

echo ""
echo "✓ 编译完成！"
echo ""

# 检查PDF
if [ -f "IEEE-conference-template-062824.pdf" ]; then
    echo "✅ PDF生成成功"
    ls -lh IEEE-conference-template-062824.pdf
    
    # 检查是否有引用
    if [ -f "IEEE-conference-template-062824.bbl" ]; then
        echo "✅ 参考文献已处理"
        echo "   参考文献数量: $(grep -c "\\bibitem" IEEE-conference-template-062824.bbl)"
    else
        echo "⚠️  参考文献文件未生成"
    fi
else
    echo "❌ PDF生成失败"
    echo "检查日志: /tmp/latex*.log"
    exit 1
fi

echo ""
echo "日志文件:"
echo "  /tmp/latex1.log - 第一次编译"
echo "  /tmp/bibtex.log - 参考文献处理"
echo "  /tmp/latex2.log - 第二次编译"
echo "  /tmp/latex3.log - 第三次编译"

