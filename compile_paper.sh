#!/bin/bash
# 正确的LaTeX编译脚本 - 确保字体正确嵌入

cd IEEE-conference-template-062824

echo "清理旧文件..."
rm -f *.aux *.log *.out *.bbl *.blg *.toc

echo "第一次编译..."
pdflatex -interaction=nonstopmode IEEE-conference-template-062824.tex

echo "第二次编译 (解决交叉引用)..."
pdflatex -interaction=nonstopmode IEEE-conference-template-062824.tex

echo "✓ 编译完成！"
echo "输出文件: IEEE-conference-template-062824.pdf"

# 检查PDF
if [ -f "IEEE-conference-template-062824.pdf" ]; then
    echo "✓ PDF生成成功"
    ls -lh IEEE-conference-template-062824.pdf
else
    echo "❌ PDF生成失败"
    exit 1
fi

