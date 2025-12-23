# IEEE会议论文LaTeX模板使用指南

## 📚 模板文件说明

### 核心文件
- **IEEE-conference-template-062824.tex** - 主要的LaTeX源文件（您要编辑的）
- **IEEEtran.cls** - IEEE格式类文件（不要修改）
- **IEEE-conference-template-062824.pdf** - 编译后的PDF示例
- **IEEEtran_HOWTO.pdf** - 详细的使用指南

---

## 🏗️ 论文结构（按顺序）

### 1. 文档类和包引入（前14行）
```latex
\documentclass[conference]{IEEEtran}  % IEEE会议格式
\usepackage{cite}                      % 引用管理
\usepackage{amsmath,amssymb,amsfonts} % 数学符号
\usepackage{algorithmic}               % 算法环境
\usepackage{graphicx}                  % 图片插入
\usepackage{textcomp}                  % 特殊符号
\usepackage{xcolor}                    % 颜色支持
```

### 2. 标题和作者信息（15-59行）
```latex
\title{您的论文标题}

\author{
    \IEEEauthorblockN{姓名1}
    \IEEEauthorblockA{学校/单位\\邮箱}
    \and
    \IEEEauthorblockN{姓名2}
    \IEEEauthorblockA{学校/单位\\邮箱}
}

\maketitle
```

### 3. Abstract（摘要）（61-65行）
```latex
\begin{abstract}
简短总结您的工作（150-200词）
- 问题是什么
- 您做了什么
- 主要结果
\end{abstract}
```

### 4. Keywords（关键词）（67-69行）
```latex
\begin{IEEEkeywords}
image quality assessment, transformer, multi-scale fusion
\end{IEEEkeywords}
```

### 5. Introduction（引言）（71-83行）
```latex
\section{Introduction}
- 研究背景
- 问题重要性
- 现有方法局限
- 您的贡献
```

### 6. Related Work（相关工作）
```latex
\section{Related Work}
- 传统IQA方法
- 深度学习IQA
- Transformer在视觉中的应用
```

### 7. Method（方法）
```latex
\section{Proposed Method}
\subsection{Overall Architecture}
\subsection{Swin Transformer Backbone}
\subsection{Multi-scale Feature Fusion}
\subsection{Attention Mechanism}
```

### 8. Experiments（实验）
```latex
\section{Experiments}
\subsection{Experimental Setup}
\subsection{Ablation Study}
\subsection{Comparison with State-of-the-art}
```

### 9. Results and Discussion（结果和讨论）
```latex
\section{Results and Discussion}
- 消融实验结果
- 与SOTA对比
- 可视化分析
```

### 10. Conclusion（结论）
```latex
\section{Conclusion}
- 总结贡献
- 未来工作
```

### 11. References（参考文献）（275-287行）
```latex
\begin{thebibliography}{00}
\bibitem{b1} 作者, "标题," 期刊, 年份.
\end{thebibliography}
```

---

## 📊 插入表格和图片

### 表格示例（214-229行）
```latex
\begin{table}[htbp]
\caption{表格标题}
\begin{center}
\begin{tabular}{|c|c|c|}
\hline
\textbf{列1} & \textbf{列2} & \textbf{列3} \\
\hline
数据1 & 数据2 & 数据3 \\
\hline
\end{tabular}
\label{tab:your_label}
\end{center}
\end{table}
```

**对应您的实验**：
```latex
\begin{table}[htbp]
\caption{Architecture Ablation Study}
\begin{center}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Method} & \textbf{Backbone} & \textbf{SRCC} & \textbf{PLCC} \\
\hline
HyperIQA & ResNet50 & 0.907 & 0.918 \\
Ours-C1 & Swin-Base & 0.9338 & 0.9445 \\
Ours-C2 & Swin+Multi & 0.9353 & 0.9469 \\
\textbf{Ours-C3} & \textbf{Full} & \textbf{0.9378} & \textbf{0.9485} \\
\hline
\end{tabular}
\label{tab:ablation}
\end{center}
\end{table}
```

### 图片示例（231-235行）
```latex
\begin{figure}[htbp]
\centerline{\includegraphics[width=\columnwidth]{your_figure.png}}
\caption{图片标题}
\label{fig:your_label}
\end{figure}
```

---

## 🔢 数学公式

### 行内公式
```latex
我们的损失函数是 $L = L_{MAE} + \alpha L_{rank}$
```

### 独立公式（116-118行）
```latex
\begin{equation}
L_{total} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
\label{eq:loss}
\end{equation}
```

**引用公式**：
```latex
如方程 \eqref{eq:loss} 所示...
```

---

## 📝 引用文献

### 在文中引用（256-273行）
```latex
Swin Transformer \cite{liu2021swin} 展示了...
多个引用 \cite{liu2021swin, dosovitskiy2020vit}
```

### 添加参考文献（275-287行）
```latex
\begin{thebibliography}{00}
\bibitem{liu2021swin} 
Z. Liu et al., ``Swin Transformer: Hierarchical vision transformer using shifted windows,'' 
in ICCV, 2021.

\bibitem{su2020hyperIQA}
S. Su et al., ``Blindly assess image quality in the wild guided by a self-adaptive hyper network,'' 
in CVPR, 2020.
\end{thebibliography}
```

---

## 🎯 适配您的IQA论文

### 建议的章节结构

```latex
\section{Introduction}
  - IQA任务的重要性
  - 现有方法的局限（ResNet50-based HyperIQA）
  - 我们的贡献：Swin Transformer + 多尺度 + 注意力

\section{Related Work}
  \subsection{Image Quality Assessment}
  \subsection{Transformer in Vision}
  \subsection{Multi-scale Feature Fusion}

\section{Proposed Method}
  \subsection{Overall Framework}
  \subsection{Swin Transformer Backbone}
  \subsection{Multi-scale Feature Extraction}
  \subsection{Attention-based Fusion}
  \subsection{Training Strategy}

\section{Experiments}
  \subsection{Experimental Setup}
    - Dataset: KonIQ-10k
    - Training details: LR=5e-7, epochs=10, etc.
  
  \subsection{Ablation Study}
    - Table 1: Architecture ablation (C0→C1→C2→C3)
    - Analysis: Swin贡献87%
  
  \subsection{Learning Rate Sensitivity}
    - Table 2: LR experiments
    - Figure: 倒U型曲线
  
  \subsection{Model Size Comparison}
    - Table 3: Tiny/Small/Base
  
  \subsection{Comparison with State-of-the-art}
    - Table 4: vs 其他IQA方法

\section{Conclusion}
  - 总结：Swin是核心，5e-7是最优LR
  - 未来工作：更大的数据集，视频质量评估

\section*{References}
```

---

## 💡 使用技巧

### 1. 编译命令
```bash
cd IEEE-conference-template-062824
pdflatex IEEE-conference-template-062824.tex
pdflatex IEEE-conference-template-062824.tex  # 编译两次更新引用
```

### 2. 图片格式
- 推荐使用 **PDF** 或 **PNG** 格式
- 图片放在与.tex同目录或子目录
- 双栏格式：单栏宽度约3.5英寸，双栏宽度约7英寸

### 3. 表格位置控制
- `[htbp]`: here, top, bottom, page
- `[t]`: 只放在页面顶部
- `[h]`: 尽量放在当前位置

### 4. 交叉引用
```latex
如表 \ref{tab:ablation} 所示...
如图 \ref{fig:architecture} 所示...
根据方程 \eqref{eq:loss}...
```

### 5. 页面限制
- IEEE会议论文通常限制：**6-8页**
- 摘要：150-200词
- 关键词：3-5个

---

## ⚠️ 注意事项

### 必须修改的地方
1. ✅ 标题
2. ✅ 作者信息
3. ✅ Abstract
4. ✅ Keywords
5. ✅ 正文内容
6. ✅ 参考文献

### 不要修改
1. ❌ `\documentclass[conference]{IEEEtran}`
2. ❌ 页边距、字体大小
3. ❌ IEEEtran.cls 文件

### 最后检查
- 🔍 删除所有模板提示文字（红色警告文字）
- 🔍 检查所有图表是否正确引用
- 🔍 确保参考文献格式统一
- 🔍 拼写和语法检查

---

## 📖 下一步

1. **熟悉模板**：先编译一次看看效果
2. **准备内容**：整理实验结果、图表
3. **分段写作**：先写Method和Experiments（最容易）
4. **逐步完善**：Introduction → Related Work → Conclusion
5. **反复修改**：检查逻辑、语言、格式

**您现在可以：**
- 先编译一次模板看效果
- 开始填写标题和作者信息
- 把实验结果表格先做出来

需要我帮您开始写哪一部分吗？

