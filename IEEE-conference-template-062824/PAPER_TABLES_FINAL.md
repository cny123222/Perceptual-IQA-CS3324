# SMART-IQA 论文表格 - 最终版本 📊

**日期**: 2024-12-24  
**说明**: 这些表格可以直接复制到LaTeX论文中

---

## 表1: SOTA对比 - KonIQ-10k性能对比

**用途**: 展示我们的方法与现有SOTA方法的对比

```latex
\begin{table*}[t]
\centering
\caption{Performance comparison with state-of-the-art methods on KonIQ-10k dataset}
\label{tab:sota_comparison}
\begin{tabular}{lccccc}
\hline
Method & Year & Backbone & SRCC & PLCC & Params \\
\hline
NIMA \cite{talebi2018nima} & 2018 & InceptionNet & 0.558 & 0.590 & - \\
DBCNN & 2020 & ResNet50 & 0.875 & 0.884 & - \\
PaQ-2-PiQ \cite{ying2020paq2piq} & 2020 & ResNet18 & 0.892 & 0.904 & - \\
HyperIQA \cite{su2020hyperiq} & 2020 & ResNet50 & 0.906 & 0.917 & 25M \\
MUSIQ \cite{ke2021musiq} & 2021 & Multi-scale ViT & 0.915 & 0.930 & 150M \\
TReS \cite{golestaneh2022tres} & 2022 & Transformer & 0.908 & 0.922 & - \\
MANIQA \cite{yang2022maniqa} & 2022 & ViT-Small & 0.920 & 0.930 & 46M \\
LIQE \cite{zhang2023liqe} & 2023 & CLIP & 0.919 & 0.927 & 120M \\
Q-Align \cite{wu2023qalign} & 2023 & mPLUG-Owl & 0.921 & 0.933 & - \\
\hline
\textbf{SMART-IQA (Ours)} & 2024 & Swin-Base & \textbf{0.9378} & \textbf{0.9485} & 88M \\
\hline
\textbf{Improvement over Best} & - & - & \textbf{+1.68\%} & \textbf{+1.55\%} & - \\
\textbf{Improvement over HyperIQA} & - & - & \textbf{+3.18\%} & \textbf{+3.15\%} & - \\
\hline
\end{tabular}
\end{table*}
```

**说明**: 
- 表头 `\begin{table*}` 表示双栏宽度，适合会议论文
- 需要引用相应的文献（已在references.bib中）
- **加粗**数字表示最佳结果

---

## 表2: 消融实验 - 组件贡献分析

**用途**: 展示每个组件的贡献

```latex
\begin{table}[t]
\centering
\caption{Ablation study on KonIQ-10k: component contribution analysis}
\label{tab:ablation_study}
\begin{tabular}{lcccc}
\hline
Configuration & Backbone & Multi-Scale & Attention & SRCC & PLCC \\
\hline
\multicolumn{6}{l}{\textit{Baseline}} \\
C0: HyperIQA & ResNet50 & - & - & 0.9070 & 0.9180 \\
\hline
\multicolumn{6}{l}{\textit{Progressive Ablation (Swin-Base)}} \\
A2: Backbone only & Swin-Base & \xmark & \xmark & 0.9338 & 0.9438 \\
A1: + Multi-Scale & Swin-Base & \cmark & \xmark & 0.9353 & 0.9458 \\
E6: + Attention & Swin-Base & \cmark & \cmark & \textbf{0.9378} & \textbf{0.9485} \\
\hline
\multicolumn{6}{l}{\textit{Component Contributions}} \\
Swin Transformer & \multicolumn{4}{l}{+0.0268 SRCC (+2.68\%, 87\% of total gain)} \\
Multi-Scale Fusion & \multicolumn{4}{l}{+0.0015 SRCC (+0.15\%, 5\% of total gain)} \\
Attention Mechanism & \multicolumn{4}{l}{+0.0025 SRCC (+0.25\%, 8\% of total gain)} \\
\hline
\textbf{Total Improvement} & \multicolumn{4}{l}{\textbf{+0.0308 SRCC (+3.08\%)}} \\
\hline
\end{tabular}
\end{table}
```

**说明**:
- 使用 `\cmark` 和 `\xmark` 需要在LaTeX序言中添加: `\usepackage{amssymb}`
- 或者用 `\checkmark` 和 `-` 代替
- 清楚展示了渐进式消融的过程

---

## 表3: 跨数据集泛化能力

**用途**: 对比HyperIQA和SMART-IQA的泛化能力

```latex
\begin{table}[t]
\centering
\caption{Cross-dataset generalization performance (trained on KonIQ-10k)}
\label{tab:cross_dataset}
\begin{tabular}{lccccc}
\hline
\multirow{2}{*}{Dataset} & \multirow{2}{*}{Type} & \multicolumn{2}{c}{HyperIQA} & \multicolumn{2}{c}{SMART-IQA (Ours)} \\
\cline{3-6}
& & SRCC & PLCC & SRCC & PLCC \\
\hline
KonIQ-10k & In-domain & 0.9060 & 0.9170 & \textbf{0.9378} & \textbf{0.9485} \\
\hline
\multicolumn{6}{l}{\textit{Cross-dataset Evaluation}} \\
SPAQ & Smartphone & 0.8490 & 0.8465 & \textbf{0.8698} & \textbf{0.8709} \\
KADID-10K & Synthetic & 0.4848 & 0.5160 & \textbf{0.5412} & \textbf{0.5591} \\
AGIQA-3K & AI-generated & 0.6627 & 0.7236 & 0.6484 & 0.6830 \\
\hline
\textbf{Avg (Cross)} & - & 0.6655 & 0.6954 & \textbf{0.6865} & \textbf{0.7044} \\
\hline
\multicolumn{6}{l}{\textit{Improvement}} \\
In-domain & - & \multicolumn{2}{c}{-} & \multicolumn{2}{c}{+3.18\% / +3.15\%} \\
Cross-domain & - & \multicolumn{2}{c}{-} & \multicolumn{2}{c}{+2.10\% / +0.90\%} \\
\hline
\end{tabular}
\end{table}
```

**说明**:
- 使用 `\multirow` 需要添加包: `\usepackage{multirow}`
- 清楚展示了在不同类型数据集上的泛化能力
- AGIQA-3K我们反而略差，这是诚实的展示，可以在正文中讨论

---

## 表4: 模型大小对比 - 精度与效率权衡

**用途**: 展示不同模型大小的性能-效率权衡

```latex
\begin{table}[t]
\centering
\caption{Performance-efficiency trade-off across model sizes on KonIQ-10k}
\label{tab:model_size}
\begin{tabular}{lcccccc}
\hline
Model & Params & FLOPs & SRCC & PLCC & Time & FPS \\
\hline
\multicolumn{7}{l}{\textit{Baseline}} \\
HyperIQA & 25M & 4.0G & 0.9070 & 0.9180 & - & $\sim$100 \\
\hline
\multicolumn{7}{l}{\textit{SMART-IQA Variants}} \\
Tiny & 28M & $\sim$5G & 0.9249 & 0.9360 & 1.5h & $\sim$25 \\
Small & 50M & $\sim$11G & 0.9338 & 0.9455 & 1.7h & $\sim$23 \\
\textbf{Base} & 88M & 18.2G & \textbf{0.9378} & \textbf{0.9485} & 1.7h & $\sim$22 \\
\hline
\multicolumn{7}{l}{\textit{Analysis: Base vs Small}} \\
Params Reduction & \multicolumn{6}{l}{-43\% parameters, only -0.40\% SRCC loss} \\
\multicolumn{7}{l}{\textit{Analysis: Base vs Tiny}} \\
Params Reduction & \multicolumn{6}{l}{-68\% parameters, -1.29\% SRCC loss} \\
\hline
\multicolumn{7}{l}{\textit{Recommendation:}} \\
\multicolumn{7}{l}{• Base: Best accuracy for research and benchmarking} \\
\multicolumn{7}{l}{• Small: Best balance for deployment (43\% smaller, 0.4\% loss)} \\
\multicolumn{7}{l}{• Tiny: Resource-constrained scenarios (68\% smaller, 1.3\% loss)} \\
\hline
\end{tabular}
\end{table}
```

**说明**:
- FPS (Frames Per Second) = 每秒处理图像数
- Time = 训练时间（10 epochs）
- 清楚展示了模型大小与性能的权衡

---

## 表5 (可选): 损失函数对比

**用途**: 如果要详细展示损失函数实验

```latex
\begin{table}[t]
\centering
\caption{Loss function comparison on KonIQ-10k (Swin-Base)}
\label{tab:loss_function}
\begin{tabular}{lcccc}
\hline
Loss Function & SRCC & PLCC & $\Delta$ SRCC & Ranking \\
\hline
\textbf{L1 (MAE)} & \textbf{0.9375} & \textbf{0.9488} & - & 🥇 1st \\
L2 (MSE) & 0.9373 & 0.9469 & -0.0002 & 🥈 2nd \\
Pairwise Fidelity & 0.9315 & 0.9373 & -0.0060 & 🥉 3rd \\
SRCC Loss & 0.9313 & 0.9416 & -0.0062 & 4th \\
Pairwise Ranking & 0.9292 & 0.9249 & -0.0083 & 5th \\
\hline
\multicolumn{5}{l}{\textit{Note: Simple L1 loss outperforms complex ranking-based losses}} \\
\hline
\end{tabular}
\end{table}
```

---

## 表6 (可选): 学习率敏感度

**用途**: 展示学习率实验结果

```latex
\begin{table}[t]
\centering
\caption{Learning rate sensitivity analysis (Swin-Base)}
\label{tab:lr_sensitivity}
\begin{tabular}{lcccc}
\hline
Learning Rate & SRCC & PLCC & $\Delta$ SRCC & Epochs \\
\hline
5e-6 & 0.9354 & 0.9448 & -0.24\% & 5 \\
3e-6 & 0.9364 & 0.9464 & -0.14\% & 5 \\
1e-6 & 0.9374 & 0.9485 & -0.04\% & 10 \\
\textbf{5e-7} & \textbf{0.9378} & \textbf{0.9485} & \textbf{0.0\%} & \textbf{10} \\
1e-7 & 0.9375 & 0.9488 & -0.03\% & 14 \\
\hline
\multicolumn{5}{l}{\textit{Note: Optimal LR is 200$\times$ lower than ResNet50 (1e-4)}} \\
\multicolumn{5}{l}{\textit{Swin Transformer requires much smaller learning rate}} \\
\hline
\end{tabular}
\end{table}
```

---

## 在LaTeX中使用这些表格

### 1. 添加必要的包

在LaTeX文件的序言部分（`\documentclass` 之后）添加：

```latex
\usepackage{multirow}  % 用于跨行单元格
\usepackage{amssymb}   % 用于 \checkmark 等符号
\usepackage{booktabs}  % 用于更美观的表格线（可选）
```

### 2. 插入表格

在正文中直接复制上面的表格代码即可：

```latex
\section{Experiments}

\subsection{Comparison with State-of-the-Art}

Table \ref{tab:sota_comparison} shows...

% 插入表1
\begin{table*}[t]
...
\end{table*}

\subsection{Ablation Study}

Table \ref{tab:ablation_study} demonstrates...

% 插入表2
\begin{table}[t]
...
\end{table}
```

### 3. 表格位置控制

- `[t]` - 页面顶部（top）
- `[b]` - 页面底部（bottom）
- `[h]` - 当前位置（here）
- `[!]` - 强制放置
- `[t]` 最常用，让LaTeX自动优化位置

---

## 表格设计说明

### 设计原则

1. **清晰度优先**: 使用粗体突出最佳结果
2. **对比性强**: 分组显示（baseline / variants）
3. **信息完整**: 包含方法、年份、backbone、性能指标
4. **统计显著**: 标注提升百分比和排名

### 表格宽度

- **单栏表格**: 用 `\begin{table}[t]`
  - 适合4-6列的表格
  - 表2、表3、表4、表5、表6

- **双栏表格**: 用 `\begin{table*}[t]`
  - 适合较宽的表格（>6列）
  - 表1 (SOTA对比)

### 美化技巧

如果想要更专业的表格，使用 `booktabs` 包：

```latex
\usepackage{booktabs}

\begin{table}[t]
\centering
\caption{...}
\begin{tabular}{lcccc}
\toprule  % 代替 \hline，更粗
Method & SRCC & PLCC \\
\midrule  % 中间分隔线
Ours & 0.9378 & 0.9485 \\
\bottomrule  % 底部线
\end{tabular}
\end{table}
```

---

## 表格与图表的分工

### 表格适合展示:
- ✅ 精确数值对比
- ✅ 多方法/多指标对比
- ✅ 消融实验结果
- ✅ SOTA对比

### 图表适合展示:
- ✅ 趋势变化（训练曲线）
- ✅ 分布情况（注意力权重）
- ✅ 视觉对比（特征图热力图）
- ✅ 关系可视化（散点图）

---

## 快速引用清单

在正文中引用表格：

```latex
As shown in Table \ref{tab:sota_comparison}, our method achieves...

The ablation study (Table \ref{tab:ablation_study}) demonstrates...

Cross-dataset results (Table \ref{tab:cross_dataset}) indicate...

Table \ref{tab:model_size} presents the performance-efficiency trade-off...
```

---

**准备完成！可以直接复制到LaTeX论文中！** 📝✨

**接下来**: 生成图像可视化（注意力热力图、训练曲线等）


