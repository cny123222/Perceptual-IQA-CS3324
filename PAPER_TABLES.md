# 论文表格 - 直接可用版本 📊

**Purpose**: 提供论文中直接可用的LaTeX表格代码

---

## Table 1: Main Results (KonIQ-10k)

```latex
\begin{table}[h]
\centering
\caption{Performance comparison on KonIQ-10k dataset}
\label{tab:main_results}
\begin{tabular}{lcccc}
\hline
Method & Backbone & SRCC & PLCC & Params \\
\hline
HyperIQA (Original) & ResNet50 & 0.907 & 0.918 & 25M \\
\textbf{Ours (Base)} & Swin-Base & \textbf{0.9378} & \textbf{0.9485} & 88M \\
\textbf{Ours (Small)} & Swin-Small & 0.9338 & 0.9455 & 50M \\
\textbf{Ours (Tiny)} & Swin-Tiny & 0.9249 & 0.9360 & 28M \\
\hline
Improvement & - & \textbf{+3.08\%} & \textbf{+3.3\%} & - \\
\hline
\end{tabular}
\end{table}
```

---

## Table 2: Ablation Study

```latex
\begin{table}[h]
\centering
\caption{Ablation study on KonIQ-10k (Swin-Base model)}
\label{tab:ablation}
\begin{tabular}{lccccc}
\hline
Config & Swin & Multi-Scale & Attention & SRCC & PLCC \\
\hline
Baseline & - & - & - & 0.907 & 0.918 \\
\hline
w/ Swin & \checkmark & - & - & 0.9338 & 0.9438 \\
w/ Swin + MS & \checkmark & \checkmark & - & 0.9353 & 0.9458 \\
\textbf{Full Model} & \checkmark & \checkmark & \checkmark & \textbf{0.9378} & \textbf{0.9485} \\
\hline
\multicolumn{6}{l}{\textit{Component Contributions:}} \\
Swin Transformer & \multicolumn{4}{l}{+0.0268 (+87\% of total improvement)} \\
Multi-Scale Fusion & \multicolumn{4}{l}{+0.0015 (+5\% of total improvement)} \\
Attention Fusion & \multicolumn{4}{l}{+0.0025 (+8\% of total improvement)} \\
\hline
\end{tabular}
\end{table}
```

---

## Table 3: Model Size Comparison

```latex
\begin{table}[h]
\centering
\caption{Performance vs efficiency trade-off across model sizes}
\label{tab:model_size}
\begin{tabular}{lccccc}
\hline
Model & Params & FLOPs & SRCC & PLCC & Time \\
\hline
Swin-Base & 88M & 18.2G & \textbf{0.9378} & \textbf{0.9485} & 1.7h \\
Swin-Small & 50M & $\sim$11G & 0.9338 & 0.9455 & 1.7h \\
Swin-Tiny & 28M & $\sim$5G & 0.9249 & 0.9360 & 1.5h \\
\hline
\multicolumn{6}{l}{\textit{Small: -43\% params, -0.4\% SRCC $\rightarrow$ Best for deployment}} \\
\multicolumn{6}{l}{\textit{Tiny: -68\% params, -1.29\% SRCC $\rightarrow$ Resource-constrained}} \\
\hline
\end{tabular}
\end{table}
```

---

## Table 4: Learning Rate Sensitivity

```latex
\begin{table}[h]
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
\hline
\end{tabular}
\end{table}
```

---

## Table 5: Cross-Dataset Generalization

```latex
\begin{table}[h]
\centering
\caption{Cross-dataset generalization performance}
\label{tab:cross_dataset}
\begin{tabular}{lcccc}
\hline
Dataset & Type & Images & SRCC & PLCC \\
\hline
\textbf{KonIQ-10k} & In-domain & 2,010 & \textbf{0.9378} & \textbf{0.9485} \\
\hline
SPAQ & Cross-domain & 2,224 & 0.8698 & 0.8709 \\
KADID-10K & Cross-domain & 2,000 & 0.5412 & 0.5591 \\
AGIQA-3K & Cross-domain & 2,982 & 0.6484 & 0.6830 \\
\hline
\multicolumn{3}{l}{Average (Cross-domain)} & 0.6865 & 0.7044 \\
\hline
\multicolumn{5}{l}{\textit{SPAQ: smartphone images; KADID: synthetic distortions;}} \\
\multicolumn{5}{l}{\textit{AGIQA: AI-generated images}} \\
\hline
\end{tabular}
\end{table}
```

---

## Table 6: Computational Complexity

```latex
\begin{table}[h]
\centering
\caption{Computational complexity analysis}
\label{tab:complexity}
\begin{tabular}{lcccc}
\hline
Model & Params & FLOPs & Inference & Throughput \\
\hline
ResNet50 (Original) & 25M & 4.0G & $\sim$10ms & $\sim$100 img/s \\
Swin-Base (Ours) & 88M & 18.2G & 45.2ms & 22 img/s \\
\hline
Overhead & 3.5$\times$ & 4.6$\times$ & 4.5$\times$ & 0.22$\times$ \\
\hline
\multicolumn{5}{l}{\textit{Note: +3.08\% SRCC at 4.6$\times$ computational cost}} \\
\hline
\end{tabular}
\end{table}
```

---

## Table 7: Comparison with State-of-the-Art (如果需要)

```latex
\begin{table*}[t]
\centering
\caption{Comparison with state-of-the-art methods on KonIQ-10k}
\label{tab:sota}
\begin{tabular}{lccccc}
\hline
Method & Year & Backbone & SRCC & PLCC & Params \\
\hline
NIMA & 2018 & InceptionNet & 0.558 & 0.590 & - \\
DBCNN & 2020 & ResNet50 & 0.875 & 0.884 & - \\
HyperIQA & 2020 & ResNet50 & 0.906 & 0.917 & 25M \\
MANIQA & 2022 & ViT & 0.920 & 0.930 & - \\
\hline
\textbf{Ours (Swin-Base)} & 2025 & Swin-Base & \textbf{0.9378} & \textbf{0.9485} & 88M \\
\textbf{Ours (Swin-Small)} & 2025 & Swin-Small & 0.9338 & 0.9455 & 50M \\
\hline
\end{tabular}
\end{table*}
```

**Note**: 需要查找并补充SOTA方法的准确数据！

---

## Figure Captions

### Figure 1: Network Architecture
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/architecture.pdf}
\caption{Overview of our improved HyperIQA framework. We replace the ResNet50 backbone with Swin Transformer, introduce multi-scale feature fusion from three hierarchical levels, and employ an attention-based fusion module to dynamically weight features.}
\label{fig:architecture}
\end{figure*}
```

### Figure 2: Ablation Study Visualization
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.48\textwidth]{figures/ablation_chart.pdf}
\caption{Component contribution analysis. Swin Transformer contributes 87\% of the total improvement, while multi-scale fusion and attention mechanism contribute 5\% and 8\%, respectively.}
\label{fig:ablation}
\end{figure}
```

### Figure 3: Learning Rate Sensitivity
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.48\textwidth]{figures/lr_sensitivity.pdf}
\caption{Learning rate sensitivity analysis. The performance shows an inverted-U curve, with the optimal learning rate at 5e-7, which is 200$\times$ lower than the original ResNet50-based model.}
\label{fig:lr_sensitivity}
\end{figure}
```

### Figure 4: Training Curves
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.48\textwidth]{training_curves_best_model.png}
\caption{Training curves for the best model. The model converges smoothly with early stopping at epoch 7, achieving SRCC of 0.9378.}
\label{fig:training_curves}
\end{figure}
```

---

## Key Numbers for Text

### Abstract
- "achieves **SRCC of 0.9378** and **PLCC of 0.9485** on KonIQ-10k"
- "**+3.08\% improvement** over the original HyperIQA"
- "Swin Transformer contributes **87\% of the total improvement**"

### Introduction
- "original HyperIQA achieves SRCC of 0.907"
- "our method improves it to **0.9378 (+3.08\%)**"

### Method
- "Base model: **88M parameters, 18.2G FLOPs**"
- "optimal learning rate: **5e-7** (200× lower than ResNet50)"

### Results
- "Swin Transformer: **+2.68\% SRCC** (87\% contribution)"
- "Attention mechanism: **+0.25\% SRCC** (8\% contribution)"
- "Multi-scale fusion: **+0.15\% SRCC** (5\% contribution)"
- "Small model: only **-0.4\% SRCC** with **43\% fewer parameters**"
- "Cross-dataset: **SPAQ 0.87**, **AGIQA-3K 0.65**"

---

**准备完成！直接复制表格代码到LaTeX即可。** 📝

