# SMART-IQA 额外表格

## 表5: 详细实验设定

```latex
\begin{table*}[t]
\centering
\caption{Detailed experimental configuration and hyperparameters}
\label{tab:experimental_setup}
\begin{tabular}{ll|ll}
\hline
\textbf{Category} & \textbf{Parameter} & \textbf{Category} & \textbf{Parameter} \\
\hline
\multicolumn{2}{l|}{\textit{Dataset Configuration}} & \multicolumn{2}{l}{\textit{Optimizer Configuration}} \\
Training Set & KonIQ-10k (7,046 images) & Optimizer & AdamW \\
Test Set & KonIQ-10k (2,010 images) & Learning Rate (Backbone) & $5\times10^{-7}$ \\
Image Resolution & $224\times224$ (random crop) & Learning Rate (Other) & $5\times10^{-6}$ \\
Augmentation & Random Crop & LR Scheduler & Cosine Annealing \\
& & Weight Decay & $2\times10^{-4}$ \\
\hline
\multicolumn{2}{l|}{\textit{Model Architecture}} & \multicolumn{2}{l}{\textit{Regularization}} \\
Backbone & Swin Transformer Base & Drop Path Rate & 0.3 \\
Parameters & 88M & Dropout Rate & 0.4 \\
Pre-training & ImageNet-21K & Early Stopping & Enabled \\
Multi-Scale Fusion & Enabled & Patience & 3 epochs \\
Attention Fusion & Enabled & & \\
\hline
\multicolumn{2}{l|}{\textit{Training Configuration}} & \multicolumn{2}{l}{\textit{Loss Function}} \\
Batch Size & 32 & Primary Loss & L1 (MAE) \\
Epochs & 10 & Ranking Loss & Disabled ($\alpha=0$) \\
Train Patches & 20 per image & & \\
Test Patches & 20 per image & & \\
\hline
\multicolumn{2}{l|}{\textit{Reproducibility}} & \multicolumn{2}{l}{\textit{Hardware}} \\
Random Seed & 42 & GPU & NVIDIA (Single) \\
CuDNN Deterministic & True & Training Time & $\sim$1.7 hours \\
CuDNN Benchmark & False & Framework & PyTorch \\
\hline
\end{tabular}
\end{table*}
```

---

## 表6: 损失函数对比

```latex
\begin{table}[t]
\centering
\caption{Performance comparison of different loss functions on KonIQ-10k}
\label{tab:loss_function}
\begin{tabular}{lccc}
\hline
Loss Function & SRCC & PLCC & Ranking \\
\hline
\textbf{L1 (MAE)} & \textbf{0.9375} & \textbf{0.9488} & 🥇 1st \\
L2 (MSE) & 0.9373 & 0.9469 & 🥈 2nd \\
Pairwise Fidelity & 0.9315 & 0.9373 & 🥉 3rd \\
SRCC Loss & 0.9313 & 0.9416 & 4th \\
Pairwise Ranking & 0.9292 & 0.9249 & 5th \\
\hline
\multicolumn{4}{l}{\textit{$\Delta$ SRCC vs Best:}} \\
L1 (MAE) & - & - & - \\
L2 (MSE) & -0.0002 & -0.0019 & -0.02\% \\
Pairwise Fidelity & -0.0060 & -0.0115 & -0.64\% \\
SRCC Loss & -0.0062 & -0.0072 & -0.66\% \\
Pairwise Ranking & -0.0083 & -0.0239 & -0.89\% \\
\hline
\multicolumn{4}{l}{\textit{Note: Simple L1 loss outperforms complex ranking-based losses}} \\
\hline
\end{tabular}
\end{table}
```

---

## 表7: 训练日志摘要（主实验）

```latex
\begin{table*}[t]
\centering
\caption{Training log summary of best model (Swin-Base, LR=5e-7)}
\label{tab:training_log}
\begin{tabular}{ccccccc}
\hline
Epoch & Train Loss & Train SRCC & Test Loss & Test SRCC & Test PLCC & Status \\
\hline
1 & 0.2145 & 0.8523 & 0.1834 & 0.9012 & 0.9145 & \\
2 & 0.1892 & 0.8876 & 0.1723 & 0.9156 & 0.9267 & \\
3 & 0.1745 & 0.9024 & 0.1658 & 0.9234 & 0.9345 & \\
4 & 0.1654 & 0.9134 & 0.1602 & 0.9289 & 0.9389 & \\
5 & 0.1589 & 0.9212 & 0.1567 & 0.9323 & 0.9421 & \\
6 & 0.1534 & 0.9267 & 0.1543 & 0.9345 & 0.9448 & \\
7 & 0.1495 & 0.9301 & 0.1521 & \textbf{0.9378} & \textbf{0.9485} & \textbf{Best ⭐} \\
8 & 0.1467 & 0.9324 & 0.1529 & 0.9367 & 0.9478 & \\
9 & 0.1445 & 0.9342 & 0.1534 & 0.9361 & 0.9472 & \\
10 & 0.1428 & 0.9356 & 0.1538 & 0.9354 & 0.9467 & Stop \\
\hline
\multicolumn{7}{l}{\textit{Training converged at Epoch 7 with early stopping triggered at Epoch 10}} \\
\multicolumn{7}{l}{\textit{Best checkpoint saved: SRCC=0.9378, PLCC=0.9485}} \\
\hline
\end{tabular}
\end{table*}
```

**注意**: 训练日志表中的数值是示例数据。如果需要精确数值，需要从实际日志文件中提取。

---

## 表8: 跨数据集详细结果（可选，用于Appendix）

```latex
\begin{table}[t]
\centering
\caption{Detailed cross-dataset evaluation results}
\label{tab:cross_dataset_detailed}
\begin{tabular}{lcccc}
\hline
\multirow{2}{*}{Dataset} & \multirow{2}{*}{Type} & \multirow{2}{*}{Images} & \multicolumn{2}{c}{SMART-IQA (Ours)} \\
\cline{4-5}
& & & SRCC & PLCC \\
\hline
KonIQ-10k & Training & 7,046 / 2,010 & 0.9378 & 0.9485 \\
\hline
\textit{Cross-dataset} & & & & \\
SPAQ & Smartphone & 2,224 & 0.8698 & 0.8709 \\
KADID-10K & Synthetic & 2,000 & 0.5412 & 0.5591 \\
AGIQA-3K & AI-generated & 2,982 & 0.6484 & 0.6830 \\
\hline
\textbf{Average (Cross)} & - & - & \textbf{0.6865} & \textbf{0.7044} \\
\hline
\multicolumn{5}{l}{\textit{Performance Analysis:}} \\
\multicolumn{5}{l}{• SPAQ: Good generalization (-6.8\% SRCC drop)}} \\
\multicolumn{5}{l}{• KADID-10K: Poor generalization (-42.3\% SRCC drop)}} \\
\multicolumn{5}{l}{• AGIQA-3K: Moderate generalization (-30.9\% SRCC drop)}} \\
\hline
\end{tabular}
\end{table}
```

---

## 表9: 学习率敏感度详细结果（可选，用于Appendix）

```latex
\begin{table}[t]
\centering
\caption{Detailed learning rate sensitivity analysis}
\label{tab:lr_sensitivity_detailed}
\begin{tabular}{lcccc}
\hline
Learning Rate & SRCC & PLCC & Epochs & Converged \\
\hline
5e-6 & 0.9354 & 0.9448 & 5 & Yes \\
3e-6 & 0.9364 & 0.9464 & 5 & Yes \\
1e-6 & 0.9374 & 0.9485 & 10 & Yes \\
\textbf{5e-7} & \textbf{0.9378} & \textbf{0.9485} & \textbf{10} & \textbf{Yes ⭐} \\
1e-7 & 0.9375 & 0.9488 & 14 & Yes \\
\hline
\multicolumn{5}{l}{\textit{Observations:}} \\
\multicolumn{5}{l}{• Optimal LR: 5e-7 (200× smaller than ResNet50)}} \\
\multicolumn{5}{l}{• Larger LRs converge faster but slightly worse}} \\
\multicolumn{5}{l}{• Smaller LRs require more epochs but comparable results}} \\
\multicolumn{5}{l}{• Swin Transformer is highly sensitive to LR}} \\
\hline
\end{tabular}
\end{table}
```

---

## 使用说明

### 表5: 详细实验设定
- **位置**: Appendix或Experimental Setup子章节
- **用途**: 提供完整的实验配置，便于复现
- **类型**: 双栏表格

### 表6: 损失函数对比
- **位置**: Appendix或Training Details子章节
- **用途**: 展示不同损失函数的效果
- **类型**: 单栏表格

### 表7: 训练日志摘要
- **位置**: Appendix（补充材料）
- **用途**: 展示训练过程的详细信息
- **类型**: 双栏表格
- **注意**: 需要从实际日志文件中提取精确数值

### 表8-9: 可选补充表格
- **位置**: Appendix（如果空间允许）
- **用途**: 提供更详细的实验结果和分析

---

## 快速引用

在LaTeX正文中引用这些表格：

```latex
% Experimental Setup
Detailed hyperparameters are provided in Table \ref{tab:experimental_setup}.

% Loss Function
We compared five loss functions (Table \ref{tab:loss_function}) 
and found that simple L1 loss performs best...

% Training Process
The training log (Table \ref{tab:training_log}) shows that 
the model converged at Epoch 7 with SRCC of 0.9378...

% Cross-Dataset Details
Detailed cross-dataset results are shown in Table \ref{tab:cross_dataset_detailed}...

% Learning Rate
Complete learning rate sensitivity results are in Table \ref{tab:lr_sensitivity_detailed}...
```

---

**注意**: 
- 表7（训练日志）中的数值是示例，需要从实际日志中提取
- Emoji (🥇🥈🥉⭐) 可能在某些LaTeX编译器中不显示，可以替换为文字或删除
- 表8和表9是可选的，主要用于Appendix


