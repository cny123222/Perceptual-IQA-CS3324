# 📊 SMART-IQA 论文表格和图表使用指南

**日期**: 2024-12-24  
**状态**: ✅ 所有表格和图表已生成

---

## 📁 文件结构

```
Perceptual-IQA-CS3324/
├── IEEE-conference-template-062824/
│   ├── PAPER_TABLES_FINAL.md          # LaTeX表格代码（6个表格）
│   └── FIGURES_AND_TABLES_GUIDE.md    # 本文件
├── paper_figures/                      # 生成的图表目录
│   ├── cross_dataset_heatmap.pdf/.png  # 图1: 跨数据集热力图
│   ├── sota_radar_chart.pdf/.png       # 图2: SOTA雷达图
│   ├── ablation_waterfall.pdf/.png     # 图3: 消融瀑布图
│   ├── model_size_scatter.pdf/.png     # 图4: 模型大小散点图
│   ├── lr_sensitivity.pdf/.png         # 图5: 学习率敏感度
│   └── contribution_pie.pdf/.png       # 图6: 组件贡献饼图
├── generate_paper_visualizations.py    # 图表生成脚本
└── PAPER_VISUALIZATION_SUGGESTIONS.md  # 更多可视化建议

```

---

## 📊 表格清单（6个表格）

所有表格的LaTeX代码在 `PAPER_TABLES_FINAL.md` 文件中。

### 表1: SOTA对比表 ⭐⭐⭐ 【必须】
- **标签**: `\label{tab:sota_comparison}`
- **用途**: 与其他SOTA方法对比
- **数据**: 9个SOTA方法 + 我们的方法
- **类型**: 双栏宽度 (`\begin{table*}`)
- **位置**: Introduction或Results开头

**引用示例**:
```latex
As shown in Table \ref{tab:sota_comparison}, SMART-IQA achieves 
state-of-the-art performance with SRCC of 0.9378...
```

---

### 表2: 消融实验表 ⭐⭐⭐ 【必须】
- **标签**: `\label{tab:ablation_study}`
- **用途**: 展示每个组件的贡献
- **数据**: C0 (Baseline), A2 (Swin), A1 (Multi-Scale), E6 (Full)
- **类型**: 单栏 (`\begin{table}`)
- **位置**: Ablation Study子章节

**引用示例**:
```latex
The ablation study (Table \ref{tab:ablation_study}) demonstrates 
that Swin Transformer contributes 87\% of the total improvement...
```

---

### 表3: 跨数据集泛化表 ⭐⭐ 【推荐】
- **标签**: `\label{tab:cross_dataset}`
- **用途**: 对比HyperIQA和SMART-IQA的泛化能力
- **数据**: 4个数据集（KonIQ, SPAQ, KADID, AGIQA）
- **类型**: 单栏
- **位置**: Cross-Dataset Generalization子章节

**引用示例**:
```latex
Cross-dataset results (Table \ref{tab:cross_dataset}) show that 
our method maintains strong generalization...
```

---

### 表4: 模型大小对比表 ⭐⭐ 【推荐】
- **标签**: `\label{tab:model_size}`
- **用途**: 展示Tiny/Small/Base三个版本的性能-效率权衡
- **数据**: 3个模型大小 + HyperIQA baseline
- **类型**: 单栏
- **位置**: Model Variants或Experiments章节

**引用示例**:
```latex
Table \ref{tab:model_size} presents the performance-efficiency 
trade-off. The Small variant achieves 0.9338 SRCC with 43\% fewer parameters...
```

---

### 表5: 损失函数对比表 ⭐ 【可选】
- **标签**: `\label{tab:loss_function}`
- **用途**: 对比5种损失函数的效果
- **数据**: L1, L2, Pairwise Fidelity, SRCC Loss, Pairwise Ranking
- **类型**: 单栏
- **位置**: Supplementary Material或Training Details

**引用示例**:
```latex
We compare five loss functions (Table \ref{tab:loss_function}) 
and find that simple L1 loss performs best...
```

---

### 表6: 学习率敏感度表 ⭐ 【可选】
- **标签**: `\label{tab:lr_sensitivity}`
- **用途**: 展示学习率对性能的影响
- **数据**: 5个学习率 (1e-7, 5e-7, 1e-6, 3e-6, 5e-6)
- **类型**: 单栏
- **位置**: Training Details或Supplementary

**引用示例**:
```latex
Learning rate sensitivity analysis (Table \ref{tab:lr_sensitivity}) 
reveals that 5e-7 is optimal, 200× lower than ResNet50...
```

---

## 🖼️ 图表清单（6个图表）

所有图表已生成在 `paper_figures/` 目录中，PDF和PNG格式。

### 图1: 跨数据集性能热力图 ⭐⭐⭐ 【强烈推荐】
- **文件**: `cross_dataset_heatmap.pdf`
- **用途**: 直观展示HyperIQA vs SMART-IQA在4个数据集上的表现
- **特点**: 
  - 颜色编码（绿色=高，红色=低）
  - 箭头标注提升
  - 底部统计信息
- **位置**: Cross-Dataset Generalization章节

**LaTeX代码**:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{paper_figures/cross_dataset_heatmap.pdf}
\caption{Cross-dataset generalization performance. Our method (SMART-IQA) 
consistently outperforms HyperIQA across most datasets, demonstrating 
strong generalization ability.}
\label{fig:cross_dataset}
\end{figure}
```

---

### 图2: SOTA雷达图 ⭐⭐⭐ 【强烈推荐】
- **文件**: `sota_radar_chart.pdf`
- **用途**: 多维度对比我们的方法与SOTA（6个维度）
- **维度**: 
  1. KonIQ-10k SRCC
  2. Cross-domain Average
  3. Parameter Efficiency
  4. Inference Speed
  5. Training Efficiency
  6. Robustness
- **位置**: Results或Comparison章节

**LaTeX代码**:
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.8\textwidth]{paper_figures/sota_radar_chart.pdf}
\caption{Multi-dimensional comparison with state-of-the-art methods. 
SMART-IQA achieves the best balance across accuracy, efficiency, 
and robustness metrics.}
\label{fig:radar}
\end{figure*}
```

---

### 图3: 消融实验瀑布图 ⭐⭐⭐ 【必须】
- **文件**: `ablation_waterfall.pdf`
- **用途**: 展示渐进式消融过程和组件贡献
- **特点**:
  - 瀑布式柱状图
  - 每个柱子标注增量和占比
  - 红色箭头标注总提升
- **位置**: Ablation Study章节

**LaTeX代码**:
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.9\textwidth]{paper_figures/ablation_waterfall.pdf}
\caption{Progressive ablation study showing component contributions. 
Swin Transformer contributes 87\% of the total improvement (+2.68\% SRCC), 
while multi-scale fusion and attention mechanism contribute 5\% and 8\%, respectively.}
\label{fig:ablation}
\end{figure*}
```

---

### 图4: 模型大小散点图 ⭐⭐ 【推荐】
- **文件**: `model_size_scatter.pdf`
- **用途**: 展示参数量与性能的权衡关系
- **特点**:
  - 我们的模型用菱形标记
  - 绿色虚线=效率前沿
  - 高性能区域阴影
- **位置**: Model Variants或Discussion章节

**LaTeX代码**:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{paper_figures/model_size_scatter.pdf}
\caption{Performance-efficiency trade-off across model sizes. 
Our Small variant offers the best balance with 43\% fewer parameters 
and only 0.4\% SRCC loss.}
\label{fig:model_size}
\end{figure}
```

---

### 图5: 学习率敏感度曲线 ⭐⭐ 【推荐】
- **文件**: `lr_sensitivity.pdf`
- **用途**: 展示学习率对性能的影响和收敛速度
- **特点**:
  - 左图：倒U型曲线，最优点用金色星标注
  - 右图：不同学习率的收敛轮数
  - 底部说明：Swin需要200×更小的学习率
- **位置**: Training Details或Experiments章节

**LaTeX代码**:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{paper_figures/lr_sensitivity.pdf}
\caption{Learning rate sensitivity analysis. (Left) SRCC vs learning rate 
shows an inverted-U curve with optimal LR at 5e-7. (Right) Training efficiency 
varies with LR. Swin Transformer requires 200× smaller LR than ResNet50.}
\label{fig:lr_sens}
\end{figure}
```

---

### 图6: 组件贡献饼图 ⭐ 【可选】
- **文件**: `contribution_pie.pdf`
- **用途**: 以饼图形式展示组件贡献占比
- **特点**:
  - Swin Transformer (87%) - 红色，突出显示
  - Attention (8%) - 蓝色
  - Multi-Scale (5%) - 绿色
  - 中心标注总提升
- **位置**: Ablation Study（作为补充）

**LaTeX代码**:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.45\textwidth]{paper_figures/contribution_pie.pdf}
\caption{Component contribution breakdown. Swin Transformer is the dominant 
contributor (87\%), while attention mechanism and multi-scale fusion provide 
additional gains of 8\% and 5\%, respectively.}
\label{fig:pie}
\end{figure}
```

---

## 📝 论文中的推荐布局

### Introduction
- **表1** (SOTA对比) - 展示我们的方法达到SOTA

### Method
- **架构图** (需要手动绘制或使用AI工具)

### Experiments

#### 4.1 Experimental Setup
- 数据集、训练细节等

#### 4.2 Comparison with State-of-the-Art
- **表1** (SOTA对比表)
- **图2** (SOTA雷达图)

#### 4.3 Ablation Study
- **表2** (消融实验表)
- **图3** (消融瀑布图)
- **图6** (组件贡献饼图) - 可选

#### 4.4 Model Variants
- **表4** (模型大小对比)
- **图4** (模型大小散点图)

#### 4.5 Cross-Dataset Generalization
- **表3** (跨数据集表)
- **图1** (跨数据集热力图)

#### 4.6 Training Details (或放在Supplementary)
- **表6** (学习率敏感度表) - 可选
- **图5** (学习率曲线)
- **表5** (损失函数对比) - 可选

---

## ✅ 使用检查清单

### 表格
- [ ] 复制LaTeX代码到`.tex`文件
- [ ] 确认所有`\cite{}`引用在`references.bib`中存在
- [ ] 检查表格标签 (`\label{}`) 是否唯一
- [ ] 在正文中添加引用 (`\ref{}`)
- [ ] 编译检查表格格式

### 图表
- [ ] 将PDF文件复制到论文目录（或保持相对路径）
- [ ] 在LaTeX中插入图表代码
- [ ] 检查图表标签 (`\label{}`) 是否唯一
- [ ] 在正文中添加引用 (`\ref{}`)
- [ ] 检查图表清晰度和大小
- [ ] 编译检查图表显示

---

## 🎨 如果需要修改图表

### 重新生成所有图表
```bash
cd /root/Perceptual-IQA-CS3324
python3 generate_paper_visualizations.py
```

### 修改图表参数
编辑 `generate_paper_visualizations.py`，然后重新运行。

**常见修改**:
- 颜色方案：修改 `colors` 变量
- 字体大小：修改 `fontsize` 参数
- 图表尺寸：修改 `figsize` 参数
- 数据更新：修改数据数组

---

## 🆘 还缺什么？

### 必须要做的：
1. **架构图** (Architecture Diagram)
   - 最重要的图！
   - 参考：`ARCHITECTURE_DIAGRAM_GUIDE.md`
   - 建议使用：Powerpoint、Draw.io、或AI绘图工具

### 强烈推荐的额外可视化：

2. **注意力热力图** (Attention Heatmap) ⭐⭐⭐
   - 展示Channel Attention的动态权重
   - 参考：`PAPER_VISUALIZATION_SUGGESTIONS.md` (1.1节)
   - 需要：运行模型提取attention_weights

3. **定性结果对比** (Visual Comparison Grid)
   - 展示5-10个样本图像
   - 对比GT、Our Pred、HyperIQA Pred
   - 参考：`PAPER_VISUALIZATION_SUGGESTIONS.md` (6.1节)

4. **特征图可视化** (Feature Map Visualization)
   - 展示4个stage的特征激活
   - 参考：`PAPER_VISUALIZATION_SUGGESTIONS.md` (1.2节)

---

## 💡 论文写作建议

### 表格和图表的分工

**表格适合**:
- ✅ 精确数值对比（SOTA对比、消融实验）
- ✅ 多维度指标（SRCC, PLCC, Params, FLOPs）
- ✅ 需要查找具体数字

**图表适合**:
- ✅ 趋势和关系（学习率曲线、散点图）
- ✅ 直观对比（热力图、雷达图）
- ✅ 视觉理解（架构图、注意力可视化）

### 引用技巧

**好的引用**:
```latex
As shown in Figure \ref{fig:ablation}, Swin Transformer contributes 
87\% of the total improvement, demonstrating its critical role in 
performance gains.
```

**避免**:
```latex
Figure 3 shows results.  % 太简单
See Table 2.             % 缺少上下文
```

---

## 📦 最终检查清单

### 提交论文前
- [ ] 所有表格都正确编译
- [ ] 所有图表都显示清晰
- [ ] 所有`\ref{}`都正确链接
- [ ] 所有`\cite{}`都在参考文献中
- [ ] PDF中表格和图表清晰可读
- [ ] 图表说明（caption）完整准确
- [ ] 双盲审稿：去除作者信息

### PDF质量检查
- [ ] 所有图表为矢量格式（PDF优先）
- [ ] 文字清晰（不模糊）
- [ ] 颜色在黑白打印时也可区分
- [ ] 图表大小适中（不过大或过小）

---

## 📞 需要帮助？

**如果需要**:
- 修改表格格式 → 告诉我具体要修改什么
- 修改图表样式 → 告诉我你的需求
- 添加新的图表 → 描述你想要什么样的可视化
- 生成注意力热力图 → 我可以帮你写代码提取attention weights
- 绘制架构图 → 我可以提供详细的绘图指导

---

**🎉 表格和图表已经准备完毕！开始写论文吧！** ✍️

**下一步建议**:
1. 📐 绘制架构图（最重要）
2. ✍️ 开始写各个章节
3. 📊 根据需要插入表格和图表
4. 🔬 如果时间允许，添加注意力可视化

**祝写作顺利！** 🚀


