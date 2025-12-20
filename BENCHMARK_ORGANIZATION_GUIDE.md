# Benchmark 模型组织指南

## 2️⃣ 其他模型 Benchmark 的组织方式

### 学术项目中的标准做法

在学术研究中，通常需要与其他 SOTA 模型进行对比。有以下几种组织方式：

---

## 方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| **A. 同仓库独立目录** | 便于管理、统一环境 | 可能代码风格不一致 | 简单调用、轻量对比 ✅ |
| **B. 单独仓库** | 完全独立、清晰 | 环境管理复杂 | 大型独立实现 |
| **C. Git Submodules** | 版本控制清晰 | 学习曲线陡 | 依赖外部仓库 |
| **D. 混合方式** | 灵活 | 需要良好文档 | 复杂项目 |

---

## 🌟 推荐方案：同仓库独立目录（方案 A）

### 为什么推荐这种方式？

1. ✅ **便于对比**：所有模型在同一数据集上测试，环境一致
2. ✅ **易于管理**：统一的依赖、统一的数据路径
3. ✅ **报告友好**：一个仓库包含所有实验，便于写论文
4. ✅ **代码复用**：可以共享 `data_loader.py`、`folders.py` 等工具
5. ✅ **学术规范**：顶会论文常见做法（CVPR/ICCV/NeurIPS）

---

## 📁 推荐的目录结构

```
Perceptual-IQA-CS3324/
├── README.md
├── requirements.txt
│
├── data_loader.py           # 共享的数据加载器
├── folders.py               # 共享的数据集处理
│
├── train_test_IQA.py        # 原始 HyperIQA (ResNet-50)
├── HyerIQASolver.py
├── models.py
│
├── train_swin.py            # 改进的 HyperIQA (Swin Transformer) ⭐ 你的方法
├── HyperIQASolver_swin.py
├── models_swin.py
│
├── benchmarks/              # 其他 SOTA 模型对比 ✨ 新增
│   ├── README.md           # Benchmark 使用说明
│   │
│   ├── maniqa/             # MANIQA 模型
│   │   ├── train_maniqa.py
│   │   ├── test_maniqa.py
│   │   ├── model_maniqa.py
│   │   └── README.md
│   │
│   ├── musiq/              # MUSIQ 模型
│   │   ├── train_musiq.py
│   │   ├── test_musiq.py
│   │   ├── model_musiq.py
│   │   └── README.md
│   │
│   ├── clipiqa/            # CLIP-IQA+ 模型
│   │   ├── test_clipiqa.py
│   │   ├── model_clipiqa.py
│   │   └── README.md
│   │
│   └── results/            # Benchmark 结果汇总
│       ├── benchmark_results.csv
│       └── comparison_plots.py
│
├── checkpoints/            # 模型检查点
├── logs/                   # 训练日志
├── complexity/             # 复杂度分析
├── docs/                   # 文档
│   ├── EXPERIMENT_COMMANDS.md
│   ├── ABLATION_STUDY_CORRECTED.md
│   └── ...
│
└── results/                # 最终结果（论文用）
    ├── main_results.csv
    ├── ablation_results.csv
    ├── benchmark_comparison.csv
    └── figures/
```

---

## 📝 具体实施步骤

### Step 1: 创建 benchmarks 目录

```bash
cd /root/Perceptual-IQA-CS3324
mkdir -p benchmarks/results
```

### Step 2: 创建 Benchmark README

```bash
cat > benchmarks/README.md << 'EOF'
# Benchmark Models

This directory contains implementations and evaluation code for 
state-of-the-art IQA models used for comparison.

## Models Included

1. **MANIQA** (CVPR 2022)
   - Paper: Multi-dimension Attention Network for No-reference Image Quality Assessment
   - Directory: `maniqa/`

2. **MUSIQ** (ICCV 2021)
   - Paper: Multi-scale Image Quality Transformer
   - Directory: `musiq/`

3. **CLIP-IQA+** (arXiv 2023)
   - Paper: Exploring CLIP for Assessing Image Quality
   - Directory: `clipiqa/`

## Usage

See individual model directories for specific usage instructions.

## Results

Benchmark comparison results are available in `results/benchmark_results.csv`.
EOF
```

### Step 3: 为每个 Benchmark 模型创建独立目录

以 MANIQA 为例：

```bash
mkdir -p benchmarks/maniqa
cd benchmarks/maniqa

# 创建测试脚本模板
cat > test_maniqa.py << 'EOF'
"""
Test MANIQA on KonIQ-10k dataset
Using official pretrained weights or train from scratch
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# 可以导入项目根目录的共享模块
import data_loader
import folders

# MANIQA 特定的代码
# ...

if __name__ == '__main__':
    print("Testing MANIQA on KonIQ-10k...")
    # Your code here
EOF

# 创建 README
cat > README.md << 'EOF'
# MANIQA Benchmark

## Setup

```bash
pip install timm scipy
```

## Test on KonIQ-10k

```bash
python test_maniqa.py --dataset koniq-10k --model_path pretrained/maniqa.pth
```

## Results

| Dataset | SRCC | PLCC |
|---------|------|------|
| KonIQ-10k | 0.9XX | 0.9XX |

## Citation

```
@inproceedings{maniqa2022,
  title={Multi-dimension Attention Network for No-reference Image Quality Assessment},
  author={...},
  booktitle={CVPR},
  year={2022}
}
```
EOF
```

---

## 🔧 共享模块的使用

### 方法 1: 直接导入（推荐）

在 benchmark 代码中：

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# 现在可以导入项目根目录的模块
from data_loader import DataLoader
from folders import Koniq_10kFolder
```

### 方法 2: 相对导入

```python
# 在 benchmarks/maniqa/test_maniqa.py 中
from ...data_loader import DataLoader  # 两级向上
```

### 方法 3: 软链接（高级）

```bash
cd benchmarks/maniqa
ln -s ../../data_loader.py .
ln -s ../../folders.py .
```

---

## 📊 Benchmark 结果汇总

### 创建统一的结果文件

```bash
cat > benchmarks/results/benchmark_results.csv << 'EOF'
Model,Backbone,Params(M),FLOPs(G),SRCC,PLCC,Year,Venue
HyperIQA (Original),ResNet-50,48.3,12,0.9009,0.9170,2020,CVPR
HyperIQA (Ours),Swin-Base,88.8,18,0.9336,0.9464,2025,-
MANIQA,ViT-B,TBD,TBD,TBD,TBD,2022,CVPR
MUSIQ,Transformer,TBD,TBD,TBD,TBD,2021,ICCV
CLIP-IQA+,CLIP ViT-L,TBD,TBD,TBD,TBD,2023,arXiv
EOF
```

---

## 🎯 具体模型的获取和集成

### 选项 1: 使用预训练模型（推荐）

```python
# benchmarks/maniqa/test_maniqa.py
import torch
from torchvision import transforms

# 下载预训练模型
model = torch.hub.load('repo', 'model', pretrained=True)

# 或者从本地加载
model.load_state_dict(torch.load('pretrained/maniqa.pth'))

# 测试
model.eval()
# ...
```

### 选项 2: Clone 官方仓库到 external/

```bash
mkdir -p external
cd external

# Clone 官方实现
git clone https://github.com/IIGROUP/MANIQA.git
git clone https://github.com/google/musiq.git

# 在 benchmarks/ 中创建简单的包装脚本
cd ../benchmarks/maniqa
cat > test_maniqa.py << 'EOF'
import sys
sys.path.append('../../external/MANIQA')

from MANIQA.model import MANIQA
# 使用官方代码
EOF
```

### 选项 3: Git Submodules（高级）

```bash
cd external
git submodule add https://github.com/IIGROUP/MANIQA.git
git submodule update --init --recursive
```

---

## 📈 生成对比图表

### 创建可视化脚本

```python
# benchmarks/results/comparison_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取结果
df = pd.read_csv('benchmark_results.csv')

# SRCC 对比柱状图
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['SRCC'])
plt.xlabel('Model')
plt.ylabel('SRCC')
plt.title('Performance Comparison on KonIQ-10k')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('srcc_comparison.png', dpi=300)

# 参数量 vs 性能散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['Params(M)'], df['SRCC'], s=100)
for i, model in enumerate(df['Model']):
    plt.annotate(model, (df['Params(M)'][i], df['SRCC'][i]))
plt.xlabel('Parameters (M)')
plt.ylabel('SRCC')
plt.title('Model Complexity vs Performance')
plt.tight_layout()
plt.savefig('complexity_vs_performance.png', dpi=300)
```

---

## 📝 论文中的引用方式

### 表格示例

```latex
\begin{table}[t]
\centering
\caption{Comparison with State-of-the-Art Methods on KonIQ-10k}
\begin{tabular}{lcccc}
\toprule
Method & Backbone & Params & SRCC & PLCC \\
\midrule
HyperIQA (2020) & ResNet-50 & 48M & 0.906 & 0.917 \\
MUSIQ (2021) & Transformer & 30M & 0.917 & 0.926 \\
MANIQA (2022) & ViT-B & 45M & 0.920 & 0.930 \\
\midrule
\textbf{Ours} & Swin-Base & 89M & \textbf{0.9336} & \textbf{0.9464} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## ⚠️ 注意事项

### 1. 依赖管理

如果不同模型需要不同依赖：

```bash
# 主项目依赖
requirements.txt

# Benchmark 特定依赖
benchmarks/maniqa/requirements.txt
benchmarks/musiq/requirements.txt
```

### 2. 数据集路径

确保所有 benchmark 使用相同的数据集路径：

```python
# 在 benchmarks/ 中创建 config.py
import os

# 统一的数据集路径
KONIQ_PATH = os.path.join(os.path.dirname(__file__), '../koniq-10k')
SPAQ_PATH = os.path.join(os.path.dirname(__file__), '../spaq-test')
```

### 3. 评估协议一致性

- ✅ 使用相同的 train/test split
- ✅ 使用相同的评估指标 (SRCC, PLCC)
- ✅ 使用相同的图像分辨率
- ✅ 记录测试时的参数设置

---

## 🎯 推荐的 Benchmark 模型列表

### IQA 领域 SOTA 模型（2020-2024）

| 模型 | 年份 | 会议 | GitHub | 推荐程度 |
|------|------|------|--------|---------|
| **HyperIQA** | 2020 | CVPR | [link](https://github.com/SSL92/hyperIQA) | ⭐⭐⭐⭐⭐ (你的 baseline) |
| **MUSIQ** | 2021 | ICCV | [link](https://github.com/google-research/musiq) | ⭐⭐⭐⭐⭐ |
| **TReS** | 2022 | WACV | [link](https://github.com/isalirezag/TReS) | ⭐⭐⭐⭐ |
| **MANIQA** | 2022 | CVPR | [link](https://github.com/IIGROUP/MANIQA) | ⭐⭐⭐⭐⭐ |
| **CLIP-IQA+** | 2023 | arXiv | [link](https://github.com/IceClear/CLIP-IQA) | ⭐⭐⭐⭐ |
| **Q-Align** | 2023 | arXiv | [link](https://github.com/Q-Future/Q-Align) | ⭐⭐⭐⭐⭐ (VLM-based) |
| **LIQE** | 2023 | arXiv | [link](https://github.com/zwx8981/LIQE) | ⭐⭐⭐⭐ |

### 选择建议

**必选（至少2-3个）**：
1. **MANIQA** - 2022 CVPR, ViT-based, 性能强
2. **MUSIQ** - 2021 ICCV, Transformer, Google 出品
3. **CLIP-IQA+** - 2023, 基于 CLIP 的方法（VLM）

**可选**：
4. TReS - 如果想对比基于 Transformer 的方法
5. Q-Align - 如果想对比大模型方法（但可能太新）

---

## 🚀 快速开始

### 1. 创建 benchmarks 目录结构

```bash
cd /root/Perceptual-IQA-CS3324
bash << 'EOF'
mkdir -p benchmarks/{maniqa,musiq,clipiqa,results}
touch benchmarks/README.md
touch benchmarks/results/benchmark_results.csv
EOF
```

### 2. 下载预训练模型（示例）

```bash
cd benchmarks
mkdir pretrained
# 下载 MANIQA 预训练权重
wget https://example.com/maniqa_koniq.pth -O pretrained/maniqa_koniq.pth
```

### 3. 测试 benchmark

```bash
cd maniqa
python test_maniqa.py --dataset koniq-10k --model_path ../pretrained/maniqa_koniq.pth
```

---

## 📚 总结

### 推荐做法 ✅

1. **在同一仓库中创建 `benchmarks/` 目录**
2. **每个模型一个子目录**（maniqa/, musiq/, clipiqa/）
3. **共享数据加载和评估代码**
4. **统一的结果记录格式** (CSV)
5. **清晰的 README 和文档**

### 不推荐做法 ❌

1. ❌ 每个模型单独 clone 到不同目录
2. ❌ 没有统一的评估协议
3. ❌ 结果分散在各处难以对比
4. ❌ 依赖冲突导致环境混乱

---

**文档版本**: 1.0  
**最后更新**: December 20, 2025  
**适用场景**: 学术项目、论文对比实验

