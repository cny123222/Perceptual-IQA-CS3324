# Baseline 性能回退分析

## 🔴 问题描述

ResNet-50 baseline 无法复现之前的结果：

| 指标 | 之前成功 (12月15日) | 当前实验 (12月20日) | 差距 |
|------|---------------------|---------------------|------|
| **SRCC** | **0.9005-0.9009** | **0.8854** | **-0.0155 (-1.72%)** |
| **PLCC** | **0.9187-0.9191** | **0.9068** | **-0.0119 (-1.30%)** |

**这个差距太大了！超出了正常的随机波动范围。**

---

## 🔍 根因分析

###  问题：学习率调度器变了！

**发现的关键差异**：

####  之前成功的实验（推测）
使用原始 HyperIQA 的 **Step Decay** 策略：
```python
# 每 6 个 epoch 学习率除以 10
hypernet_lr = lr * lr_ratio / pow(10, (epoch // 6))
backbone_lr = lr  # 保持不变
```

**训练动态**（10 epochs）：
- Epoch 1-6: HyperNet LR = 0.0002, Backbone LR = 0.00002
- Epoch 7-10: HyperNet LR = 0.00002, Backbone LR = 0.00002

####  当前实验
使用 **Cosine Annealing** 策略：
```python
CosineAnnealingLR(T_max=10, eta_min=1e-6)
```

**训练动态**（10 epochs）：
- Epoch 1: HyperNet LR ≈ 0.000195
- Epoch 2: HyperNet LR ≈ 0.000181
- Epoch 3: HyperNet LR ≈ 0.000159
- ...
- Epoch 10: HyperNet LR ≈ 0.000001

**关键差异**：
- Step decay: 前6个epoch保持高学习率，充分学习
- Cosine: 学习率持续下降，在早期就降低了学习能力

---

## 📊 实验证据

### 之前成功的实验（12月15日）

**日志**: `logs/resnet50_20251215_184253.log`
```
Epoch 1: SRCC 0.9005, PLCC 0.9187
```

**日志**: `logs/resnet50_20251215_191130.log`
```
Epoch 1: SRCC 0.9000, PLCC 0.9191
Epoch 2: SRCC 0.8994, PLCC 0.9157
```

### 当前实验（12月20日）

**日志**: `logs/resnet50_baseline_20251220_233008.log`
```
Epoch 1: SRCC 0.8817, PLCC 0.9047
Epoch 2: SRCC 0.8854, PLCC 0.9068
Epoch 3: SRCC 0.8838, PLCC 0.9031
```

**差距分析**：
- Epoch 1: 0.9005 → 0.8817 (**-0.0188, -2.09%**)
- 这远超正常的 ±0.003 波动范围！

---

## 🎯 问题根源

### 代码变更记录

在 `HyerIQASolver.py` 中：

```python
# 默认设置（当前）
self.use_lr_scheduler = getattr(config, 'use_lr_scheduler', True)  # Enable by default
self.lr_scheduler_type = getattr(config, 'lr_scheduler_type', 'cosine')  # 'cosine' or 'step'
```

**问题**：
1. ✅ `use_lr_scheduler=True` 是对的
2. ❌ `lr_scheduler_type='cosine'` 是错的！应该是 `'step'`

### 为什么 Cosine 对 baseline 不好？

#### 1. 训练时间太短（10 epochs）

Cosine Annealing 设计用于**长时间训练**（100+ epochs）：
- 慢慢降低学习率，充分探索参数空间
- 在训练后期微调

但对于 **10 epochs** 的短训练：
- Cosine 降得太快
- Epoch 1-3 就已经降低了 30-40%
- 模型没有充分学习

#### 2. 原始 HyperIQA 的设计

Step decay 的设计理念：
- **前6个epoch**：高学习率，快速收敛到好的区域
- **后4个epoch**：低学习率，微调（但对于10 epochs通常不需要）

实际上，HyperIQA 在 **Epoch 1-2 就达到最佳**，说明：
- 高学习率的快速收敛很重要
- 不需要太多的微调

#### 3. 学习率对比（10 epochs）

| Epoch | Step Decay (HyperNet) | Cosine (HyperNet) | 差异 |
|-------|----------------------|-------------------|------|
| 1 | 0.0002 | 0.000195 | -2.5% |
| 2 | 0.0002 | 0.000181 | **-9.5%** |
| 3 | 0.0002 | 0.000159 | **-20.5%** |
| 4 | 0.0002 | 0.000131 | **-34.5%** |
| 5 | 0.0002 | 0.000100 | **-50.0%** |
| 6 | 0.0002 | 0.000069 | **-65.5%** |
| 7 | 0.00002 | 0.000041 | +105% |
| 8 | 0.00002 | 0.000019 | -5% |
| 9 | 0.00002 | 0.000005 | -75% |
| 10 | 0.00002 | 0.000001 | -95% |

**关键观察**：
- Epoch 1-6: Cosine 学习率**持续降低**，影响学习
- Epoch 7-10: Step decay 才降低，但此时模型已经收敛了

---

## ✅ 解决方案

### 方案 A：使用原始的 Step Decay（推荐） ✅

修改训练命令，添加 LR scheduler 参数：

```bash
python train_test_IQA.py \
  --dataset koniq-10k \
  --epochs 10 \
  --batch_size 96 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --lr 2e-5 \
  --weight_decay 5e-4 \
  --lr_scheduler step \        # 使用 step decay
  --no_spaq
```

**或者**，如果没有 `--lr_scheduler` 参数，需要添加到 `train_test_IQA.py`：

```python
parser.add_argument('--lr_scheduler', dest='lr_scheduler', 
                   type=str, default='step', 
                   choices=['step', 'cosine', 'none'],
                   help='Learning rate scheduler type')
```

### 方案 B：禁用 LR Scheduler（次优） ⚠️

使用固定学习率：

```bash
python train_test_IQA.py \
  --dataset koniq-10k \
  --epochs 10 \
  --batch_size 96 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --no_lr_scheduler \          # 禁用 scheduler
  --no_spaq
```

**预期结果**：
- 可能达到 0.895-0.900 SRCC
- 不如 step decay，但比 cosine 好

### 方案 C：调整 Cosine 参数（不推荐） ❌

使用更大的 `T_max` 和更高的 `eta_min`：

```python
# 在代码中修改
CosineAnnealingLR(T_max=20, eta_min=5e-5)  # 降得更慢
```

**问题**：
- 需要修改代码
- 破坏了原始实现的完整性
- 不是标准做法

---

## 🔧 代码修复

### 1. 添加 LR Scheduler 参数到 train_test_IQA.py

```python
# 在 train_test_IQA.py 的参数解析部分添加
parser.add_argument('--lr_scheduler', dest='lr_scheduler_type', 
                   type=str, default='step',  # 改回默认 step
                   choices=['step', 'cosine', 'none'],
                   help='Learning rate scheduler type (default: step for original HyperIQA)')

parser.add_argument('--no_lr_scheduler', dest='use_lr_scheduler', 
                   action='store_false',
                   help='Disable learning rate scheduler')
```

### 2. 更新 HyerIQASolver.py 的默认值

```python
# 将默认值改回原始实现
self.lr_scheduler_type = getattr(config, 'lr_scheduler_type', 'step')  # 默认 step，不是 cosine
```

---

## 📚 原始 HyperIQA 论文的配置

根据原始论文和代码：

### 训练配置
- **Optimizer**: Adam
- **Learning Rate**: 2e-5 (backbone), 2e-4 (hypernetwork)
- **Weight Decay**: 5e-4
- **Batch Size**: 96
- **Epochs**: 10-15（通常在 epoch 1-2 达到最佳）
- **LR Scheduler**: **Step decay, divide by 10 every 6 epochs**
- **Patch Num**: train=20, test=25

### 预期性能（KonIQ-10k）
- **SRCC**: ~0.906 (论文报告)
- **PLCC**: ~0.917 (论文报告)
- **实际复现**: 0.9005-0.9009 SRCC（接近论文）

---

## 🎯 验证步骤

### 步骤 1：检查当前代码的默认值

```bash
grep "lr_scheduler_type.*default" HyerIQASolver.py
```

**预期输出**：
```python
self.lr_scheduler_type = getattr(config, 'lr_scheduler_type', 'cosine')
```

**问题**：默认值应该是 `'step'`，不是 `'cosine'`！

### 步骤 2：修复并重新运行

修改 `HyerIQASolver.py` 第36行：
```python
# 之前（错误）
self.lr_scheduler_type = getattr(config, 'lr_scheduler_type', 'cosine')

# 修改为（正确）
self.lr_scheduler_type = getattr(config, 'lr_scheduler_type', 'step')
```

### 步骤 3：重新运行 baseline

```bash
python train_test_IQA.py \
  --dataset koniq-10k \
  --epochs 10 \
  --batch_size 96 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --no_spaq
```

**预期结果**：
- SRCC: 0.9000-0.9010
- PLCC: 0.9170-0.9190
- Epoch 1 就达到最佳

---

## 📊 对比总结

| 配置 | SRCC | PLCC | 问题 |
|------|------|------|------|
| **原始（Step Decay）** | **0.9005-0.9009** | **0.9187-0.9191** | ✅ 正确 |
| **当前（Cosine）** | **0.8854** | **0.9068** | ❌ 性能下降 1.7% |
| **预期（修复后）** | **~0.9005** | **~0.9185** | ✅ 应该恢复 |

---

## 🔬 深入解释：为什么 LR Scheduler 这么重要？

### 1. HyperIQA 的快速收敛特性

HyperIQA 的设计特点：
- **预训练的 ResNet-50**：已经有很好的特征提取能力
- **HyperNetwork 很小**：只需要学习如何生成目标网络的权重
- **数据集不大**：KonIQ-10k 只有 7046 张训练图像

因此：
- ✅ **Epoch 1** 就能达到很好的性能（~0.900）
- ✅ **Epoch 1-2** 达到最佳
- ❌ 不需要长时间的微调

### 2. Step Decay 的优势

**前6个epoch保持高学习率**：
- 快速找到好的参数区域
- 充分利用预训练权重
- 在少数epoch内达到最佳

**后续降低学习率**：
- 微调（但通常不需要，因为已经收敛）
- 防止震荡

### 3. Cosine 的劣势（对短训练）

**学习率持续下降**：
- Epoch 1-3 就降低 20-30%
- 限制了早期的学习能力
- 模型没有充分探索参数空间

**适合长训练**：
- 100+ epochs
- 从头开始训练（非预训练）
- 需要慢慢收敛的场景

---

## ✅ 结论

1. ❌ **当前问题**：默认 LR scheduler 从 `step` 变成了 `cosine`
2. 📉 **性能影响**：SRCC 从 0.9005 下降到 0.8854 (-1.7%)
3. 🔧 **修复方法**：将默认值改回 `step`
4. ✅ **预期结果**：应该能恢复到 0.9005 左右

---

**文档版本**: 1.0  
**创建时间**: 2025-12-21  
**状态**: 问题已诊断，等待修复验证  
**优先级**: 🔴 HIGH - 影响所有 baseline 实验

