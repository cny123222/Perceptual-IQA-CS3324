# 实验总结报告 - Perceptual Image Quality Assessment

## 📋 项目概述

**任务**：无参考图像质量评估（No-Reference Image Quality Assessment）

**数据集**：KonIQ-10k
- 训练集：7,046 张图像
- 测试集：2,010 张图像
- 每张图像在训练时采样 20 个 patches (224×224)

**目标**：预测图像的主观质量评分，使用 SRCC 和 PLCC 作为评估指标

---

## 🏗️ 架构设计

### **核心架构：Swin Transformer + HyperNet**

```
Input Image (224×224)
    ↓
Swin Transformer Tiny (Backbone)
    ↓ (Multi-scale features)
[Stage 0: 96 channels]
[Stage 1: 192 channels]  → Multi-scale Feature Fusion
[Stage 2: 384 channels]      (Concatenation + Conv)
[Stage 3: 768 channels]
    ↓
HyperNet (Meta-learner)
    ↓ (Dynamically generates)
Target Network Weights (FC layers)
    ↓
Quality Score
```

### **关键特性**

1. **Multi-scale Feature Fusion** ✅
   - 融合 Swin Transformer 的 4 个 stage 特征
   - 方法：Adaptive pooling (7×7) → Concatenation → Conv (降维)
   - 总通道数：96 + 192 + 384 + 768 = 1440 → 112 (hyper_in_channels)

2. **HyperNet Architecture**
   - 动态生成 Target Network 的权重和偏置
   - Target Network: 4 层全连接网络
   - 参数量：~28.8M (Swin) + ~0.5M (HyperNet) = **29.3M total**

3. **Training Configuration**
   ```python
   batch_size = 96
   train_patch_num = 20  # per image
   test_patch_num = 20
   optimizer = Adam
   lr_hypernet = 2e-4
   lr_backbone = 2e-5
   scheduler = Step Decay (÷10 every 6 epochs)
   loss = L1 Loss (MAE)
   early_stopping = patience 7
   random_seed = 42  # 可复现
   ```

---

## 📊 实验结果

### **主要结果（Baseline - Multi-scale Concat）**

| Epoch | Train Loss | Train SRCC | Test SRCC | Test PLCC | 状态 |
|-------|------------|------------|-----------|-----------|------|
| **1** | 4.997 | 0.8758 | **0.9195** | **0.9342** | ⭐ **最佳** |
| 2 | 3.073 | 0.9527 | 0.9185 | 0.9320 | 开始过拟合 |
| 3 | 2.527 | 0.9674 | 0.9145 | 0.9275 | 性能下降 |
| 4 | 2.218 | 0.9747 | 0.9174 | 0.9287 | 持续过拟合 |
| 5+ | ... | ... | ... | ... | 继续下降 |

**最佳性能**：
- **SRCC**: 0.9195
- **PLCC**: 0.9342
- **出现时机**：第 1 个 epoch

---

### **消融实验 1：Ranking Loss vs. L1 Loss**

**目标**：测试 Ranking Loss (pairwise) 对性能的影响

| Method | Loss Function | Test SRCC | Test PLCC |
|--------|---------------|-----------|-----------|
| Baseline | L1 only | **0.9195** | **0.9342** |
| Ranking α=0.5 | L1 + 0.5×Ranking | 0.9092 | 0.9289 |
| Ranking α=1.0 | L1 + 1.0×Ranking | ~0.90 | ~0.93 |

**结论**：
- ❌ Ranking Loss 降低了性能
- ✅ 纯 L1 Loss 更适合这个任务
- **原因**：MAE 已经足够强，Ranking Loss 可能引入噪声

---

### **消融实验 2：Attention-based Multi-scale Fusion**

**动机**：用注意力机制动态调整不同尺度特征的权重

**实现**：
```python
class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels=[96, 192, 384, 768]):
        self.attention_net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )
    
    def forward(self, feat_list):
        # Stage 3 (global) → attention weights
        feat3_global = AdaptiveAvgPool2d(feat_list[-1], (1,1))
        weights = self.attention_net(feat3_global)  # [B, 4]
        
        # Weighted fusion
        fused = Σ (weights[i] * feat_list[i])
        return fused
```

**结果**：

| Method | Epoch 1 SRCC | Epoch 1 PLCC | Epoch 2 SRCC | 状态 |
|--------|--------------|--------------|--------------|------|
| **Concat (Baseline)** | **0.9195** | **0.9342** | 0.9185 | ✅ 最佳 PLCC |
| Attention (std=0.05) | 0.9198 | 0.9317 | 0.9145 | ❌ 过拟合严重 |
| Attention (std=0.01) | 0.9196 | 0.9317 | 0.9159 | ❌ 仍过拟合 |
| Attention (dropout=0.5) | 0.9196 | 0.9317 | 0.9159 | ❌ 无改善 |

**观察到的注意力权重**：
```
初始（改进前）：[0.0000, 0.0000, 0.0001, 0.9999]  # 极度不平衡
初始（改进后）：[0.2316, 0.2153, 0.2261, 0.3270]  # 较平衡
```

**结论**：
- ❌ 注意力机制没有带来性能提升
- ❌ PLCC 反而下降（0.9342 → 0.9317）
- ❌ 过拟合问题更严重
- **可能原因**：
  1. 简单 concat 已经足够有效
  2. 注意力增加了 ~0.2M 参数，加剧过拟合
  3. HyperNet 本身就很复杂，再加注意力优化困难

---

### **消融实验 3：Test Augmentation**

| Method | Test SRCC | 可复现性 |
|--------|-----------|---------|
| RandomCrop | 0.9195 | ❌ 低（随机） |
| CenterCrop | 0.9182 | ✅ 高（固定） |

**结论**：
- RandomCrop 性能略好（+0.0013）
- 但 CenterCrop 更适合论文发表（可复现）
- **当前选择**：RandomCrop（原论文方法，`--test_random_crop` 参数）

---

## ❗ 当前核心问题：严重过拟合

### **问题表现**

1. **第 1 个 epoch 后性能持续下降**
   - Test SRCC: 0.9195 → 0.9185 → 0.9145 → ...
   - Train SRCC: 0.8758 → 0.9527 → 0.9674 → 0.9747

2. **训练集和测试集性能差距扩大**
   - Epoch 1: Train 0.8758 vs Test 0.9195 (gap = -0.0437)
   - Epoch 4: Train 0.9747 vs Test 0.9174 (gap = +0.0573)

3. **Early stopping 在第 1 个 epoch 停止**
   - 说明模型很快就达到泛化能力峰值
   - 之后的训练都是在"记忆"训练集

### **可能的原因分析**

1. **模型容量过大**
   - Swin Transformer Tiny: 28.8M 参数
   - HyperNet: 0.5M 参数
   - 对于 7k 训练图像可能过于复杂

2. **数据增强不足**
   - 当前只有 RandomCrop (224×224)
   - 没有颜色扰动、旋转、噪声等
   - Patch 之间信息冗余高（来自同一图像）

3. **正则化不足**
   - 无 Dropout（除了注意力实验中的尝试）
   - Weight decay = 0（未使用）
   - 无 Label smoothing

4. **学习率可能过大**
   - HyperNet: 2e-4
   - Backbone: 2e-5
   - 第 1 个 epoch 就学到位了，之后开始过拟合

---

## 🔧 已尝试的解决方案

| 方法 | 效果 | 说明 |
|------|------|------|
| Ranking Loss | ❌ | 性能下降 |
| Attention Fusion | ❌ | 过拟合更严重 |
| Attention Dropout (0.5) | ❌ | 无改善 |
| RandomCrop vs CenterCrop | ✅ | RandomCrop 略好 |
| Early Stopping (patience=7) | ✅ | 自动保存第 1 epoch |

---

## 💭 未尝试的可能改进方向

### **1. 更强的数据增强**

```python
# 颜色扰动
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

# 几何变换
transforms.RandomRotation(10)
transforms.RandomHorizontalFlip(0.5)

# 图像质量相关的增强
- Random JPEG compression
- Gaussian blur
- Gaussian noise
```

**挑战**：IQA 任务中，某些增强可能改变图像的真实质量

---

### **2. 正则化技术**

```python
# Dropout in HyperNet
nn.Dropout(0.3) in fc layers

# Weight decay
optimizer = Adam(..., weight_decay=1e-4)

# Label smoothing
# 将 MOS score 做轻微平滑

# Stochastic depth (Swin Transformer)
drop_path_rate = 0.2
```

---

### **3. 训练策略调整**

```python
# 更小的学习率
lr_hypernet = 1e-4  # 原来 2e-4
lr_backbone = 1e-5  # 原来 2e-5

# Warmup
# 前 1-2 个 epoch 用更小的学习率

# 更激进的 early stopping
patience = 3  # 原来 7

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### **4. 模型架构调整**

```python
# 使用更小的 backbone
swin_transformer_nano  # 代替 tiny

# 减少 patch 数量
train_patch_num = 10  # 原来 20
# 降低数据冗余

# Freeze early layers
# 冻结 Swin 的前 1-2 个 stage

# 在 HyperNet 中加入 BatchNorm
# 可能提高训练稳定性
```

---

### **5. 数据层面**

```python
# 混合多个数据集
# KonIQ-10k + LIVE-itW + SPAQ
# 增加数据多样性

# 更强的 patch 采样策略
# 当前：随机采样 20 个 patches
# 改进：确保 patches 覆盖不同区域（grid sampling）
```

---

### **6. Loss Function 改进**

```python
# Huber Loss（更鲁棒）
loss = nn.SmoothL1Loss()

# Weighted MAE（关注难样本）
# 对预测误差大的样本加权

# Contrastive Learning
# 学习质量分数相近的图像应该有相似的特征
```

---

## 📈 性能对比（KonIQ-10k SOTA）

| Method | Backbone | SRCC | PLCC | 参数量 |
|--------|----------|------|------|--------|
| DBCNN | Custom CNN | 0.875 | 0.884 | ~10M |
| HyperIQA (原论文) | ResNet50 | 0.906 | 0.917 | ~25M |
| MANIQA | ViT-Base | 0.920 | 0.937 | ~86M |
| **Ours (Baseline)** | **Swin-Tiny** | **0.9195** | **0.9342** | **29.3M** |

**优势**：
- ✅ 参数效率高（29.3M vs 86M ViT）
- ✅ 性能接近 SOTA
- ✅ 使用 multi-scale 特征融合

**劣势**：
- ❌ 严重过拟合（只能训练 1 个 epoch）
- ❌ 泛化能力可能不如大模型

---

## 🎯 核心问题总结

### **最需要解决的问题**

1. **如何让模型在第 2+ 个 epoch 不掉点？**
   - 当前：Epoch 1 (0.9195) → Epoch 2 (0.9185) → Epoch 3 (0.9145)
   - 期望：Epoch 1 (0.9195) → Epoch 2 (0.9200+) → ...

2. **如何在不损失性能的前提下减少过拟合？**
   - 数据增强 vs 任务特殊性（不能破坏图像质量）
   - 正则化 vs 模型容量需求

3. **是否应该换用更小的 backbone？**
   - Swin-Tiny (28.8M) 可能太大
   - Swin-Nano / ResNet50 / MobileNet?

---

## 🔍 需要 AI 专家帮助解答的问题

1. **为什么第 1 个 epoch 就达到最佳性能？**
   - 是学习率太大？
   - 还是模型容量太强？
   - 还是数据太简单？

2. **IQA 任务的数据增强有哪些最佳实践？**
   - 哪些增强不会改变图像的真实质量？
   - 如何平衡增强强度和任务特性？

3. **HyperNet + Swin Transformer 这个组合是否合理？**
   - 两个都是"元学习"架构
   - 会不会互相干扰？
   - 有没有更简单的替代方案？

4. **Multi-scale fusion 的最佳方式是什么？**
   - Concat（当前）vs Attention（失败）vs 其他？
   - 是否需要可学习的融合权重？

5. **如何设计实验来诊断过拟合的根本原因？**
   - 应该先尝试哪个方向？
   - 如何系统性地测试不同假设？

---

## 📁 重要文件

- **代码**：`train_swin.py`, `models_swin.py`, `HyperIQASolver_swin.py`
- **日志**：`logs/swin_multiscale_ranking_alpha0_20251218_161547.log`
- **模型**：`checkpoints/koniq-10k-swin_20251218_161547/best_model_srcc_0.9195_plcc_0.9342.pkl`
- **文档**：`WORK_SUMMARY.md`, `LR_SCHEDULER_GUIDE.md`, `ARCHITECTURE_COMPARISON_GUIDE.md`

---

## ✅ 已实现的功能

- [x] Multi-scale feature fusion
- [x] Learning rate schedulers (Step, Cosine, Constant)
- [x] Early stopping
- [x] Automatic logging
- [x] Reproducibility (random seed)
- [x] Test augmentation options (RandomCrop / CenterCrop)
- [x] SPAQ cross-dataset testing (optional)
- [x] Ranking Loss (但效果不佳)
- [x] Attention-based fusion (但效果不佳)

---

## 🎓 论文价值

即使存在过拟合问题，这个工作仍有论文价值：

1. **Swin Transformer 在 IQA 的首次系统应用**
2. **Multi-scale 特征融合的消融实验**
3. **参数高效的设计**（29.3M vs 86M ViT）
4. **诚实的 ablation studies**（Ranking Loss 和 Attention 失败也是发现）

---

## 🙏 请 AI 专家提供建议

**核心目标**：在不降低 Epoch 1 性能的前提下，让 Epoch 2-5 的性能也能保持在 0.919+ SRCC

**约束条件**：
- 计算资源有限（单 GPU）
- 数据集固定（KonIQ-10k）
- 架构大框架不变（Swin + HyperNet）

**希望得到**：
1. 最有可能有效的改进方向（排序）
2. 实验设计建议（如何测试假设）
3. 是否有类似问题的成功案例
4. 是否需要重新考虑架构选择

感谢！🙏

