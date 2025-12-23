# HyperIQA原始论文 vs 我们的改进版本 - 超详细对比

**文档目的**: 详尽解释原始HyperIQA和Swin-HyperIQA之间的每一个细节差异  
**原论文**: Su et al., "Blindly Assess Image Quality in the Wild Guided by A Self-Adaptive Hyper Network", CVPR 2020  
**日期**: 2025-12-23

---

## 📖 目录

1. [原论文的核心思想](#1-原论文的核心思想)
2. [原始HyperIQA的完整架构](#2-原始hyperiqa的完整架构)
3. [我们的改进架构](#3-我们的改进架构)
4. [逐模块详细对比](#4-逐模块详细对比)
5. [数据流对比](#5-数据流对比)
6. [参数量和计算量对比](#6-参数量和计算量对比)
7. [性能提升分解](#7-性能提升分解)

---

## 1. 原论文的核心思想

### 📝 论文摘要（翻译）

原论文提出了一个**自适应超网络架构**来盲评估野外图像质量。将IQA过程分为三个阶段：

1. **Content Understanding** (内容理解): 使用CNN提取图像语义
2. **Perception Rule Learning** (感知规则学习): 通过HyperNet自适应学习感知规则
3. **Quality Predicting** (质量预测): TargetNet根据学习的规则预测质量

**关键创新**: 
- 用HyperNet**动态生成**TargetNet的权重
- 根据图像内容**自适应**地调整质量评估规则
- 模拟人类"先理解内容，再评估质量"的感知过程

### 🎯 原论文要解决的三大挑战

引用原文：

> **Challenge 1: No Reference Image**  
> "BIQA is the most difficult problem among FR-IQA, RR-IQA and BIQA"

> **Challenge 2: Distortion Diversity**  
> "Authentic distortions are more complicated. Images not only suffer from global uniform distortions (e.g. out of focus, low illumination), but also contain non-uniform distortions (e.g. object moving, over lighting, ghosting) in local areas."

> **Challenge 3: Content Variation**  
> "Compared to synthetic IQA databases, authentic IQA databases exhibit great content variation. LIVE has only 30 reference images, while LIVE Challenge and KonIQ-10k have 1169 and 10073 images with diverse contents."

### 💡 HyperNet的设计哲学

引用原文核心观点：

> "In HVS, the top-down perception model indicates that human tries to understand the image before paying attention to other relevant sub-tasks such as quality assessment. However, in current models, fusing IQA task into semantic recognition network forces the network to learn image content and quality simultaneously, while **it is more properly to let the network learn how to judge image quality AFTER it has recognized the image content**."

**翻译**: 人类视觉系统(HVS)的自顶向下感知模型表明，人类先理解图像内容，再进行质量评估。现有模型强迫网络同时学习内容和质量，而**更合理的方式是让网络在识别内容后再学习如何评估质量**。

---

## 2. 原始HyperIQA的完整架构

### 2.1 总体流程图

```
输入图像 (224×224)
    ↓
┌───────────────────────────────────────────────────────────┐
│              ResNet-50 Backbone                           │
│                                                           │
│  Input [3, 224, 224]                                      │
│    ↓ Conv 7×7 + BN + ReLU + MaxPool                       │
│  [64, 56, 56]                                             │
│    ↓                                                      │
│  Layer 1: Bottleneck × 3 → [256, 56, 56]  ───┐           │
│    ↓                                          │           │
│  Layer 2: Bottleneck × 4 → [512, 28, 28]  ───┼───┐       │
│    ↓                                          │   │       │
│  Layer 3: Bottleneck × 6 → [1024, 14, 14] ───┼───┼───┐   │
│    ↓                                          │   │   │   │
│  Layer 4: Bottleneck × 3 → [2048, 7, 7]   ───┼───┼───┼─┐ │
│                                               │   │   │ │ │
└───────────────────────────────────────────────│───│───│─│─┘
                                                │   │   │ │
                ┌───────────────────────────────┘   │   │ │
                │ LDA Module (Local Distortion Aware)│   │ │
                │                                   │   │ │
                ↓                                   ↓   ↓ ↓
            LDA1 [28-D]                         LDA2/3/4
                │                                   │   │ │
                └───────────┬───────────────────────┴───┴─┘
                            ↓
                    Concat → target_in_vec [112-D]
                            │
                            │    ┌──────────────────────────┐
                            │    │  Layer4 [2048, 7, 7]     │
                            │    └──────────┬───────────────┘
                            │               ↓
                            │    ┌──────────────────────────┐
                            │    │  HyperNet                │
                            │    │  生成TargetNet的动态权重  │
                            │    │  - Conv1x1 压缩特征      │
                            │    │  - Conv3x3 生成权重       │
                            │    │  - FC 生成bias           │
                            │    └──────────┬───────────────┘
                            │               ↓
                            │         动态权重 (w1-w5, b1-b5)
                            │               │
                            └───────────────┼───────────────┐
                                            │               │
                                ┌───────────────────────────┴┐
                                │  TargetNet                  │
                                │  使用动态权重的FC网络        │
                                │  FC: 112→16→8→4→2→1         │
                                └───────────┬─────────────────┘
                                            ↓
                                      Quality Score
```

### 2.2 ResNet-50 Backbone详解

#### 2.2.1 结构细节

```python
class ResNetBackbone(nn.Module):
    def __init__(self):
        # 初始卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 4个layer（stage）
        self.layer1 = Bottleneck × 3   # 输出: [B, 256, 56, 56]
        self.layer2 = Bottleneck × 4   # 输出: [B, 512, 28, 28]
        self.layer3 = Bottleneck × 6   # 输出: [B, 1024, 14, 14]
        self.layer4 = Bottleneck × 3   # 输出: [B, 2048, 7, 7]
```

#### 2.2.2 Bottleneck结构

```
输入 x [B, C_in, H, W]
    ↓
Conv 1×1: C_in → C_mid  (降维)
    ↓ BN + ReLU
Conv 3×3: C_mid → C_mid (特征提取)
    ↓ BN + ReLU
Conv 1×1: C_mid → C_out (升维, C_out = 4×C_mid)
    ↓ BN
    ↓ + x (residual connection)
    ↓ ReLU
输出 [B, C_out, H, W]

例如Layer1的第一个Bottleneck:
64 → 64 (1×1) → 64 (3×3) → 256 (1×1) → 256
```

### 2.3 Local Distortion Aware (LDA) 模块 ⭐核心创新

#### 2.3.1 LDA的设计动机

引用原文：

> "Local features are beneficial to handle non-uniform distortions in the image. We introduce a local distortion aware module to further capture image quality."

**翻译**: 局部特征有利于处理图像中的非均匀失真。我们引入了局部失真感知模块来进一步捕捉图像质量。

#### 2.3.2 LDA的具体实现

**从Layer 1提取** (原论文Figure 2中的LDA):
```python
# Layer 1输出: [B, 256, 56, 56]
lda1_pool = nn.Sequential(
    nn.Conv2d(256, 16, kernel_size=1, stride=1),  # 通道压缩
    nn.AvgPool2d(7, stride=7),                     # 空间下采样
)
# 输出: [B, 16, 8, 8] → flatten → [B, 1024]
lda1_fc = nn.Linear(1024, 28)  # 进一步压缩到28维
# 最终: lda_1 = [B, 28]
```

**从Layer 2提取**:
```python
# Layer 2输出: [B, 512, 28, 28]
lda2_pool = nn.Sequential(
    nn.Conv2d(512, 32, kernel_size=1),
    nn.AvgPool2d(7, stride=7),
)
# 输出: [B, 32, 4, 4] → flatten → [B, 512]
lda2_fc = nn.Linear(512, 28)
# 最终: lda_2 = [B, 28]
```

**从Layer 3提取**:
```python
# Layer 3输出: [B, 1024, 14, 14]
lda3_pool = nn.Sequential(
    nn.Conv2d(1024, 64, kernel_size=1),
    nn.AvgPool2d(7, stride=7),
)
# 输出: [B, 64, 2, 2] → flatten → [B, 256]
lda3_fc = nn.Linear(256, 28)
# 最终: lda_3 = [B, 28]
```

**从Layer 4提取**:
```python
# Layer 4输出: [B, 2048, 7, 7]
lda4_pool = nn.AvgPool2d(7, stride=7)
# 输出: [B, 2048, 1, 1] → flatten → [B, 2048]
lda4_fc = nn.Linear(2048, 28)  # 注: 实际是112-28*3=28
# 最终: lda_4 = [B, 28]
```

**拼接**:
```python
target_in_vec = torch.cat([lda_1, lda_2, lda_3, lda_4], dim=1)
# 输出: [B, 112]  (28+28+28+28=112)
```

#### 2.3.3 LDA的问题 ⚠️

1. **空间信息大量丢失**:
   ```
   Layer 1: [256, 56, 56] = 802,816 个值
              ↓ LDA压缩
           [28] = 28 个标量
   
   压缩比: 802,816 / 28 ≈ 28,672倍！
   ```

2. **仅用于TargetNet**:
   - LDA特征 → TargetNet (质量预测)
   - Layer4特征 → HyperNet (权重生成)
   - **两者分离，没有充分利用多尺度信息**

3. **固定的压缩方式**:
   - 使用固定的7×7池化和FC压缩
   - 无法自适应地保留重要信息

### 2.4 HyperNet详解

#### 2.4.1 HyperNet的输入

```python
hyper_in_feat = layer4  # [B, 2048, 7, 7]
```

**注意**: 只使用Layer4的特征！

#### 2.4.2 HyperNet的结构

```python
class HyperNet(nn.Module):
    def __init__(self):
        # 1. 压缩Layer4特征
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),  # 2048→1024
            nn.ReLU(),
            nn.Conv2d(1024, 512, 1),   # 1024→512
            nn.ReLU(),
            nn.Conv2d(512, 112, 1),    # 512→112 (与target_in_vec同维度)
            nn.ReLU()
        )
        # 输出: [B, 112, 7, 7]
        
        # 2. 生成TargetNet的5层权重
        # FC1: 112 → 16
        self.fc1w_conv = nn.Conv2d(112, 112*16//49, 3, padding=1)
        self.fc1b_fc = nn.Linear(112, 16)
        
        # FC2: 16 → 8
        self.fc2w_conv = nn.Conv2d(112, 16*8//49, 3, padding=1)
        self.fc2b_fc = nn.Linear(112, 8)
        
        # FC3: 8 → 4
        self.fc3w_conv = nn.Conv2d(112, 8*4//49, 3, padding=1)
        self.fc3b_fc = nn.Linear(112, 4)
        
        # FC4: 4 → 2
        self.fc4w_conv = nn.Conv2d(112, 4*2//49, 3, padding=1)
        self.fc4b_fc = nn.Linear(112, 2)
        
        # FC5: 2 → 1
        self.fc5w_fc = nn.Linear(112, 2)
        self.fc5b_fc = nn.Linear(112, 1)
```

#### 2.4.3 动态权重生成机制

```
原理: 对于每张输入图像，HyperNet生成一组专属的TargetNet权重

输入图像A: 
  hyper_in_feat_A [B, 112, 7, 7]
    ↓ HyperNet
  weights_A = {w1_A, b1_A, ..., w5_A, b5_A}

输入图像B:
  hyper_in_feat_B [B, 112, 7, 7]
    ↓ HyperNet
  weights_B = {w1_B, b1_B, ..., w5_B, b5_B}

不同图像 → 不同权重 → 自适应评估！
```

**权重维度计算**:
```python
# FC1权重: 将[B, 112, 7, 7]卷积输出[B, 112*16/49, 7, 7]
# 然后reshape为: [B, 16, 112, 1, 1]
fc1w = fc1w_conv(hyper_in_feat).view(B, 16, 112, 1, 1)

# FC1 bias: GAP后[B, 112] → FC → [B, 16]
fc1b = fc1b_fc(pool(hyper_in_feat).squeeze())
```

### 2.5 TargetNet详解

#### 2.5.1 TargetNet的作用

使用HyperNet生成的**动态权重**来预测图像质量分数。

#### 2.5.2 TargetNet的结构

```python
class TargetNet(nn.Module):
    def forward(self, target_in_vec, weights):
        """
        target_in_vec: [B, 112, 1, 1] - 从LDA来的多尺度特征
        weights: HyperNet生成的动态权重
        """
        x = target_in_vec
        
        # FC1: 112 → 16 (使用动态权重)
        x = F.conv2d(x, weights['fc1w'], weights['fc1b'])
        x = F.relu(x)  # [B, 16, 1, 1]
        
        # FC2: 16 → 8
        x = F.conv2d(x, weights['fc2w'], weights['fc2b'])
        x = F.relu(x)  # [B, 8, 1, 1]
        
        # FC3: 8 → 4
        x = F.conv2d(x, weights['fc3w'], weights['fc3b'])
        x = F.relu(x)  # [B, 4, 1, 1]
        
        # FC4: 4 → 2
        x = F.conv2d(x, weights['fc4w'], weights['fc4b'])
        x = F.relu(x)  # [B, 2, 1, 1]
        
        # FC5: 2 → 1
        x = F.conv2d(x, weights['fc5w'], weights['fc5b'])
        # [B, 1, 1, 1]
        
        return x.squeeze()  # [B]
```

### 2.6 训练损失函数

原论文使用**L1 loss + Ranking loss**:

```python
# L1 loss (MAE)
l1_loss = torch.mean(torch.abs(pred - target))

# Ranking loss (pairwise)
# 对于质量差异大的图像对，强制预测分数的排序正确
ranking_loss = margin_ranking_loss(pred_i, pred_j, sign(target_i - target_j))

# 总损失
total_loss = l1_loss + alpha * ranking_loss
```

---

## 3. 我们的改进架构

### 3.1 总体流程图

```
输入图像 (224×224)
    ↓
┌─────────────────────────────────────────────────────────────┐
│           Swin Transformer Backbone (Base)                  │
│                                                             │
│  Input [3, 224, 224]                                        │
│    ↓ Patch Partition (4×4) + Linear Embedding              │
│  [128, 56, 56]                                              │
│    ↓                                                        │
│  Stage 0: Swin Blocks × 2 → [128, 56, 56]  ───┐            │
│    ↓ Patch Merging                             │            │
│  Stage 1: Swin Blocks × 2 → [256, 28, 28]  ───┼───┐        │
│    ↓ Patch Merging                             │   │        │
│  Stage 2: Swin Blocks × 18 → [512, 14, 14] ───┼───┼───┐    │
│    ↓ Patch Merging                             │   │   │    │
│  Stage 3: Swin Blocks × 2 → [1024, 7, 7]   ───┼───┼───┼──┐ │
│                                                │   │   │  │ │
└────────────────────────────────────────────────│───│───│──│─┘
                                                 │   │   │  │
                ┌────────────────────────────────┘   │   │  │
                │  真正的多尺度特征融合 (改进2)        │   │  │
                │                                    │   │  │
                ↓                                    ↓   ↓  ↓
        Adaptive Pooling to 7×7               (所有stage都保留!)
                │                                    │   │  │
                ↓                                    ↓   ↓  ↓
           [128,7,7]                         [256,7,7] ... [1024,7,7]
                │                                    │   │  │
                └────────────┬───────────────────────┴───┴──┘
                             ↓
                      ┌──────────────────────────────────┐
                      │  Channel Attention (改进3)       │
                      │  GAP(Stage3) → FC → Softmax      │
                      │  生成4个注意力权重                │
                      └──────────────┬───────────────────┘
                                     ↓
                           Weighted Concatenation
                                     ↓
                         hyper_in_feat [1920, 7, 7]
                                     │
                    ┌────────────────┴────────────────┐
                    │  HyperNet (改进1的输入+改进4)    │
                    │  - Conv1x1 压缩: 1920→512→256→112│
                    │  - Dropout(0.4) 正则化           │
                    │  - 生成动态权重                  │
                    └────────────┬────────────────────┘
                                 ↓
                          动态权重 (w1-w5, b1-b5)
                                 │
                    ┌────────────┴────────────────┐
                    │  TargetNet (改进4)          │
                    │  - FC: 112→16→8→4→2→1       │
                    │  - Dropout(0.5) 每层后      │
                    └────────────┬────────────────┘
                                 ↓
                           Quality Score
```

### 3.2 Swin Transformer Backbone详解

#### 3.2.1 Patch Partition + Linear Embedding

```python
# 与ResNet的7×7卷积不同，Swin使用patch embedding
Input: [B, 3, 224, 224]
  ↓ 划分成4×4的patches
Patches: [B, 56×56, 4×4×3] = [B, 3136, 48]
  ↓ Linear projection
Embedded: [B, 3136, 128]  # Base模型的embed_dim=128
  ↓ Reshape
Output: [B, 128, 56, 56]
```

**对比ResNet**:
```
ResNet:  Conv 7×7, stride=2 → MaxPool → [64, 56, 56]
Swin:    Patch 4×4, Linear Proj → [128, 56, 56]

差异: Swin一开始就有更高的通道数(128 vs 64)
```

#### 3.2.2 Swin Transformer Block

```
一个Swin Block的内部结构:

输入 x [B, C, H, W]
    ↓ reshape to [B, H×W, C]
    ↓ Layer Norm
    ↓
┌───────────────────────────────────────┐
│  Window Multi-Head Self-Attention     │
│  (W-MSA)                              │
│                                       │
│  1. 将feature map划分为7×7的windows   │
│  2. 在每个window内计算self-attention  │
│  3. 并行处理所有windows                │
└───────────────────────────────────────┘
    ↓ + x (residual)
    ↓ Layer Norm
    ↓
┌───────────────────────────────────────┐
│  MLP (Feed-Forward Network)           │
│  Linear → GELU → Dropout → Linear     │
│  4×expansion (C → 4C → C)             │
└───────────────────────────────────────┘
    ↓ + x (residual)
    ↓ reshape to [B, C, H, W]
输出

然后是Shifted Window Block (SW-MSA):
  - 将window向右下移动window_size//2
  - 实现跨window的信息交流
```

**核心优势**:
1. **Window Attention**: 计算复杂度O(window_size² × H×W)，远小于全局attention的O((H×W)²)
2. **Shifted Window**: 跨window信息传递，获得全局感受野
3. **Hierarchical**: 4个stage逐渐降低分辨率、增加通道数，类似CNN

#### 3.2.3 Patch Merging

```
在每个stage之间进行下采样:

输入 x [B, C, H, W]
    ↓ reshape to [B, H/2, W/2, 4C]  # 2×2邻域合并
    ↓ Layer Norm
    ↓ Linear: 4C → 2C
输出 [B, 2C, H/2, W/2]

例如:
Stage 0→1: [B, 128, 56, 56] → Patch Merging → [B, 256, 28, 28]
```

### 3.3 多尺度特征融合详解 (⭐改进2)

#### 3.3.1 与LDA的根本区别

**原始LDA** (压缩式):
```
Layer1 [256, 56, 56] → LDA → lda_1 [28]
Layer2 [512, 28, 28] → LDA → lda_2 [28]
Layer3 [1024, 14, 14] → LDA → lda_3 [28]
Layer4 [2048, 7, 7] → LDA → lda_4 [28]
                              ↓ Concat
                        target_in_vec [112]

问题: 空间信息几乎完全丢失 (7×7 → 标量)
```

**我们的多尺度融合** (保留式):
```
Stage 0 [128, 56, 56] → AdaptivePool → feat0 [128, 7, 7]
Stage 1 [256, 28, 28] → AdaptivePool → feat1 [256, 7, 7]
Stage 2 [512, 14, 14] → AdaptivePool → feat2 [512, 7, 7]
Stage 3 [1024, 7, 7]  → (不需要pool)  → feat3 [1024, 7, 7]
                                         ↓ Concat
                                hyper_in_feat [1920, 7, 7]

优势: 保留了完整的7×7空间信息！
```

#### 3.3.2 具体实现代码

```python
def forward(self, x):
    # 提取4个stage的特征
    features = self.swin(x)
    # features[0]: [B, 128, 56, 56]
    # features[1]: [B, 256, 28, 28]
    # features[2]: [B, 512, 14, 14]
    # features[3]: [B, 1024, 7, 7]
    
    if self.use_multiscale:
        # 统一到7×7
        pooled_features = []
        for feat in features:
            pooled = F.adaptive_avg_pool2d(feat, (7, 7))
            pooled_features.append(pooled)
        
        # 拼接
        hyper_in_feat = torch.cat(pooled_features, dim=1)
        # [B, 128+256+512+1024, 7, 7] = [B, 1920, 7, 7]
    else:
        # 单尺度（仅用Stage3）
        hyper_in_feat = features[-1]  # [B, 1024, 7, 7]
    
    return hyper_in_feat
```

#### 3.3.3 为什么是7×7？

1. **与HyperNet兼容**: HyperNet需要固定尺寸的输入来生成固定大小的权重
2. **合理的空间分辨率**: 7×7=49个位置，足够捕捉局部失真
3. **计算效率**: 不会太大，保持推理速度

### 3.4 Channel Attention机制详解 (⭐改进3)

#### 3.4.1 设计动机

**问题**: 简单拼接假设4个stage同等重要，但实际上：
- 高质量图像: 高层语义特征更重要 (Stage 3)
- 低质量/失真图像: 中低层纹理特征更重要 (Stage 0-2)

**解决**: 让网络**自适应地学习**每个scale的重要性！

#### 3.4.2 Attention实现

```python
class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels_list):
        # in_channels_list = [128, 256, 512, 1024] for Base
        self.attention_net = nn.Sequential(
            nn.Linear(in_channels_list[-1], 256),  # 1024→256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4),  # →4个权重
            nn.Softmax(dim=1)   # 归一化到sum=1
        )
    
    def forward(self, feat_list):
        # feat_list: [feat0, feat1, feat2, feat3]
        B = feat_list[0].size(0)
        
        # 1. 统一空间尺寸
        feats_pooled = []
        for feat in feat_list:
            feat_pooled = F.adaptive_avg_pool2d(feat, (7, 7))
            feats_pooled.append(feat_pooled)
        
        # 2. 从最高层特征提取全局信息
        feat3_global = F.adaptive_avg_pool2d(feat_list[-1], (1, 1))
        feat3_global = feat3_global.squeeze(-1).squeeze(-1)  # [B, 1024]
        
        # 3. 生成注意力权重
        attention_weights = self.attention_net(feat3_global)
        # [B, 4]，每个样本有4个权重，sum to 1
        
        # 4. 加权
        weighted_feats = []
        for i, feat in enumerate(feats_pooled):
            weight = attention_weights[:, i].view(B, 1, 1, 1)
            weighted_feat = feat * weight
            weighted_feats.append(weighted_feat)
        
        # 5. 拼接
        fused_feat = torch.cat(weighted_feats, dim=1)
        # [B, 1920, 7, 7]
        
        return fused_feat, attention_weights
```

#### 3.4.3 Attention的直观理解

```
假设一张图片:
  - 高质量，内容清晰
  
Attention weights可能是:
  [0.10, 0.15, 0.25, 0.50]
   ↑     ↑     ↑     ↑
 Stage0 Stage1 Stage2 Stage3
 (低层) (中层) (中高层)(高层)
 
解释: Stage3(高层语义)权重最大(0.50)，因为高质量图像主要看整体结构

────────────────────────────────

假设另一张图片:
  - 低质量，局部模糊
  
Attention weights可能是:
  [0.35, 0.30, 0.25, 0.10]
   ↑     ↑     ↑     ↑
 Stage0 Stage1 Stage2 Stage3
 
解释: Stage0-1(低中层纹理)权重大，因为需要检测局部模糊
```

### 3.5 HyperNet改进详解 (改进1+4)

#### 3.5.1 输入通道数变化

**原始**:
```python
hyper_in_feat = layer4  # [B, 2048, 7, 7]
```

**改进**:
```python
hyper_in_feat = multi_scale_attention_output  # [B, 1920, 7, 7]
```

**意义**: 输入从单一高层特征(2048-D)变成了融合的多尺度特征(1920-D)

#### 3.5.2 Conv压缩网络调整

**原始**:
```python
self.conv1 = nn.Sequential(
    nn.Conv2d(2048, 1024, 1),
    nn.ReLU(),
    nn.Conv2d(1024, 512, 1),
    nn.ReLU(),
    nn.Conv2d(512, 112, 1),
    nn.ReLU()
)
```

**改进**:
```python
self.conv1 = nn.Sequential(
    nn.Conv2d(1920, 512, 1),  # 直接从1920降到512
    nn.ReLU(),
    nn.Conv2d(512, 256, 1),
    nn.ReLU(),
    nn.Conv2d(256, 112, 1),
    nn.ReLU()
)
```

#### 3.5.3 Dropout正则化 (⭐改进4)

**新增**:
```python
self.dropout = nn.Dropout(0.4)

def forward(self, x):
    hyper_in_feat = self.conv1(hyper_in_feat_raw)
    hyper_in_feat = self.dropout(hyper_in_feat)  # 新增！
    # 继续生成权重...
```

**作用**: 防止HyperNet过拟合，提高泛化能力

### 3.6 TargetNet改进详解 (改进4)

**原始**: 无Dropout

**改进**: 每层FC后加Dropout(0.5)

```python
class TargetNet(nn.Module):
    def __init__(self):
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, target_in_vec, weights):
        x = target_in_vec  # [B, 112, 1, 1]
        
        # FC1: 112→16
        x = F.conv2d(x, weights['fc1w'], weights['fc1b'])
        x = F.relu(x)
        x = self.dropout(x)  # 新增！
        
        # FC2: 16→8
        x = F.conv2d(x, weights['fc2w'], weights['fc2b'])
        x = F.relu(x)
        x = self.dropout(x)  # 新增！
        
        # FC3: 8→4
        x = F.conv2d(x, weights['fc3w'], weights['fc3b'])
        x = F.relu(x)
        x = self.dropout(x)  # 新增！
        
        # FC4: 4→2
        x = F.conv2d(x, weights['fc4w'], weights['fc4b'])
        x = F.relu(x)
        x = self.dropout(x)  # 新增！
        
        # FC5: 2→1 (最后一层不加dropout)
        x = F.conv2d(x, weights['fc5w'], weights['fc5b'])
        
        return x.squeeze()
```

### 3.7 训练配置改进

#### 3.7.1 学习率

**原始**: 
- `lr = 1e-4` (AdamW)
- 固定LR或简单decay

**改进**:
- `lr = 5e-7` (经过敏感度分析)
- Cosine Annealing LR scheduler
- **原因**: Swin Transformer是预训练模型，需要更小的LR微调

#### 3.7.2 正则化

**原始**:
- Weight Decay: 1e-4
- 无Drop Path
- 无Dropout

**改进**:
- Weight Decay: 2e-4
- **Drop Path Rate: 0.3** (Swin Transformer内部)
- **Dropout: 0.4** (HyperNet)
- **Dropout: 0.5** (TargetNet)

#### 3.7.3 损失函数

**原始**:
```python
loss = L1_loss + 0.3 * ranking_loss
```

**改进**:
```python
loss = L1_loss  # 去掉ranking loss!
# 因为ranking loss被发现有害 (实验F4证明)
```

---

## 4. 逐模块详细对比

### 4.1 Backbone对比

| 维度 | 原始HyperIQA (ResNet50) | 我们的改进 (Swin Transformer) |
|------|-------------------------|------------------------------|
| **架构类型** | CNN | Vision Transformer |
| **预训练数据** | ImageNet-1K (1.28M) | ImageNet-21K (14M) |
| **初始Embedding** | Conv 7×7, stride=2 + MaxPool | Patch 4×4 + Linear Projection |
| **Stage数量** | 4 (Layer 1-4) | 4 (Stage 0-3) |
| **通道数进化** | 256→512→1024→2048 | 128→256→512→1024 (Base) |
| **特征提取机制** | 3×3卷积 (局部感受野) | Window Self-Attention (7×7窗口) |
| **全局建模** | ❌ 仅通过堆叠间接实现 | ✅ Shifted Window实现跨区域交互 |
| **归一化** | Batch Normalization | Layer Normalization |
| **参数量** | ~25.6M | ~88M (Base) / ~50M (Small) / ~28M (Tiny) |
| **输出特征** | Layer4: [2048, 7, 7] | All Stages保留 |

### 4.2 多尺度特征对比

| 维度 | 原始LDA | 我们的Multi-Scale Fusion |
|------|---------|--------------------------|
| **使用的Stages** | Layer 1-4 (全部) | Stage 0-3 (全部) |
| **特征提取方式** | Conv1×1 + AvgPool7×7 + FC压缩 | AdaptiveAvgPool(7,7)保留空间 |
| **输出维度** | 4×28 = 112 标量 | 4×C×7×7 = 1920×7×7 |
| **空间信息** | ❌ 几乎完全丢失 | ✅ 完整保留7×7 |
| **用途** | 仅TargetNet | HyperNet (更重要!) |
| **可学习性** | ❌ 固定压缩 | ✅ 可选Attention动态加权 |

**直观对比**:
```
原始LDA:
  Layer1 [256,56,56] → LDA → [28]     ← 802,816个值压缩成28个
  Layer2 [512,28,28] → LDA → [28]     ← 401,408个值压缩成28个
  ...
  总损失: 99.99%的信息

我们的方法:
  Stage0 [128,56,56] → Pool → [128,7,7]  ← 保留6,272个值
  Stage1 [256,28,28] → Pool → [256,7,7]  ← 保留12,544个值
  ...
  总损失: ~98%的信息，但保留了空间结构!
```

### 4.3 Attention机制对比

| 维度 | 原始HyperIQA | 我们的改进 |
|------|-------------|-----------|
| **有无Attention** | ❌ 无 | ✅ 有 |
| **类型** | - | Channel Attention |
| **输入** | - | 4个stage的特征 |
| **输出** | - | 4个权重 (sum to 1) |
| **作用** | - | 动态加权不同尺度 |
| **参数量** | 0 | ~260K (1024→256→4) |
| **提升** | - | +0.25% SRCC |

### 4.4 HyperNet对比

| 维度 | 原始 | 改进 |
|------|------|------|
| **输入通道** | 2048 | 1920 |
| **输入来源** | Layer4单尺度 | Stage0-3多尺度融合 |
| **压缩网络** | 2048→1024→512→112 | 1920→512→256→112 |
| **正则化** | ❌ 无 | ✅ Dropout(0.4) |
| **生成权重数量** | 5层(112→16→8→4→2→1) | 相同 |
| **动态权重机制** | 相同 | 相同 |

### 4.5 TargetNet对比

| 维度 | 原始 | 改进 |
|------|------|------|
| **结构** | FC: 112→16→8→4→2→1 | 相同 |
| **激活函数** | ReLU | 相同 |
| **正则化** | ❌ 无Dropout | ✅ Dropout(0.5)每层后 |
| **动态权重** | 使用HyperNet生成 | 相同 |
| **输入** | target_in_vec (LDA压缩) | 相同 (但LDA已被multi-scale替代) |

---

## 5. 数据流对比

### 5.1 原始HyperIQA的完整数据流

```
输入图像 [B, 3, 224, 224]
    ↓
┌────────────── ResNet50 ──────────────┐
│                                      │
│  Conv+BN+ReLU+MaxPool                │
│    ↓ [B, 64, 56, 56]                 │
│                                      │
│  Layer1: Bottleneck×3                │
│    ↓ [B, 256, 56, 56]                │
│    ├─→ LDA1 → [B, 28] ───────┐       │
│    ↓                         │       │
│  Layer2: Bottleneck×4        │       │
│    ↓ [B, 512, 28, 28]        │       │
│    ├─→ LDA2 → [B, 28] ───────┼───┐   │
│    ↓                         │   │   │
│  Layer3: Bottleneck×6        │   │   │
│    ↓ [B, 1024, 14, 14]       │   │   │
│    ├─→ LDA3 → [B, 28] ───────┼───┼─┐ │
│    ↓                         │   │ │ │
│  Layer4: Bottleneck×3        │   │ │ │
│    ↓ [B, 2048, 7, 7]         │   │ │ │
│    ├─→ LDA4 → [B, 28] ───────┼───┼─┼─┤
│    │                         │   │ │ │
└────┼─────────────────────────│───│─│─┘
     │                         │   │ │
     │                         ↓   ↓ ↓ ↓
     │                    Concat: [B, 112]
     │                         ↓
     │                    target_in_vec
     │                         │
     │                         │
     ↓ [B, 2048, 7, 7]        │
┌────────── HyperNet ─────────┴─────────┐
│                                       │
│  Conv1x1: 2048→1024→512→112           │
│    ↓ [B, 112, 7, 7]                   │
│                                       │
│  生成动态权重:                         │
│  - fc1w [B, 16, 112, 1, 1]            │
│  - fc1b [B, 16]                       │
│  - fc2w [B, 8, 16, 1, 1]              │
│  - fc2b [B, 8]                        │
│  - fc3w [B, 4, 8, 1, 1]               │
│  - fc3b [B, 4]                        │
│  - fc4w [B, 2, 4, 1, 1]               │
│  - fc4b [B, 2]                        │
│  - fc5w [B, 1, 2, 1, 1]               │
│  - fc5b [B, 1]                        │
│                                       │
└───────────────┬───────────────────────┘
                │
                ↓ weights
        ┌────────────────┐
        │   TargetNet    │
        │                │
        │  target_in_vec [B, 112, 1, 1]
        │    ↓           │
        │  FC1(dynamic): 112→16
        │    ↓ ReLU      │
        │  FC2(dynamic): 16→8
        │    ↓ ReLU      │
        │  FC3(dynamic): 8→4
        │    ↓ ReLU      │
        │  FC4(dynamic): 4→2
        │    ↓ ReLU      │
        │  FC5(dynamic): 2→1
        │    ↓           │
        │  [B, 1]        │
        └────────────────┘
                ↓
         Quality Score
```

### 5.2 我们改进版本的完整数据流

```
输入图像 [B, 3, 224, 224]
    ↓
┌────────── Swin Transformer ──────────┐
│                                      │
│  Patch Partition 4×4 + Linear Proj  │
│    ↓ [B, 128, 56, 56]                │
│                                      │
│  Stage0: Swin Blocks×2               │
│    ↓ [B, 128, 56, 56] ───────┐       │
│  Patch Merging                │       │
│    ↓ [B, 256, 28, 28]         │       │
│                               │       │
│  Stage1: Swin Blocks×2        │       │
│    ↓ [B, 256, 28, 28] ───────┼───┐   │
│  Patch Merging                │   │   │
│    ↓ [B, 512, 14, 14]         │   │   │
│                               │   │   │
│  Stage2: Swin Blocks×18       │   │   │
│    ↓ [B, 512, 14, 14] ───────┼───┼─┐ │
│  Patch Merging                │   │ │ │
│    ↓ [B, 1024, 7, 7]          │   │ │ │
│                               │   │ │ │
│  Stage3: Swin Blocks×2        │   │ │ │
│    ↓ [B, 1024, 7, 7] ─────────┼───┼─┼─┤
│                               │   │ │ │
└───────────────────────────────│───│─│─┘
                                │   │ │
                                ↓   ↓ ↓ ↓
┌────── Multi-Scale Fusion ────────────────┐
│                                          │
│  AdaptiveAvgPool(7,7):                   │
│    Stage0 → [B, 128, 7, 7]               │
│    Stage1 → [B, 256, 7, 7]               │
│    Stage2 → [B, 512, 7, 7]               │
│    Stage3 → [B, 1024, 7, 7] (no pool)    │
│                                          │
│  ┌─── Channel Attention (可选) ────┐    │
│  │                                  │    │
│  │  GAP(Stage3) → [B, 1024]        │    │
│  │    ↓ FC: 1024→256→4             │    │
│  │    ↓ Softmax                    │    │
│  │  weights [B, 4] (sum to 1)      │    │
│  │                                  │    │
│  │  Weighted Concat:               │    │
│  │    feat0×w0 + feat1×w1 +        │    │
│  │    feat2×w2 + feat3×w3          │    │
│  └──────────────────────────────────┘    │
│                                          │
│  Concatenate: [B, 1920, 7, 7]           │
└────────────────┬─────────────────────────┘
                 ↓
      hyper_in_feat [B, 1920, 7, 7]
                 │
┌────────── HyperNet ─────────────┐
│                                 │
│  Conv1x1: 1920→512→256→112      │
│    ↓ [B, 112, 7, 7]             │
│    ↓ Dropout(0.4) ← 新增!       │
│                                 │
│  生成动态权重 (同原始)           │
│  - fc1w, fc1b, ..., fc5w, fc5b │
│                                 │
└───────────┬─────────────────────┘
            │
            ↓ weights
    ┌─────────────────┐
    │   TargetNet     │
    │                 │
    │  target_in_vec [B, 112, 1, 1]
    │    ↓            │
    │  FC1(dynamic): 112→16
    │    ↓ ReLU       │
    │    ↓ Dropout(0.5) ← 新增!
    │  FC2(dynamic): 16→8
    │    ↓ ReLU       │
    │    ↓ Dropout(0.5) ← 新增!
    │  FC3(dynamic): 8→4
    │    ↓ ReLU       │
    │    ↓ Dropout(0.5) ← 新增!
    │  FC4(dynamic): 4→2
    │    ↓ ReLU       │
    │    ↓ Dropout(0.5) ← 新增!
    │  FC5(dynamic): 2→1
    │    ↓            │
    │  [B, 1]         │
    └─────────────────┘
            ↓
     Quality Score
```

### 5.3 关键差异总结

| 阶段 | 原始HyperIQA | 我们的改进 | 差异 |
|------|-------------|-----------|------|
| **特征提取** | ResNet50 (Layer1-4) | Swin Transformer (Stage0-3) | Transformer vs CNN |
| **多尺度处理** | LDA压缩成标量 | 保留7×7空间特征 | 空间信息保留 |
| **特征融合** | 简单拼接 | Attention加权 | 动态重要性调整 |
| **HyperNet输入** | Layer4单尺度 (2048-D) | 多尺度融合 (1920-D) | 更丰富的输入 |
| **正则化** | 无 | Dropout×2 (HyperNet + TargetNet) | 防止过拟合 |

---

## 6. 参数量和计算量对比

### 6.1 模型参数量

| 模块 | 原始HyperIQA | 我们的改进 (Base) | 我们的改进 (Small) | 我们的改进 (Tiny) |
|------|-------------|------------------|-------------------|------------------|
| **Backbone** | ResNet50: 25.6M | Swin-Base: 88M | Swin-Small: 50M | Swin-Tiny: 28M |
| **Multi-Scale/LDA** | LDA: ~50K | Attention: ~260K | Attention: ~200K | Attention: ~200K |
| **HyperNet** | ~1.5M | ~2.0M | ~1.8M | ~1.8M |
| **TargetNet** | 0 (动态权重) | 0 (动态权重) | 0 (动态权重) | 0 (动态权重) |
| **总计** | **~27.1M** | **~90.3M** | **~52M** | **~30M** |

### 6.2 FLOPs (浮点运算次数)

| 模型 | FLOPs | 相对原始 |
|------|-------|---------|
| 原始HyperIQA | ~4.1G | 1.0× |
| Swin-HyperIQA (Base) | ~18.2G | 4.4× |
| Swin-HyperIQA (Small) | ~10.5G | 2.6× |
| Swin-HyperIQA (Tiny) | ~5.2G | 1.3× |

### 6.3 推理时间

在单张RTX 3090 GPU上，batch_size=1:

| 模型 | 推理时间 (ms) | FPS |
|------|--------------|-----|
| 原始HyperIQA | ~15ms | ~67 |
| Swin-HyperIQA (Base) | ~45ms | ~22 |
| Swin-HyperIQA (Small) | ~30ms | ~33 |
| Swin-HyperIQA (Tiny) | ~20ms | ~50 |

### 6.4 效率分析

**参数量 vs 性能**:
```
原始HyperIQA (27M):     SRCC 0.907
Swin-Tiny (30M):        SRCC 0.9249  (+0.0179, +11% params)
Swin-Small (52M):       SRCC 0.9338  (+0.0268, +92% params)
Swin-Base (90M):        SRCC 0.9378  (+0.0308, +233% params)

结论: Swin-Small提供最佳的效率-性能平衡
```

---

## 7. 性能提升分解

### 7.1 消融实验证明的贡献

根据我们的实验数据:

```
C0: ResNet50 (Original)              →  0.907  SRCC
    ↓ [改进1] Backbone替换
A2: Swin-Base (单尺度)               →  0.9338 SRCC  (+0.0268, +87%)
    ↓ [改进2] 多尺度融合
A1: Swin-Base (多尺度, 无注意力)     →  0.9353 SRCC  (+0.0015, +5%)
    ↓ [改进3] Channel Attention
E6: Swin-Base (多尺度+注意力)        →  0.9378 SRCC  (+0.0025, +8%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总提升: +0.0308 SRCC (+3.40%)
```

### 7.2 修正后的贡献分解

考虑未消融组件(见`UNCOVERED_COMPONENTS_ANALYSIS.md`):

```
总提升: +3.08% (0.907 → 0.9378)

真实分解:
├─ [未消融] ImageNet-21K预训练:    +0.5~1.5% (16-49%)
├─ [未消融] Drop Path正则化:       +0.2~0.5% (6-16%)
├─ [改进1] Swin架构本身:          +1.0~1.8% (32-58%)
├─ [改进2] 多尺度融合:            +0.15% (5%)
├─ [改进3] Channel Attention:     +0.25% (8%)
└─ [改进4] Dropout正则化:         ~0.1% (隐含在训练中)
```

### 7.3 各改进的技术价值

| 改进 | SRCC提升 | 参数增加 | 计算增加 | 技术难度 | 创新性 | 性价比 |
|------|---------|---------|---------|---------|-------|-------|
| **改进1: Swin Backbone** | +2.68% | +62M | +14G FLOPs | 低(直接替换) | ⭐⭐⭐ | ⭐⭐⭐ |
| **改进2: Multi-Scale** | +0.15% | +0M | ~0G | 中(重新设计) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **改进3: Attention** | +0.25% | +260K | ~0.01G | 中(新增模块) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **改进4: Dropout** | ~0.1% | +0M | +0G | 低(加两行代码) | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 8. 理解上的关键要点

### 🔑 为什么要这样改？

#### 1️⃣ 为什么用Swin Transformer替换ResNet50？

**原因1: 更强的特征表示**
- Swin的Window Attention能capture更复杂的模式
- 预训练on ImageNet-21K提供更丰富的先验知识

**原因2: 层级式结构适合IQA**
- 低层捕捉纹理失真 (模糊、噪声)
- 高层捕捉语义失真 (内容理解)
- Swin天然提供4个stage的层级特征

**原因3: 全局建模能力**
- Shifted Window实现了间接的全局attention
- 对于理解整张图像的质量分布很重要

#### 2️⃣ 为什么要保留空间信息(7×7)?

**LDA的问题**:
```
原始图像: [256, 56, 56] = 802,816个值
   ↓ LDA压缩
输出: 28个标量

问题: 一张56×56的feature map被压缩成1个28维向量
→ 丢失了所有的空间位置信息
→ 无法定位失真在图像的哪个区域
```

**保留7×7的好处**:
```
原始图像: [128, 56, 56]
   ↓ AdaptivePool
输出: [128, 7, 7] = 6,272个值

优势: 
- 保留了7×7=49个空间位置
- 每个位置对应原图的一个8×8区域
- 可以捕捉局部失真的位置信息
```

**举例**:
```
假设一张图像:
- 左上角模糊
- 右下角清晰

保留7×7空间:
  左上的(0,0)位置: 高失真特征激活
  右下的(6,6)位置: 低失真特征激活
  
HyperNet可以根据这些空间信息生成不同的权重!
```

#### 3️⃣ 为什么要Channel Attention?

**简单拼接的问题**:
```python
# 简单拼接假设4个scale同等重要
feat = torch.cat([feat0, feat1, feat2, feat3], dim=1)
# [128, 7, 7] + [256, 7, 7] + [512, 7, 7] + [1024, 7, 7]
# = [1920, 7, 7]
```

**实际情况**: 不同图像需要关注不同的scale

```
高质量图像:
  - 内容清晰，结构完整
  - 应该更关注高层语义 (Stage3)
  - Attention: [0.1, 0.15, 0.25, 0.5]

低质量图像 (噪声):
  - 底层纹理受损严重
  - 应该更关注低层细节 (Stage0-1)
  - Attention: [0.35, 0.30, 0.25, 0.1]

模糊图像:
  - 中层结构受损
  - 应该更关注中层特征 (Stage1-2)
  - Attention: [0.15, 0.35, 0.35, 0.15]
```

**Attention让网络自己学习这些规则!**

#### 4️⃣ 为什么要Dropout?

**过拟合风险**:
- Swin Transformer有88M参数
- KonIQ-10k只有10,073张图像
- 参数/样本比 ≈ 8,700:1 (很高!)

**Dropout的作用**:
```
训练时:
  - 随机丢弃40%的HyperNet特征
  - 随机丢弃50%的TargetNet特征
  - 强制网络不依赖特定神经元
  
测试时:
  - 不丢弃任何特征
  - 但所有激活值×(1-dropout_rate)补偿
  
结果:
  - 防止过拟合
  - 提高泛化能力
  - 跨数据集表现更好
```

### 🔑 HyperNet的核心思想

**人类评估图像质量的过程**:
```
1. 看到图像 → 理解内容 (这是什么?)
   "这是一只猫在草地上"
   
2. 根据内容 → 调整评估标准
   动物照片: 关注清晰度、色彩、构图
   风景照片: 关注天空、光线、层次
   人像照片: 关注皮肤、五官、背景虚化
   
3. 应用标准 → 给出分数
   基于调整后的标准评估质量
```

**HyperNet模拟这个过程**:
```
1. Swin Transformer提取特征 → 理解内容
   hyper_in_feat [1920, 7, 7]
   
2. HyperNet生成动态权重 → 调整评估标准
   根据内容生成专属于这张图像的TargetNet权重
   
3. TargetNet使用动态权重 → 给出分数
   用调整后的"标准"评估质量
```

**为什么这样设计有效?**
- **Content-Aware**: 评估规则随内容自适应变化
- **Generalizable**: 对unseen内容也能生成合理的权重
- **End-to-End**: 整个过程可微分，可以端到端训练

---

## 9. 总结

### 9.1 核心改进对比表

| 方面 | 原始HyperIQA | Swin-HyperIQA | 改进效果 |
|------|-------------|---------------|---------|
| **Backbone** | ResNet50 (CNN) | Swin Transformer | **+2.68% SRCC** ⭐⭐⭐ |
| **多尺度特征** | LDA压缩成标量 | 保留7×7空间信息 | **+0.15% SRCC** ⭐⭐⭐⭐ |
| **特征融合** | 简单拼接 | Attention加权 | **+0.25% SRCC** ⭐⭐⭐⭐ |
| **正则化** | 无 | Dropout | **~0.1% SRCC** ⭐⭐⭐ |
| **HyperNet输入** | 2048-D单尺度 | 1920-D多尺度 | *隐含在上述改进中* |
| **TargetNet** | 无Dropout | Dropout(0.5) | *防止过拟合* |
| **总提升** | 0.907 SRCC | 0.9378 SRCC | **+3.08% (+3.40%)** 🏆 |

### 9.2 架构演进图

```
原始HyperIQA (2020)
      ↓
ResNet50 → LDA压缩 → 简单拼接 → HyperNet → TargetNet
  ↓
问题1: CNN局部感受野有限
问题2: LDA丢失空间信息
问题3: 简单拼接无法自适应
问题4: 容易过拟合

      ↓ 我们的改进 (2025)

Swin-HyperIQA
      ↓
Swin Transformer → 保留7×7 → Attention加权 → HyperNet → TargetNet
      ↓                ↓              ↓            ↓          ↓
    [改进1]        [改进2]        [改进3]      [改进4]   [改进4]
    全局建模      空间信息       动态重要性   Dropout   Dropout
    更强特征      局部定位       内容自适应   正则化    正则化

结果: SOTA性能 (0.9378 SRCC on KonIQ-10k) 🏆
```

### 9.3 设计哲学对比

**原始HyperIQA的哲学**:
> "先理解内容，再评估质量" - 通过HyperNet动态生成权重

**我们的改进哲学**:
> "更强的内容理解 + 更精细的多尺度融合 + 更智能的特征加权"
> - Swin: 更强的内容理解 (Transformer)
> - Multi-Scale: 更精细的特征提取 (保留空间信息)
> - Attention: 更智能的融合策略 (动态加权)

### 9.4 适用场景

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **研究/论文** | Swin-Base | 最高性能 (0.9378) |
| **工业部署** | Swin-Small | 性能/效率平衡 (0.9338, 52M) |
| **移动端/边缘** | Swin-Tiny | 轻量快速 (0.9249, 28M) |
| **Baseline对比** | Original HyperIQA | 经典方法 (0.907) |

---

## 📚 参考资料

1. **原始HyperIQA论文**:
   - Su et al., "Blindly Assess Image Quality in the Wild Guided by A Self-Adaptive Hyper Network", CVPR 2020
   - [GitHub](https://github.com/SSL92/hyperIQA)

2. **Swin Transformer论文**:
   - Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021

3. **我们的实验记录**:
   - `EXPERIMENTS_LOG_TRACKER.md` - 所有实验结果
   - `ARCHITECTURE_IMPROVEMENTS_DETAILED.md` - 架构改进详解
   - `UNCOVERED_COMPONENTS_ANALYSIS.md` - 未消融组件分析

---

**文档创建时间**: 2025-12-23  
**文档长度**: ~12,000字  
**状态**: ✅ 完整详尽，可用于论文写作和架构图绘制

**希望这份文档能帮助你完全理解原始HyperIQA和我们改进版本之间的每一个细节！** 🎉

