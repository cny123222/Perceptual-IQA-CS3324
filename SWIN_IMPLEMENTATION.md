# Swin Transformer Backbone 实现详解

## 1. 具体实现方法

### 1.1 整体架构替换

我将原始的 ResNet-50 backbone 替换为 Swin Transformer Tiny，主要修改点：

**原始架构 (ResNet-50)**:
```
Input (224×224×3)
  → ResNet-50 Backbone
    → Layer1 (256 ch, 56×56)
    → Layer2 (512 ch, 28×28)
    → Layer3 (1024 ch, 14×14)
    → Layer4 (2048 ch, 7×7)  [用于 hyper network]
```

**新架构 (Swin Transformer Tiny)**:
```
Input (224×224×3)
  → Swin Transformer Tiny Backbone
    → Stage 1 (96 ch, 56×56)
    → Stage 2 (192 ch, 28×28)
    → Stage 3 (384 ch, 14×14)
    → Stage 4 (768 ch, 7×7)   [用于 hyper network]
```

### 1.2 代码实现步骤

#### 步骤 1: 创建 SwinBackbone 类 (`models_swin.py`)

```python
class SwinBackbone(nn.Module):
    def __init__(self, lda_out_channels, in_chn):
        super(SwinBackbone, self).__init__()
        
        # 使用 timm 库创建预训练模型
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,           # ✅ 使用 ImageNet 预训练权重
            features_only=True,        # ✅ 仅提取特征图，不包含分类头
            out_indices=(0, 1, 2, 3)  # ✅ 提取所有4个阶段的特征
        )
```

#### 步骤 2: 为每个阶段设计 LDA (Local Distortion Aware) 模块

每个阶段的特征图需要经过 LDA 模块提取局部失真感知特征：

**Stage 1** (96通道, 56×56):
```python
self.lda1_pool = nn.Sequential(
    nn.Conv2d(96, 16, kernel_size=1),      # 降维: 96→16
    nn.AvgPool2d(7, stride=7)              # 池化: 56×56 → 8×8
)
self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)  # 16×8×8=1024 → lda_out
```

**Stage 2** (192通道, 28×28):
```python
self.lda2_pool = nn.Sequential(
    nn.Conv2d(192, 32, kernel_size=1),     # 降维: 192→32
    nn.AvgPool2d(7, stride=7)              # 池化: 28×28 → 4×4
)
self.lda2_fc = nn.Linear(32 * 16, lda_out_channels)  # 32×4×4=512 → lda_out
```

**Stage 3** (384通道, 14×14):
```python
self.lda3_pool = nn.Sequential(
    nn.Conv2d(384, 64, kernel_size=1),     # 降维: 384→64
    nn.AvgPool2d(7, stride=7)              # 池化: 14×14 → 2×2
)
self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)   # 64×2×2=256 → lda_out
```

**Stage 4** (768通道, 7×7):
```python
self.lda4_pool = nn.AvgPool2d(7, stride=7)  # 池化: 7×7 → 1×1
self.lda4_fc = nn.Linear(768, in_chn - 3*lda_out_channels)  # 768 → (224-3*lda_out)
```

#### 步骤 3: 多尺度特征融合

```python
def forward(self, x):
    # 提取多尺度特征
    features = self.backbone(x)  # 返回4个特征图的列表
    
    # 对每个阶段应用 LDA 模块
    lda_1 = self.lda1_fc(self.lda1_pool(features[0]).view(x.size(0), -1))
    lda_2 = self.lda2_fc(self.lda2_pool(features[1]).view(x.size(0), -1))
    lda_3 = self.lda3_fc(self.lda3_pool(features[2]).view(x.size(0), -1))
    lda_4 = self.lda4_fc(self.lda4_pool(features[3]).view(x.size(0), -1))
    
    # 拼接所有尺度的特征 → target_in_vec (用于 Target Network)
    vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)
    
    # Stage 4 的特征用于 Hyper Network
    return {
        'hyper_in_feat': features[3],  # [B, 768, 7, 7]
        'target_in_vec': vec           # [B, 224] (4个LDA输出拼接)
    }
```

#### 步骤 4: 适配 Hyper Network

```python
class HyperNet(nn.Module):
    def __init__(self, ...):
        self.swin = swin_backbone(lda_out_channels, target_in_size, pretrained=True)
        
        # 将 Swin 的 768 通道特征压缩到 112 通道 (用于 Hyper Network)
        self.conv1 = nn.Sequential(
            nn.Conv2d(768, 512, 1),  # 768 → 512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),  # 512 → 256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 112, 1),  # 256 → 112 (hyperInChn)
            nn.ReLU(inplace=True)
        )
```

---

## 2. 预训练权重和特征提取设置

### 2.1 ✅ 设置了 `pretrained=True`

**代码位置**: `models_swin.py` 第 180 行和第 35 行

```python
self.backbone = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,  # ✅ 加载 ImageNet 预训练权重
    ...
)
```

**作用**:
- 使用在 ImageNet 上预训练的 Swin Transformer Tiny 权重
- 提供更好的特征表示，加速收敛
- 相比从零训练，能更好地提取通用图像特征

### 2.2 ✅ 设置了 `features_only=True`

**代码位置**: `models_swin.py` 第 181 行

```python
self.backbone = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    features_only=True,  # ✅ 仅提取特征图，不包含分类头
    out_indices=(0, 1, 2, 3)  # ✅ 指定提取哪些阶段的特征
)
```

**作用**:
- 移除最后的分类层，只保留特征提取部分
- 返回中间层的特征图列表，而不是最终的分类结果
- `out_indices=(0, 1, 2, 3)` 指定提取所有4个阶段的特征图

**返回格式**:
```python
features = self.backbone(x)  
# features[0]: Stage 1 特征 [B, 96, 56, 56]
# features[1]: Stage 2 特征 [B, 192, 28, 28]
# features[2]: Stage 3 特征 [B, 384, 14, 14]
# features[3]: Stage 4 特征 [B, 768, 7, 7]
```

---

## 3. 多尺度特征融合与下游网络适配

### 3.1 多尺度特征提取策略

Swin Transformer 的4个阶段天然提供多尺度特征：

```
Stage 1: 96 ch,  56×56  → 细粒度局部特征 (局部纹理、边缘)
Stage 2: 192 ch, 28×28  → 中等粒度特征 (局部模式、小块区域)
Stage 3: 384 ch, 14×14  → 粗粒度特征 (区域语义、较大结构)
Stage 4: 768 ch, 7×7    → 全局语义特征 (图像级表示、全局上下文)
```

### 3.2 LDA 模块设计

每个阶段通过 LDA 模块提取局部失真感知特征：

```python
# Stage 1: 局部纹理失真
[B, 96, 56, 56] 
  → Conv(96→16) → [B, 16, 56, 56]
  → AvgPool(7×7) → [B, 16, 8, 8]
  → Flatten → [B, 1024]
  → Linear → [B, lda_out_channels]  # 例如: [B, 16]

# Stage 2: 中等范围失真
[B, 192, 28, 28]
  → Conv(192→32) → [B, 32, 28, 28]
  → AvgPool(7×7) → [B, 32, 4, 4]
  → Flatten → [B, 512]
  → Linear → [B, lda_out_channels]

# Stage 3: 区域级失真
[B, 384, 14, 14]
  → Conv(384→64) → [B, 64, 14, 14]
  → AvgPool(7×7) → [B, 64, 2, 2]
  → Flatten → [B, 256]
  → Linear → [B, lda_out_channels]

# Stage 4: 全局语义特征
[B, 768, 7, 7]
  → AvgPool(7×7) → [B, 768, 1, 1]
  → Flatten → [B, 768]
  → Linear → [B, in_chn - 3*lda_out_channels]  # 例如: [B, 176]
```

### 3.3 特征融合方式

**目标网络输入向量 (target_in_vec)**:
```python
target_in_vec = torch.cat([lda_1, lda_2, lda_3, lda_4], dim=1)
# [B, 16] + [B, 16] + [B, 16] + [B, 176] = [B, 224]
```

**融合优势**:
- ✅ **多尺度感知**: 同时捕获从局部到全局的失真信息
- ✅ **层次化表示**: 不同阶段关注不同粒度的图像质量
- ✅ **互补信息**: 细粒度特征(Stage 1-3) + 全局语义(Stage 4)

### 3.4 Hyper Network 特征适配

**全局语义特征 (Stage 4)**:
```python
hyper_in_feat = features[3]  # [B, 768, 7, 7] - 全局语义特征

# 降维适配 Hyper Network
hyper_in_feat = self.conv1(hyper_in_feat)  
# [B, 768, 7, 7] 
#   → [B, 512, 7, 7]
#   → [B, 256, 7, 7]
#   → [B, 112, 7, 7]
```

**为什么使用 Stage 4**:
- 全局语义信息适合生成 Target Network 的权重
- 7×7 的空间尺寸匹配原始 ResNet-50 的设计
- 768 维特征包含丰富的图像级表示

### 3.5 下游网络流程

```
输入图像 (224×224×3)
    ↓
Swin Transformer Backbone
    ├─→ Stage 1-4 多尺度特征
    │   ├─→ Stage 1-3: LDA模块 → 局部失真特征
    │   └─→ Stage 4: LDA模块 → 全局语义特征
    │           ↓
    │   拼接 → target_in_vec [B, 224]
    │           ↓
    │       用于 Target Network 输入
    │
    └─→ Stage 4 特征 [B, 768, 7, 7]
            ↓
        降维 → [B, 112, 7, 7]
            ↓
        用于 Hyper Network
            ↓
        生成 Target Network 的权重和偏置
```

### 3.6 关键设计优势

1. **多尺度融合**:
   - 不同尺度捕获不同类型的图像失真
   - Stage 1-3: 关注局部纹理、细节失真
   - Stage 4: 关注全局结构、语义失真

2. **特征维度适配**:
   - LDA 模块统一各阶段的输出维度
   - 通过池化将不同空间尺寸归一化
   - 最终拼接成固定长度的向量

3. **保持架构一致性**:
   - Target Network 输入维度仍为 224 (与原 ResNet 版本一致)
   - Hyper Network 输入仍为 112 通道 (与原设计一致)
   - 无需修改下游网络结构

---

## 4. 对比原始 ResNet-50 版本

| 特性 | ResNet-50 版本 | Swin Transformer 版本 |
|------|---------------|---------------------|
| **Backbone** | ResNet-50 | Swin-T Tiny |
| **特征通道数** | 256/512/1024/2048 | 96/192/384/768 |
| **Stage 4 用于 Hyper** | 2048 ch | 768 ch (需降维到112) |
| **多尺度特征** | 4个阶段 | 4个阶段 |
| **LDA 模块数** | 4个 | 4个 |
| **预训练权重** | ImageNet | ImageNet |
| **优势** | 成熟的 CNN 架构 | 更强的全局建模能力 |

---

## 5. 训练配置

在 `HyperIQASolver_swin.py` 中，Swin backbone 参数使用较小的学习率：

```python
paras = [
    {'params': self.hypernet_params, 'lr': self.lr * self.lrratio},  # 10倍学习率
    {'params': self.model_hyper.swin.parameters(), 'lr': self.lr}    # 基础学习率
]
```

这样可以：
- 预训练权重微调：backbone 使用较小学习率
- 新模块快速学习：LDA 和 Hyper Network 使用较大学习率

