# 架构实现验证 - 重要发现！

## 问题
用户发现：架构图只画了3个向量有权重，但实际注意力权重分析显示有4个stage都有权重。

## 实际实现检查

### 代码实现（models_swin.py）

```python
class MultiScaleAttention(nn.Module):
    def forward(self, feat_list):
        """
        Args:
            feat_list: List of 4 feature maps [feat0, feat1, feat2, feat3]
        """
        # 生成4个attention weights
        attention_weights = self.attention_net(feat3_global)  # [B, 4]
        
        # 对4个特征都应用权重
        for i, feat in enumerate(feats_pooled):
            weight = attention_weights[:, i].view(B, 1, 1, 1)
            weighted_feat = feat * weight
```

在HyperNet的forward中：
```python
feat0, feat1, feat2, feat3 = swin_out['hyper_in_feat_multi']
hyper_in_feat_raw, attention_weights = self.multiscale_attention([feat0, feat1, feat2, feat3])
```

**结论：实际使用了4个stages (feat0, feat1, feat2, feat3)**

### Swin Transformer的Stage编号

Swin Transformer标准架构：
- **Patch Embedding** -> [B, 96/128, 56, 56]
- **Stage 0** (layer1) -> [B, 96/128, 56, 56]  ← feat0
- **Stage 1** (layer2) -> [B, 192/256, 28, 28]  ← feat1
- **Stage 2** (layer3) -> [B, 384/512, 14, 14]  ← feat2
- **Stage 3** (layer4) -> [B, 768/1024, 7, 7]   ← feat3

### 论文中的描述

Figure 1 caption说：
> "Adaptive Feature Aggregation (AFA) module that unifies spatial dimensions of **Stage 1-3 features** to 7×7"

但是论文正文说：
> "The Swin Transformer produces features at four stages... Stage 0 (56×56), Stage 1 (28×28), Stage 2 (14×14), and Stage 3 (7×7)."

Figure 6 caption说：
> "Attention weights across **four Swin Transformer stages**"

## 不一致之处

1. **Architecture图的caption**说只用了Stage 1-3（3个stages）
2. **实际代码**使用了所有4个stages (Stage 0-3)
3. **Attention可视化结果**显示有4个权重（S1, S2, S3, S4）

## 正确的理解

实际实现：
- 使用了**Swin Transformer的全部4个stages**
- 即：feat0 (Stage 0), feat1 (Stage 1), feat2 (Stage 2), feat3 (Stage 3)
- Channel Attention为所有4个stages分配权重
- 最终融合：concat([weighted_feat0, weighted_feat1, weighted_feat2, weighted_feat3])

## 需要修正

### 选项1：修改论文描述（推荐）
将Figure 1 caption改为：
> "...unifies spatial dimensions of **Stage 0-3 features** to 7×7..."

或者：
> "...unifies spatial dimensions of **all four stage features** to 7×7..."

### 选项2：修改代码（不推荐，会影响已训练模型）
如果真的要只用3个stages，需要：
1. 移除feat0
2. 只使用[feat1, feat2, feat3]
3. Attention权重从4个改为3个
4. 重新训练模型

**强烈不推荐选项2**，因为：
- 已经训练好的模型使用4个stages
- 实验结果基于4个stages
- 修改会导致所有结果失效

## 建议行动

1. ✅ **修改论文文字**：将所有"Stage 1-3"改为"Stage 0-3"或"four stages"
2. ✅ **检查架构图**：确保架构图显示4个feature maps和4个attention weights
3. ✅ **统一术语**：
   - 论文中明确说明使用"4个hierarchical stages"
   - 在所有描述中保持一致

## 架构图应该显示

```
Swin Transformer Backbone
    ↓
[feat0]  [feat1]  [feat2]  [feat3]
56×56    28×28    14×14    7×7
  ↓        ↓        ↓        ↓
AdaptiveAvgPool2d(7, 7)
  ↓        ↓        ↓        ↓
[7×7]    [7×7]    [7×7]    [7×7]
  ↓        ↓        ↓        ↓
  ↓  ┌─────────────────────┐
  ↓  │ Channel Attention   │
  ↓  │    (feat3 → 4 weights)│
  ↓  └─────────────────────┘
  ↓        ↓        ↓        ↓
  w0       w1       w2       w3
  ↓        ↓        ↓        ↓
weighted features (element-wise multiply)
  ↓        ↓        ↓        ↓
    Concatenate
        ↓
    HyperNet
```

每个特征都有对应的attention weight！

