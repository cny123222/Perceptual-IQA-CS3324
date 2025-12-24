# 架构图修正指南

## 当前架构图的问题

### 问题1：AFA模块只处理3个stages
**错误**：绿色虚线框里只有3组 AAP→Conv1x1→BN
**正确**：应该有4组

### 问题2：Channel Attention只有3个权重
**错误**：橙色框里只显示 w₁, w₂, w₃
**正确**：应该显示 w₁, w₂, w₃, w₄

### 问题3：输入特征数量不匹配
**错误**：从Swin Transformer到AFA的绿色箭头只有3条
**正确**：应该有4条（对应4个stages）

## 正确的架构应该是

```
Input Image
    ↓
┌─────────────────────────────────────────────────┐
│   Swin Transformer Backbone                     │
│   Stage 0, 1, 2, 3                              │
└─────────────────────────────────────────────────┘
    ↓           ↓           ↓           ↓
  Stage 0     Stage 1     Stage 2     Stage 3
  (56×56)     (28×28)     (14×14)     (7×7)
    ↓           ↓           ↓           ↓
┌──────────────────────────────────────────────────┐
│  Adaptive Feature Aggregation (AFA) Module       │
│                                                  │
│  AAP → Conv1x1 → BN  (Stage 0 → 7×7)            │
│  AAP → Conv1x1 → BN  (Stage 1 → 7×7)            │
│  AAP → Conv1x1 → BN  (Stage 2 → 7×7)            │
│  (Stage 3 already 7×7, no AAP needed)           │
│                                                  │
└──────────────────────────────────────────────────┘
    ↓           ↓           ↓           ↓
  [7×7]       [7×7]       [7×7]       [7×7]
    ↓           ↓           ↓           ↓
┌──────────────────────────────────────────────────┐
│  Channel Attention Fusion Module                 │
│                                                  │
│  ┌─────────────────────┐                        │
│  │  GAP(Stage 3)       │                        │
│  │        ↓            │                        │
│  │   FC → ReLU → FC    │                        │
│  │        ↓            │                        │
│  │   Softmax           │                        │
│  │        ↓            │                        │
│  │  [w₁, w₂, w₃, w₄]   │  ← 4个权重！           │
│  └─────────────────────┘                        │
│                                                  │
│     ↓       ↓       ↓       ↓                    │
│    ⊗w₁    ⊗w₂    ⊗w₃    ⊗w₄                     │
│     ↓       ↓       ↓       ↓                    │
│  Weighted Features (element-wise multiply)      │
│     ↓       ↓       ↓       ↓                    │
│         Concatenate                              │
│               ↓                                  │
└──────────────────────────────────────────────────┘
               ↓
         HyperNet (3×Conv1x1)
               ↓
    Weights & Bias Generation
               ↓
         Target Network
               ↓
          Score Output
```

## 具体修改建议

### 修改1：AFA模块（绿色虚线框）
**当前**：3组 AAP→Conv1x1→BN
**改为**：显示4个输入
- 方案A：画4组完整的AAP→Conv1x1→BN
- 方案B：画3组AAP→Conv1x1→BN，然后加一条直通线（因为Stage 3已经是7×7不需要AAP）

**推荐方案B**（更准确）：
```
Stage 0 → AAP → Conv1x1 → BN → [7×7]
Stage 1 → AAP → Conv1x1 → BN → [7×7]
Stage 2 → AAP → Conv1x1 → BN → [7×7]
Stage 3 (已经7×7) ─────────→ [7×7]
```

### 修改2：Channel Attention模块（橙色虚线框）
**当前**：显示3个权重 w₁, w₂, w₃
**改为**：显示4个权重 w₁, w₂, w₃, w₄

具体：
1. 将权重框改为显示4个值：[w₁ w₂ w₃ w₄]
2. 从这个框画4条线到4个⊗符号
3. 每个⊗符号对应一个stage的特征

### 修改3：连接关系
确保：
1. **从Swin Transformer出来有4条绿色箭头** → 指向AFA模块
2. **AFA模块输出4条黄色箭头** → 指向Channel Attention区域
3. **4个加权特征** → Concat → 一条箭头 → HyperNet

### 修改4：标注文字（可选）
- Swin Transformer下方标注：**"Stage 0, 1, 2, 3"** 或 **"4 Hierarchical Stages"**
- AFA模块标注清楚：**"Adaptive Feature Aggregation Module (4 scales)"**
- Channel Attention标注：**"4 attention weights"**

## 关键点总结

**必须修改的地方**：
1. ✅ AFA模块：3组 → 4组（或3组+1条直通）
2. ✅ Attention权重：w₁w₂w₃ → w₁w₂w₃w₄
3. ✅ 输入/输出连线：确保所有地方都是4个流

**为什么是4个**：
- Swin Transformer有4个hierarchical stages
- 代码实现使用了全部4个stages: `[feat0, feat1, feat2, feat3]`
- Channel Attention生成4个权重，每个stage一个
- 实验结果显示的attention weights就是4个值

**不改的话会怎样**：
- 和实际代码实现不符
- 和实验结果（4个attention weights）不符
- 审稿人会发现不一致

## 论文其他地方也要检查

1. **Figure 1的caption**：
   - 错误："unifies spatial dimensions of **Stage 1-3 features**"
   - 正确："unifies spatial dimensions of **all four stage features**"

2. **正文描述**：
   - 确保提到"4 hierarchical stages"
   - 描述AFA时说"aggregates features from all four stages"
   - 描述Attention时说"generates 4 attention weights"

3. **Figure 6（注意力可视化）**：
   - Caption已经正确："across **four** Swin Transformer stages"
   - 图中显示的S1, S2, S3, S4也是4个，这个是对的

## 示例文字修改

### Figure 1 Caption（当前第63行）
**修改前**：
> (2) Adaptive Feature Aggregation (AFA) module that unifies spatial dimensions of Stage 1-3 features to 7×7

**修改后**：
> (2) Adaptive Feature Aggregation (AFA) module that unifies spatial dimensions of all four stage features to 7×7

### 正文描述（Section 3.2）
建议加一句：
> "The AFA module processes features from all four Swin Transformer stages, applying adaptive average pooling to unify their spatial dimensions to 7×7, followed by 1×1 convolutions and batch normalization. The channel attention mechanism then generates four attention weights, one for each stage, enabling dynamic feature weighting based on image content and distortion characteristics."

