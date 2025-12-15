# 训练超参数设置说明

## 当前超参数配置

### 权重衰减 (Weight Decay)

**设置值**: `5e-4` (0.0005)

**位置**:
- 默认值定义: `train_test_IQA.py:129`
- 应用位置: `HyerIQASolver.py:44, 48, 134`

**代码**:
```python
# 默认值
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, ...)

# 应用
self.weight_decay = config.weight_decay
self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)
```

**作用**: 
- L2正则化，防止过拟合
- 对所有权重参数应用相同的权重衰减

---

### 学习率设置

#### 初始学习率 (Learning Rate)

**Backbone (ResNet)**: `2e-5` (0.00002)
**HyperNetwork**: `2e-4` (0.0002) = Backbone LR × `lr_ratio`

**代码位置**:
- 默认值: `train_test_IQA.py:128`
- 学习率比率: `train_test_IQA.py:130`
- 应用: `HyerIQASolver.py:42-43, 45-46`

**配置**:
```python
--lr 2e-5              # Backbone初始学习率
--lr_ratio 10          # HyperNetwork学习率 = Backbone LR × 10
```

---

### 学习率衰减策略

**衰减公式**: `lr = initial_lr / pow(10, (epoch // 6))`

**代码位置**: `HyerIQASolver.py:124`

**具体衰减时间表**:

| Epoch 范围 | Backbone LR | HyperNetwork LR | 说明 |
|-----------|-------------|-----------------|------|
| 0-5       | 2e-5        | 2e-4            | 初始学习率 |
| 6-11      | 2e-6        | 2e-5            | 每6个epoch衰减10倍 |
| 12-17     | 2e-7        | 2e-6            | 继续衰减 |
| ...       | ...         | ...             | 以此类推 |

**特殊规则**:
- 当 `epoch > 8` 时，`lr_ratio = 1`
- 这意味着从 epoch 9 开始，HyperNetwork 和 Backbone 使用**相同的学习率**

**代码逻辑**:
```python
backbone_lr = self.lr / pow(10, (t // 6))  # 每6个epoch衰减10倍
hypernet_lr = backbone_lr * self.lrratio

if t > 8:
    self.lrratio = 1  # 从epoch 9开始，lr_ratio变为1
    hypernet_lr = backbone_lr  # HyperNetwork LR = Backbone LR
```

**实际衰减示例** (假设训练16个epoch):

| Epoch | Backbone LR | HyperNetwork LR | lr_ratio |
|-------|-------------|-----------------|----------|
| 0     | 2e-5        | 2e-4            | 10       |
| 1     | 2e-5        | 2e-4            | 10       |
| ...   | ...         | ...             | ...      |
| 5     | 2e-5        | 2e-4            | 10       |
| 6     | 2e-6        | 2e-5            | 10       |
| 7     | 2e-6        | 2e-5            | 10       |
| 8     | 2e-6        | 2e-5            | 10       |
| 9     | 2e-6        | **2e-6**        | **1**    |
| 10    | 2e-6        | 2e-6            | 1        |
| ...   | ...         | ...             | ...      |
| 12    | 2e-7        | 2e-7            | 1        |

---

## 其他训练参数

### 优化器
- **类型**: Adam
- **权重衰减**: 5e-4 (L2正则化)
- **Beta参数**: 默认 (β1=0.9, β2=0.999)

### 损失函数
- **类型**: L1 Loss (MAE - Mean Absolute Error)
- **公式**: `loss = |pred - label|`

---

## 参数调整建议

如果需要解决过拟合问题，可以考虑：

### 1. 增加权重衰减
```bash
--weight_decay 1e-3  # 从5e-4增加到1e-3
```

### 2. 降低初始学习率
```bash
--lr 1e-5  # 从2e-5降低到1e-5
```

### 3. 调整学习率比率
```bash
--lr_ratio 5  # 从10降低到5，减小HyperNetwork学习率
```

### 4. 修改学习率衰减策略
- 更频繁的衰减：改为每3个epoch衰减
- 更温和的衰减：改为每epoch衰减0.9倍

---

## 总结

**当前设置**:
- ✅ 权重衰减: **5e-4**
- ✅ 初始学习率: **2e-5** (Backbone), **2e-4** (HyperNetwork)
- ✅ 学习率衰减: **每6个epoch衰减10倍**
- ✅ 从epoch 9开始，HyperNetwork和Backbone使用相同学习率

这些设置与原论文保持一致。

