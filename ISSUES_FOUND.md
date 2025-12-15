# 在当前分支发现的问题

## 分支信息
- **分支**: `fix-training-issue` (基于 `a4d1eda017d8ac9a8a04c62d73593ae6e6f77b92`)
- **状态**: 已确认SRCC/PLCC计算逻辑与原始论文一致

---

## 发现的问题

### 🔴 问题1: Backbone学习率不衰减

**位置**: `HyerIQASolver.py:122-129`

**代码**:
```python
lr = self.lr / pow(10, (t // 6))  # 只有hypernet的lr会衰减
if t > 8:
    self.lrratio = 1
self.paras = [
    {'params': self.hypernet_params, 'lr': lr * self.lrratio},  # 会衰减
    {'params': self.model_hyper.res.parameters(), 'lr': self.lr}  # ❌ 永远不会衰减！
]
self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
```

**问题**:
- HyperNetwork的学习率会衰减（每6个epoch衰减10倍）
- **Backbone (ResNet) 的学习率始终保持为初始值 `self.lr`**
- 这意味着Backbone在整个训练过程中一直用**高学习率**更新
- 导致Backbone过度更新，破坏了预训练特征，泛化能力下降

**影响**:
- Epoch 1: Backbone特征还比较接近预训练权重，泛化好 ✅
- Epoch 2+: Backbone持续被高学习率更新，偏离预训练权重，泛化变差 ❌

---

### 🔴 问题2: 优化器状态被重置

**位置**: `HyerIQASolver.py:129`

**代码**:
```python
self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
```

**问题**:
- 每个epoch结束后，都重新创建optimizer
- **Adam的momentum buffers（一阶和二阶矩估计）被清空**
- 这意味着每个epoch都是"从头开始"的Adam优化，失去了历史梯度信息
- 可能导致训练不稳定

**正确做法**:
- 应该只更新learning rate，而不是重新创建optimizer
- 使用 `optimizer.param_groups[i]['lr'] = new_lr` 来更新学习率

---

### ⚠️ 问题3: filter() 迭代器耗尽bug

**位置**: `HyerIQASolver.py:40`

**代码**:
```python
self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
```

**问题**:
- `filter()` 返回一个迭代器，只能使用一次
- 在第一次创建optimizer后，迭代器被耗尽
- 后续epoch重新创建optimizer时，`self.hypernet_params` 是空的
- 这会导致只有backbone参数被优化，hypernetwork参数不被更新

**修复**:
```python
self.hypernet_params = list(filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters()))
```

---

## 修复方案

### 修复1: Backbone学习率也衰减

```python
# 修复后
backbone_lr = self.lr / pow(10, (t // 6))  # Backbone LR也衰减
hypernet_lr = backbone_lr * self.lrratio
if t > 8:
    self.lrratio = 1
    hypernet_lr = backbone_lr
```

### 修复2: 保持优化器状态

```python
# 修复后
if t == 0:
    # First epoch: create optimizer
    self.paras = [
        {'params': self.hypernet_params, 'lr': hypernet_lr},
        {'params': self.model_hyper.res.parameters(), 'lr': backbone_lr}
    ]
    self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
else:
    # Subsequent epochs: only update learning rates
    self.solver.param_groups[0]['lr'] = hypernet_lr
    self.solver.param_groups[1]['lr'] = backbone_lr
```

### 修复3: 修复filter() bug

```python
# 修复后
self.hypernet_params = list(filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters()))
```

---

## 预期效果

修复这些问题后，预期：
1. ✅ Backbone特征不会过度偏离预训练权重
2. ✅ 保持更好的泛化能力
3. ✅ 训练更稳定
4. ✅ 测试SRCC可能在后续epoch保持稳定或继续提升

---

## 下一步

1. 应用所有三个修复
2. 运行训练，观察测试SRCC是否在后续epoch保持或提升
3. 如果仍有问题，考虑添加Early Stopping

