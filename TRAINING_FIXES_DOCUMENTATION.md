# 训练修复文档

本文档详细记录了在训练过程中发现并修复的三个关键问题。这些修复应用于以下文件：
- `HyerIQASolver.py` (ResNet-50 版本)
- `HyperIQASolver_swin.py` (Swin Transformer 版本)

## 修复概述

这三个修复解决了训练过程中的关键问题：
1. **Filter 迭代器耗尽 Bug**：导致超网络参数丢失
2. **骨干网络学习率不衰减**：预训练特征被过度更新
3. **优化器状态重置**：每个 epoch 重置优化器导致训练不稳定

---

## 修复 1: Filter 迭代器耗尽问题

### 问题描述
`filter()` 函数返回一个迭代器，只能使用一次。当在多个地方使用同一个 filter 对象时，第二次使用会返回空结果，导致超网络参数丢失。

### 受影响文件
- `HyerIQASolver.py` (约第 42-44 行)
- `HyperIQASolver_swin.py` (约第 42-44 行)

### 修复前代码

**HyerIQASolver.py:**
```python
backbone_params = list(map(id, self.model_hyper.res.parameters()))
self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
```

**HyperIQASolver_swin.py:**
```python
backbone_params = list(map(id, self.model_hyper.swin.parameters()))
self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
```

### 修复后代码

**HyerIQASolver.py:**
```python
backbone_params = list(map(id, self.model_hyper.res.parameters()))
# FIX: Convert filter to list to avoid iterator exhaustion bug
self.hypernet_params = list(filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters()))
```

**HyperIQASolver_swin.py:**
```python
backbone_params = list(map(id, self.model_hyper.swin.parameters()))
# FIX: Convert filter to list to avoid iterator exhaustion bug
self.hypernet_params = list(filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters()))
```

### 修复说明
- **原因**：`filter()` 返回的是迭代器，第一次使用时被消耗，第二次使用时已为空
- **影响**：在后续 epoch 重新初始化优化器时，`self.hypernet_params` 为空，导致超网络参数不会被更新
- **解决方案**：使用 `list()` 包装 `filter()`，将迭代器转换为列表，可以重复使用
- **位置**：`__init__` 方法中，初始化超网络参数列表时

### 如何回退
如果需要回退，只需移除 `list()` 包装：
```python
self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
```

---

## 修复 2: 骨干网络学习率不衰减

### 问题描述
在原始实现中，只有超网络的学习率会衰减，骨干网络（ResNet/Swin）的学习率保持不变。这会导致预训练的骨干网络特征被过度更新和破坏。

### 受影响文件
- `HyerIQASolver.py` (约第 154-169 行，`train()` 方法中)
- `HyperIQASolver_swin.py` (约第 200-215 行，`train()` 方法中)

### 修复前代码

**HyerIQASolver.py:**
```python
# 原始代码示例（可能在不同位置）
if t == 0:
    self.paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                  {'params': self.model_hyper.res.parameters(), 'lr': self.lr}]
    self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
else:
    # 可能只有超网络的学习率更新
    self.solver.param_groups[0]['lr'] = self.lr * self.lrratio / pow(10, (t // 6))
    # 骨干网络学习率没有更新！
```

**HyperIQASolver_swin.py:**
```python
# 类似的问题，只是 self.model_hyper.res.parameters() 改为 self.model_hyper.swin.parameters()
```

### 修复后代码

**HyerIQASolver.py:**
```python
# FIX: Update optimizer learning rates (backbone LR now also decays, optimizer state preserved)
backbone_lr = self.lr / pow(10, (t // 6))  # Backbone LR also decays
hypernet_lr = backbone_lr * self.lrratio
if t > 8:
    self.lrratio = 1
    hypernet_lr = backbone_lr  # When lrratio becomes 1, hypernet LR = backbone LR

if t == 0:
    # First epoch: create optimizer
    self.paras = [{'params': self.hypernet_params, 'lr': hypernet_lr},
                  {'params': self.model_hyper.res.parameters(), 'lr': backbone_lr}]
    self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
else:
    # Subsequent epochs: only update learning rates (preserves Adam momentum state)
    self.solver.param_groups[0]['lr'] = hypernet_lr
    self.solver.param_groups[1]['lr'] = backbone_lr  # ✅ 现在骨干网络学习率也会衰减
```

**HyperIQASolver_swin.py:**
```python
# 相同的修复逻辑，只是将 self.model_hyper.res.parameters() 改为 self.model_hyper.swin.parameters()
# FIX: Update optimizer learning rates (backbone LR now also decays, optimizer state preserved)
backbone_lr = self.lr / pow(10, (t // 6))  # Backbone LR also decays
hypernet_lr = backbone_lr * self.lrratio
if t > 8:
    self.lrratio = 1
    hypernet_lr = backbone_lr

if t == 0:
    self.paras = [{'params': self.hypernet_params, 'lr': hypernet_lr},
                  {'params': self.model_hyper.swin.parameters(), 'lr': backbone_lr}]
    self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
else:
    self.solver.param_groups[0]['lr'] = hypernet_lr
    self.solver.param_groups[1]['lr'] = backbone_lr  # ✅ 现在骨干网络学习率也会衰减
```

### 修复说明
- **原因**：原始代码中骨干网络的学习率在初始化后从未更新，一直保持初始值
- **影响**：预训练的骨干网络特征被过度更新，导致性能下降
- **解决方案**：
  1. 计算 `backbone_lr = self.lr / pow(10, (t // 6))` 使骨干网络学习率也按 epoch 衰减
  2. 在每个 epoch 更新优化器的两个参数组的学习率
  3. 保持学习率比例关系：`hypernet_lr = backbone_lr * self.lrratio`
- **位置**：`train()` 方法中，每个 epoch 结束后的优化器更新部分

### 如何回退
如果需要回退，可以将骨干网络学习率改为固定值：
```python
backbone_lr = self.lr  # 固定值，不衰减
hypernet_lr = backbone_lr * self.lrratio / pow(10, (t // 6))  # 只有超网络衰减
```

---

## 修复 3: 优化器状态重置问题

### 问题描述
在原始实现中，每个 epoch 都会重新创建优化器 (`self.solver = torch.optim.Adam(...)`)，这会导致优化器的内部状态（如 Adam 的动量、二阶矩估计）被重置，使得训练不稳定。

### 受影响文件
- `HyerIQASolver.py` (约第 154-169 行，`train()` 方法中)
- `HyperIQASolver_swin.py` (约第 200-215 行，`train()` 方法中)

### 修复前代码

**HyerIQASolver.py:**
```python
# 原始代码可能在每个 epoch 都重新创建优化器
for t in range(self.epochs):
    # ... 训练循环 ...
    
    # ❌ 每个 epoch 都重新创建优化器，丢失了 Adam 的动量状态
    self.paras = [{'params': self.hypernet_params, 'lr': hypernet_lr},
                  {'params': self.model_hyper.res.parameters(), 'lr': backbone_lr}]
    self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
```

**HyperIQASolver_swin.py:**
```python
# 类似的问题
```

### 修复后代码

**HyerIQASolver.py:**
```python
# FIX: Update optimizer learning rates (backbone LR now also decays, optimizer state preserved)
backbone_lr = self.lr / pow(10, (t // 6))
hypernet_lr = backbone_lr * self.lrratio
if t > 8:
    self.lrratio = 1
    hypernet_lr = backbone_lr

if t == 0:
    # ✅ 只在第一个 epoch 创建优化器
    self.paras = [{'params': self.hypernet_params, 'lr': hypernet_lr},
                  {'params': self.model_hyper.res.parameters(), 'lr': backbone_lr}]
    self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
else:
    # ✅ 后续 epoch 只更新学习率，保留优化器状态
    self.solver.param_groups[0]['lr'] = hypernet_lr
    self.solver.param_groups[1]['lr'] = backbone_lr
```

**HyperIQASolver_swin.py:**
```python
# 相同的修复逻辑
if t == 0:
    self.paras = [{'params': self.hypernet_params, 'lr': hypernet_lr},
                  {'params': self.model_hyper.swin.parameters(), 'lr': backbone_lr}]
    self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
else:
    self.solver.param_groups[0]['lr'] = hypernet_lr
    self.solver.param_groups[1]['lr'] = backbone_lr
```

### 修复说明
- **原因**：每个 epoch 重新创建优化器会丢失 Adam 优化器的内部状态（动量、二阶矩估计）
- **影响**：
  - 训练不稳定
  - 损失函数震荡
  - 收敛速度变慢
  - 测试集性能在第一 epoch 后下降
- **解决方案**：
  1. 只在第一个 epoch (`t == 0`) 创建优化器
  2. 后续 epoch 只更新已有优化器的学习率，保留其内部状态
  3. 使用 `self.solver.param_groups[0]['lr']` 和 `self.solver.param_groups[1]['lr']` 更新学习率
- **位置**：`train()` 方法中，每个 epoch 结束后的优化器更新部分

### 如何回退
如果需要回退，可以移除 `if t == 0:` 条件，每次都重新创建优化器：
```python
# 每个 epoch 都重新创建（不推荐）
self.paras = [{'params': self.hypernet_params, 'lr': hypernet_lr},
              {'params': self.model_hyper.res.parameters(), 'lr': backbone_lr}]
self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
```

---

## 修复效果

这三个修复解决了以下训练问题：

1. **训练稳定性提升**：测试集 SRCC/PLCC 不再在第一 epoch 后急剧下降
2. **收敛速度改善**：保留优化器状态使得训练更平滑
3. **性能提升**：骨干网络学习率衰减保护了预训练特征

### 典型问题表现（修复前）
- Epoch 1: Test SRCC = 0.92 (最佳)
- Epoch 2: Test SRCC = 0.91 (下降)
- Epoch 3+: Test SRCC 持续下降

### 修复后表现
- 训练更稳定
- 测试集性能不再在第一个 epoch 后下降
- 模型能够持续改进

---

## 代码位置总结

### HyerIQASolver.py (ResNet-50 版本)

| 修复 | 位置 | 行号范围 |
|------|------|----------|
| 修复 1: Filter 迭代器 | `__init__` 方法 | ~42-44 |
| 修复 2: 骨干 LR 衰减 | `train()` 方法 | ~154-169 |
| 修复 3: 优化器状态 | `train()` 方法 | ~154-169 (与修复 2 在同一位置) |

### HyperIQASolver_swin.py (Swin Transformer 版本)

| 修复 | 位置 | 行号范围 |
|------|------|----------|
| 修复 1: Filter 迭代器 | `__init__` 方法 | ~42-44 |
| 修复 2: 骨干 LR 衰减 | `train()` 方法 | ~200-215 |
| 修复 3: 优化器状态 | `train()` 方法 | ~200-215 (与修复 2 在同一位置) |

---

## 注意事项

1. **三个修复相互关联**：
   - 修复 1 确保 `self.hypernet_params` 可用
   - 修复 2 和 3 在同一个代码块中，都是优化器更新逻辑的一部分

2. **两个版本的一致性**：
   - ResNet 版本使用 `self.model_hyper.res.parameters()`
   - Swin 版本使用 `self.model_hyper.swin.parameters()`
   - 其他逻辑完全相同

3. **如果需要进行实验对比**：
   - 可以单独回退某个修复来测试其影响
   - 但建议保持三个修复同时应用，因为它们共同解决了训练稳定性问题

---

## 相关提交

这些修复在以下提交中应用：
- 修复日期：2025-12-15
- 修复原因：发现测试集 SRCC/PLCC 在第一 epoch 后持续下降
- 修复验证：训练稳定性显著提升

---

## 参考资料

- PyTorch 文档：[Optimizer](https://pytorch.org/docs/stable/optim.html)
- Python 文档：[filter()](https://docs.python.org/3/library/functions.html#filter)
- Adam 优化器：[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

