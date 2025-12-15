# SRCC/PLCC 计算逻辑对比分析

## 对比Commit
- **当前分支**: `a4d1eda017d8ac9a8a04c62d73593ae6e6f77b92` (原始实现)
- **原始论文**: `c42e7279717e7dcb693a24b891fc14a4189a45ee`

## 对比结果

### ✅ **测试集上的SRCC/PLCC计算 - 完全一致**

#### 当前分支 (`HyerIQASolver.py:163-166`)
```python
pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
```

#### 原始论文 (`c42e727`)
```python
pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
```

**结论**: ✅ **完全一致** - 都是在图像级别计算（先对每个图像的所有patch取平均，再计算相关性）

---

### ✅ **训练集上的SRCC计算 - 完全一致**

#### 当前分支 (`HyerIQASolver.py:107`)
```python
train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
# 其中 pred_scores 和 gt_scores 是收集的所有patch的预测和标签
```

#### 原始论文 (`c42e727`)
```python
train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
# 其中 pred_scores 和 gt_scores 是收集的所有patch的预测和标签
```

**结论**: ✅ **完全一致** - 都是在patch级别计算（所有patch的预测和标签直接计算相关性）

---

## 关键发现

### 1. **评估尺度不一致是原论文的设计**

这是**有意的设计选择**，不是bug：
- **训练SRCC**: 在patch级别计算（所有patch的预测vs标签）
- **测试SRCC**: 在图像级别计算（先对patch取平均，再计算图像级别的相关性）

### 2. **这解释了为什么训练SRCC和测试SRCC差异很大**

- 训练SRCC在patch级别，噪声更大，更容易拟合
- 测试SRCC在图像级别，需要模型对同一图像的不同patch预测一致

### 3. **训练问题与SRCC/PLCC计算逻辑无关**

既然计算逻辑与原始论文一致，那么：
- **第一个epoch测试SRCC最高，后续下降的问题** 不是由SRCC/PLCC计算方式造成的
- 问题应该出在**训练策略**或**模型架构特性**上

---

## 下一步

既然计算逻辑是正确的，我们需要专注于解决训练问题：

1. **检查优化器设置**：Backbone学习率不衰减的问题
2. **检查优化器状态**：每个epoch重新创建optimizer导致momentum丢失
3. **考虑Early Stopping**：既然第一个epoch就是最好的，可以提前停止

