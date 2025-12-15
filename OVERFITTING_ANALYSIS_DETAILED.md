# 过拟合问题深度分析：为什么 Loss 下降但测试指标下降？

## 问题现象

在使用 Ranking Loss (alpha=1.0) 的训练过程中，观察到以下现象：
- ✅ **训练 Loss 持续下降**：从 5.937 → 1.534
- ✅ **训练集 SRCC 持续上升**：从 0.8684 → 0.9892
- ❌ **测试集 SRCC/PLCC 持续下降**：从 0.9188 → 0.9107 (SRCC), 从 0.9326 → 0.9212 (PLCC)

这是典型的**过拟合（Overfitting）**问题。

---

## 根本原因分析

### 1. **评估尺度的不一致性**

这是导致"Loss 下降但测试指标下降"的核心原因：

#### **训练时的评估方式**：
- **Loss 计算**：在 **patch 级别**计算 L1 Loss 和 Ranking Loss
  - 每个 patch 有独立的预测值和标签值
  - Loss = L1(pred_patch, label_patch) + alpha * Ranking_Loss(pred_patches_in_batch)
- **训练 SRCC 计算**：在 **patch 级别**计算
  - 代码位置：`HyperIQASolver_swin.py:142`
  - `train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)` 
  - 这里的 `pred_scores` 和 `gt_scores` 是所有 patch 的预测和标签列表

#### **测试时的评估方式**：
- **测试 SRCC/PLCC 计算**：在 **图像级别**计算
  - 代码位置：`HyperIQASolver_swin.py:251-254`
  - 先将所有 patch 预测值重塑为 `(num_images, test_patch_num)` 形状
  - 对每个图像的所有 patch **取平均值**：`np.mean(..., axis=1)`
  - 然后在图像级别的平均预测值和标签值上计算 SRCC/PLCC

#### **关键差异**：
```
训练时：Patch 级别 (更多噪声，更难拟合)
  - 每个 patch 独立计算
  - 同一图像的不同 patch 可能预测不一致
  
测试时：图像级别 (更稳定，需要一致性)
  - 同一图像的所有 patch 必须预测一致
  - 需要模型对图像整体质量有准确理解
```

### 2. **Loss 函数优化目标与评估指标的不匹配**

#### **L1 Loss 的优化目标**：
- 最小化 patch 级别的**绝对误差**
- 只要每个 patch 的预测值接近其标签值，Loss 就会下降
- 但**不要求**同一图像的不同 patch 预测一致

#### **Ranking Loss 的优化目标**：
- 优化 patch 之间的**相对排序**
- 在 batch 内，要求预测的相对顺序与标签的相对顺序一致
- **但仅在 batch 内有效**，跨 batch 的排序可能不一致

#### **SRCC/PLCC 的评估目标**：
- 要求**图像级别**的预测值与真实质量分数的**单调关系**（SRCC）或**线性关系**（PLCC）
- 需要模型能够：
  1. 对同一图像的不同 patch 预测一致
  2. 对图像整体质量有准确理解

### 3. **过拟合的具体机制**

#### **阶段 1：初期（Epoch 1-2）**
- 模型学习到基本的图像质量特征
- **图像级别**的预测相对准确，测试 SRCC/PLCC 较高
- Loss 下降，训练 SRCC 上升，测试指标也上升（或保持）

#### **阶段 2：过度优化（Epoch 3+）**
- 模型开始在 **patch 级别**过度拟合训练数据
- Ranking Loss 迫使模型在 batch 内优化排序，但可能导致：
  - **Patch 预测的不一致性增加**：为了满足 batch 内的排序约束，模型可能对同一图像的不同 patch 做出不一致的预测
  - **泛化能力下降**：模型记住了训练数据的 patch 级别模式，但无法很好地泛化到测试集
- 结果：
  - ✅ Patch 级别的 Loss 继续下降（过拟合到训练数据的 patch 模式）
  - ✅ Patch 级别的训练 SRCC 继续上升（过拟合到训练数据的 patch 分布）
  - ❌ 图像级别的测试 SRCC/PLCC 下降（泛化能力下降）

### 4. **Ranking Loss 放大过拟合**

当 `alpha` 从 0.5 增加到 1.0 时，Ranking Loss 的权重增加：

#### **Alpha=0.5**：
- L1 Loss 占主导，Ranking Loss 起辅助作用
- 最佳结果：Epoch 1 (Test SRCC: 0.9178)

#### **Alpha=1.0**：
- Ranking Loss 和 L1 Loss 权重相同
- Ranking Loss 更强地约束 batch 内的排序
- 可能导致：
  - 模型过度关注 batch 内的相对排序，忽略了绝对值的准确性
  - 不同 batch 之间的排序可能不一致
  - 测试时，图像级别的平均预测可能因为 patch 不一致而变差

---

## 解决方案建议

### 1. **降低 Ranking Loss 权重**
- 尝试更小的 `alpha` 值（如 0.1, 0.2）
- 让 L1 Loss 保持主导，Ranking Loss 仅作为辅助

### 2. **Early Stopping**
- 基于测试集的 SRCC 进行 early stopping
- 在测试 SRCC 不再提升时停止训练
- 当前最佳：Epoch 1 (SRCC: 0.9188, PLCC: 0.9326)

### 3. **图像级别的 Ranking Loss（高级方案）**
- 当前 Ranking Loss 在 patch 级别计算
- 可以考虑在图像级别计算 Ranking Loss：
  - 先将同一图像的所有 patch 预测值取平均
  - 然后在图像级别计算 Ranking Loss
  - 这样可以更好地与测试评估方式对齐

### 4. **正则化**
- 增加 `weight_decay` 的值
- 使用 Dropout（在 TargetNet 中，已注释掉）

### 5. **降低学习率**
- 当前学习率可能过大，导致过度优化
- 尝试更小的初始学习率

### 6. **调整学习率衰减策略**
- 当前策略：`lr = self.lr / pow(10, (t // 6))`
- 可能导致后期学习率过小，难以跳出过拟合
- 考虑使用更温和的衰减策略（如 Cosine Annealing）

---

## 当前最佳策略

基于实验结果：
1. **使用 Epoch 1 的模型**（alpha=1.0, Test SRCC: 0.9188）
2. **或降低 alpha 到 0.5**（Epoch 1, Test SRCC: 0.9178）
3. **实现 Early Stopping**，防止后续 epoch 的过拟合

---

## 总结

这个问题的本质是：
- **训练目标**（Patch 级别的 Loss）与**评估指标**（图像级别的 SRCC/PLCC）之间存在**尺度不一致**
- **Ranking Loss** 进一步放大了这种不一致，因为它专注于 batch 内的排序，可能导致 patch 级别的不一致性
- **过拟合**发生在 patch 级别，但影响在图像级别的评估中表现出来

因此，虽然 Loss 在下降，但测试指标在下降，这是正常的过拟合现象。

