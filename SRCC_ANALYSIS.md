# Training Set SRCC vs Test Set SRCC Analysis

## 问题描述

训练时发现测试集的 SRCC 明显高于训练集的 SRCC，这是什么原因？

## 原作者的实现方式（commit c42e727）

### 训练集 SRCC 计算（原始代码）
```python
# 在 train() 函数中，第 49 行
train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
# 直接在 patch 级别计算，没有先对图像取平均
```

### 测试集 SRCC 计算（原始代码）
```python
# 在 test() 函数中
pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
# 先对每个图像的 patches 取平均，再在图像级别计算
```

## 结论

**这是原作者的设计，不是 bug！**

### 为什么会这样设计？

1. **训练集：Patch 级别计算**
   - 每个 epoch 中，训练集从每张图像采样 `train_patch_num` 个随机 patches
   - 直接在所有 patches 上计算 SRCC，反映模型对 patch 级别的排序能力
   - Patch 级别的预测方差较大，所以 SRCC 会偏低

2. **测试集：图像级别计算**
   - 对每张图像采样 `test_patch_num` 个 patches，然后取平均得到图像分数
   - 在图像级别计算 SRCC，这是最终评估指标
   - 取平均后预测更稳定，所以 SRCC 更高

### 为什么测试集 SRCC 会高于训练集？

1. **计算尺度不同**
   - 训练集：Patch 级别（方差大，噪声多）
   - 测试集：图像级别（经过平均，更稳定）

2. **模型状态不同**
   - 训练集：`model.train(True)` - 启用 dropout、batch norm 训练模式（随机性）
   - 测试集：`model.train(False)` - 评估模式（确定性）

3. **数据增强**
   - 训练集：使用 RandomCrop、RandomHorizontalFlip 等增强（引入随机性）
   - 测试集：只使用 RandomCrop（相对更稳定）

4. **这是正常现象**
   - 原作者的设计就是这样的
   - 测试集 SRCC 是真正的评估指标
   - 训练集 SRCC 只是一个参考，用于监控训练趋势

## 我们的实现

我们的代码与原作者完全一致：
- `HyperIQASolver_swin.py` 第 108 行：训练集在 patch 级别计算
- `HyperIQASolver_swin.py` 第 163-166 行：测试集在图像级别计算

**无需修改！这是正确的实现。**

## 验证

查看原始代码（commit c42e727）：
```bash
git show c42e727:HyerIQASolver.py
```

可以看到：
- 训练集：直接 `stats.spearmanr(pred_scores, gt_scores)` （patch 级别）
- 测试集：先 `np.mean(reshape(...), axis=1)` 再 `stats.spearmanr(...)` （图像级别）

**完全一致！**

## 总结

✅ **测试集 SRCC 高于训练集 SRCC 是正常的、预期的行为**

✅ **我们的实现与原作者一致，没有 bug**

✅ **这是 Hyper-IQA 论文的设计，训练集和测试集使用不同的评估尺度是合理的**

**关键点：**
- 训练集 SRCC：监控训练趋势（patch 级别，偏低是正常的）
- 测试集 SRCC：真正的模型性能指标（图像级别，这是我们应该关注的）

