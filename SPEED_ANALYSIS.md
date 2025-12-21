# 测试速度分析报告

## 当前测试速度

### Swin-Base (最佳模型)
- **测试速度**: 57.78 batch/s
- **图像预加载**: 84.53 img/s (23秒加载2010张图)
- **总测试时间**: 约11分钟 (40,200 patches)

### ResNet-50 (预训练模型)
- **测试速度**: 48.15 batch/s
- **总测试时间**: 约17分钟 (50,250 patches，25 patches/img)

## 对比原始HyperIQA代码

我克隆并对比了原始HyperIQA仓库，发现：

**测试方法完全一致！**
- 都是每次循环创建新的TargetNet
- 都是batch_size=1
- 都是逐个处理patches

**唯一差异**:
```python
# 原始: 没有torch.no_grad()
for img, label in data:
    ...

# 当前: 添加了torch.no_grad()（应该更快）
with torch.no_grad():
    for img, label in data:
        ...
```

## 可能的"变慢"原因

### 1. 模型大小差异
- ResNet-50: ~25M parameters
- Swin-Base: ~88M parameters (3.5倍)
- **更大的模型 = 更慢的推理**

### 2. patch_num设置
- 如果之前用的patch_num < 20，那确实会更快
- 测试时间 ∝ patch_num

### 3. 实际没有变慢！
- Swin-Base (57.78 batch/s) 比 ResNet-50 (48.15 batch/s) 还快20%
- 可能是因为：
  - 图像预加载优化
  - GPU利用率更高

## 结论

**代码本身没有性能问题！**

测试速度由以下因素决定：
1. 模型大小（HyperNet + TargetNet）
2. patch_num（每张图测试多少次）
3. 图像数量
4. GPU性能

**如果感觉变慢了，可能是：**
- 换了更大的模型（Swin-Base）
- patch_num增加了
- 或者之前的记忆有误

## 建议

如果需要加速测试：
1. 减少patch_num（20 → 10），准确度略降但速度翻倍
2. 使用更小的模型（Swin-Tiny）
3. 使用更少的测试图像

但这些都会影响结果的准确性！
