# 训练结果对比分析

## 参数对比

### 之前的训练（ResNet-50 版本）

| 参数 | 值 |
|------|-----|
| Backbone | ResNet-50 |
| Epochs | 10 |
| Train/Test Rounds | 2 |
| Batch Size | 96 |
| Train Patch Num | 20 |
| Test Patch Num | 20 |
| **Epoch 1 结果** | **SRCC: 0.9009, PLCC: 0.9170** ✅ |

### 本次训练（Swin Transformer 版本）

| 参数 | 值 |
|------|-----|
| Backbone | Swin Transformer Tiny |
| Epochs | 1 ⚠️ |
| Train/Test Rounds | 1 |
| Batch Size | 8 ⚠️ |
| Train Patch Num | 2 ⚠️ |
| Test Patch Num | 2 ⚠️ |
| **结果** | **SRCC: 0.8860, PLCC: 0.9007** |

## 性能差异分析

| 指标 | 之前 (ResNet Epoch 1) | 本次 (Swin Epoch 1) | 差异 |
|------|---------------------|-------------------|------|
| **SRCC** | 0.9009 | 0.8860 | -0.0149 (-1.65%) |
| **PLCC** | 0.9170 | 0.9007 | -0.0163 (-1.78%) |

## 主要问题原因

### ⚠️ 1. 训练轮数不足

**问题**：只训练了 1 个 epoch
- 模型还未充分学习
- 特别是 Swin Transformer 作为新 backbone，需要更多训练来适应任务

**建议**：
- 至少训练 6-10 个 epoch
- Swin Transformer 可能需要更多 epoch 才能收敛

### ⚠️ 2. Batch Size 太小

**问题**：batch_size = 8 vs 之前的 96
- **梯度估计不稳定**：小 batch 导致梯度噪声大，训练不稳定
- **收敛速度慢**：需要更多迭代才能达到相同效果
- **内存利用不充分**：GPU/MPS 资源未充分利用

**影响**：
- 训练损失较高（7.145 vs 正常情况应该更低）
- 性能下降

**建议**：
- 至少使用 batch_size = 32 以上
- 理想情况下使用 96（与原始设置一致）

### ⚠️ 3. Patch 采样数量太少

**问题**：train_patch_num = 2, test_patch_num = 2 vs 之前的 20

**训练阶段影响**：
- **特征提取不充分**：每个图像只采样 2 个 patch，无法覆盖图像的所有区域
- **数据增强不足**：patch 多样性不够，模型泛化能力差
- **局部失真检测能力弱**：无法充分学习局部质量特征

**测试阶段影响**：
- **评估不稳定**：只采样 2 个 patch，结果方差大
- **无法全面评估**：可能错过某些区域的失真

**建议**：
- train_patch_num ≥ 10（快速测试）或 20-25（标准训练）
- test_patch_num ≥ 20-25（保证评估稳定性）

### ⚠️ 4. Swin Transformer 的适配

**潜在因素**：
- Swin Transformer 是 Transformer 架构，可能需要不同的学习率或更多训练
- 预训练权重来自 ImageNet，需要微调适应 IQA 任务
- Transformer 的全局建模能力需要足够的数据和训练才能体现优势

## 参数设置建议

### 快速测试（验证代码正确性）

```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 6 \
  --train_test_num 1 \
  --batch_size 32 \
  --train_patch_num 10 \
  --test_patch_num 25
```

### 标准训练（性能对比）

```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 10 \
  --train_test_num 2 \
  --batch_size 96 \
  --train_patch_num 25 \
  --test_patch_num 25
```

### 完整训练（论文配置）

```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 16 \
  --train_test_num 3 \
  --batch_size 96 \
  --train_patch_num 25 \
  --test_patch_num 25
```

## 预期改进

如果使用标准参数设置，预期：

1. **Epoch 1 性能**：SRCC ≥ 0.89, PLCC ≥ 0.91
2. **训练稳定后**（Epoch 6-10）：SRCC ≥ 0.90, PLCC ≥ 0.915
3. **充分训练后**（Epoch 16）：可能达到或超过 ResNet 版本性能

## 结论

**当前结果偏低的根本原因**：
1. ✅ **主要是超参数设置问题**，而非实现错误
2. ✅ **batch_size 和 patch_num 太小**导致训练不充分
3. ✅ **训练轮数太少**，模型未收敛

**建议**：
- 使用推荐的参数重新训练
- 至少训练 6 个 epoch 后再评估性能
- 使用标准参数（batch_size=96, patch_num=25）进行公平对比

