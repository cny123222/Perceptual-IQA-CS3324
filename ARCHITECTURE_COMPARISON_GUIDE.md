# 架构对比实验指南

本指南帮助您运行和对比三个不同的架构，**全部在 ranking-loss 分支运行**：
1. **ResNet-50** (原始架构)
2. **Swin Transformer** (无 Ranking Loss)
3. **Swin Transformer + Ranking Loss**

---

## 重要说明

✅ **所有实验都在 ranking-loss 分支运行，无需切换分支**

✅ **所有实验都包含以下功能：**
- 每个epoch保存checkpoint（带时间戳的文件夹名，防止覆盖）
- 在SPAQ数据集上进行跨数据集测试（如果数据集存在）
- 训练稳定性修复：
  - Filter iterator exhaustion bug 修复
  - Backbone learning rate decay 修复
  - Optimizer state preservation 修复

---

## 方法一：自动运行所有实验（推荐）

使用统一脚本自动运行所有三个实验并生成对比报告：

```bash
chmod +x run_architecture_comparison.sh
./run_architecture_comparison.sh
```

**功能：**
- 自动依次运行三个实验
- 自动收集结果
- 生成对比报告
- 所有实验在同一分支（ranking-loss）

**结果保存位置：**
- 对比报告: `comparison_results/comparison_YYYYMMDD_HHMMSS.txt`
- 详细日志: `comparison_results/*_YYYYMMDD_HHMMSS.log`

---

## 方法二：分别运行单个实验

如果您想分别运行每个实验，可以使用：

```bash
chmod +x run_single_architecture.sh

# 运行 ResNet-50
./run_single_architecture.sh resnet

# 运行 Swin Transformer (无 Ranking Loss)
./run_single_architecture.sh swin

# 运行 Swin Transformer + Ranking Loss
./run_single_architecture.sh swin-ranking
```

---

## 方法三：手动运行（完全控制）

### 实验1: ResNet-50

```bash
# 确保在 ranking-loss 分支
git checkout ranking-loss

python train_test_IQA.py \
    --dataset koniq-10k \
    --epochs 10 \
    --train_test_num 1 \
    --batch_size 96 \
    --train_patch_num 20 \
    --test_patch_num 20
```

**特点：**
- 使用 `train_test_IQA.py`
- 使用 `HyerIQASolver.py` (ResNet backbone)
- Checkpoint保存在: `checkpoints/koniq-10k-resnet_TIMESTAMP/`

### 实验2: Swin Transformer

```bash
# 确保在 ranking-loss 分支
git checkout ranking-loss

python train_swin.py \
    --dataset koniq-10k \
    --epochs 10 \
    --train_test_num 1 \
    --batch_size 96 \
    --train_patch_num 20 \
    --test_patch_num 20 \
    --ranking_loss_alpha 0
```

**特点：**
- 使用 `train_swin.py`
- 使用 `HyperIQASolver_swin.py` (Swin backbone)
- `--ranking_loss_alpha 0` 禁用 Ranking Loss
- Checkpoint保存在: `checkpoints/koniq-10k-swin_TIMESTAMP/`

### 实验3: Swin Transformer + Ranking Loss

```bash
# 确保在 ranking-loss 分支
git checkout ranking-loss

python train_swin.py \
    --dataset koniq-10k \
    --epochs 10 \
    --train_test_num 1 \
    --batch_size 96 \
    --train_patch_num 20 \
    --test_patch_num 20 \
    --ranking_loss_alpha 0.3 \
    --ranking_loss_margin 0.1
```

**特点：**
- 使用 `train_swin.py`
- 使用 `HyperIQASolver_swin.py` (Swin backbone)
- `--ranking_loss_alpha 0.3` 启用 Ranking Loss
- Checkpoint保存在: `checkpoints/koniq-10k-swin-ranking-alpha0.3_TIMESTAMP/`

---

## 参数说明

所有实验使用相同的参数以确保公平对比：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--dataset` | `koniq-10k` | 数据集 |
| `--epochs` | `10` | 训练轮数 |
| `--train_test_num` | `1` | 训练轮次 |
| `--batch_size` | `96` | 批次大小 |
| `--train_patch_num` | `20` | 训练时每张图片的patch数 |
| `--test_patch_num` | `20` | 测试时每张图片的patch数 |
| `--ranking_loss_alpha` | `0` (实验2) / `0.3` (实验3) | Ranking Loss权重 |
| `--ranking_loss_margin` | `0.1` (仅实验3) | Ranking Loss边界 |

---

## 三个架构对比

| 架构 | 训练脚本 | Solver | Backbone | Ranking Loss | Checkpoint目录 |
|------|----------|--------|----------|--------------|----------------|
| ResNet-50 | `train_test_IQA.py` | `HyerIQASolver.py` | ResNet-50 | ❌ | `koniq-10k-resnet_TIMESTAMP` |
| Swin Transformer | `train_swin.py` | `HyperIQASolver_swin.py` | Swin Tiny | ❌ | `koniq-10k-swin_TIMESTAMP` |
| Swin + Ranking Loss | `train_swin.py` | `HyperIQASolver_swin.py` | Swin Tiny | ✅ (α=0.3) | `koniq-10k-swin-ranking-alpha0.3_TIMESTAMP` |

---

## 结果对比

### 查看自动生成的对比报告

```bash
cat comparison_results/comparison_*.txt
```

### 手动对比关键指标

从训练输出中提取最佳结果：

**ResNet-50:**
- 查找: `Best test SRCC`
- 查看: KonIQ-10k Test SRCC/PLCC, SPAQ SRCC/PLCC

**Swin Transformer:**
- 查找: `Best test SRCC`
- 查看: KonIQ-10k Test SRCC/PLCC, SPAQ SRCC/PLCC

**Swin Transformer + Ranking Loss:**
- 查找: `Best test SRCC`
- 查看: KonIQ-10k Test SRCC/PLCC, SPAQ SRCC/PLCC

### 对比表格模板

| 架构 | 最佳Epoch | KonIQ-10k Test SRCC | KonIQ-10k Test PLCC | SPAQ SRCC | SPAQ PLCC | 论文基准 | 超出幅度 |
|------|-----------|---------------------|---------------------|-----------|-----------|----------|----------|
| ResNet-50 | ? | ? | ? | ? | ? | 0.906 / 0.917 | ? |
| Swin Transformer | ? | ? | ? | ? | ? | 0.906 / 0.917 | ? |
| Swin + Ranking Loss | ? | ? | ? | ? | ? | 0.906 / 0.917 | ? |

---

## Checkpoint和测试

### Checkpoint保存

- **每个epoch都保存**（不再是每2个epoch）
- **文件夹名包含时间戳**（防止覆盖）
- **文件名包含SRCC和PLCC**（便于识别最佳模型）
- **如果SPAQ测试可用，文件名也包含SPAQ指标**

示例文件名：
```
checkpoint_epoch_10_srcc_0.9206_plcc_0.9334_spaq_srcc_0.8234_spaq_plcc_0.8456.pkl
```

### SPAQ跨数据集测试

- **自动检测**：如果 `spaq-test/spaq_test.json` 存在，自动进行SPAQ测试
- **每个epoch后测试**：训练完每个epoch后自动在SPAQ上测试
- **结果记录**：SPAQ的SRCC和PLCC会显示在训练输出和checkpoint文件名中

---

## 注意事项

1. **分支**: 所有实验都在 `ranking-loss` 分支运行，无需切换
2. **训练时间**: 每个实验大约需要数小时，请确保有足够时间
3. **磁盘空间**: 每个实验会生成10个checkpoint（每个epoch一个），确保有足够空间
4. **GPU内存**: 如果GPU内存不足，可以减小 `--batch_size`
5. **结果保存**: 所有checkpoint保存在 `checkpoints/` 目录下，带时间戳的文件夹
6. **SPAQ数据集**: 如果SPAQ测试数据集不存在，实验仍会正常运行，只是不会报告SPAQ指标

---

## 快速测试（减少训练时间）

如果想快速测试脚本是否正常工作，可以修改参数：

```bash
# 编辑 run_architecture_comparison.sh
# 修改为：
EPOCHS=2
TRAIN_PATCH_NUM=10
TEST_PATCH_NUM=10
```

---

## 故障排除

### 问题1: 训练脚本不存在

确保在 ranking-loss 分支有以下文件：
- `train_test_IQA.py` (ResNet)
- `train_swin.py` (Swin)
- `HyerIQASolver.py` (ResNet solver)
- `HyperIQASolver_swin.py` (Swin solver)

### 问题2: 依赖问题

确保所有依赖都已安装：
```bash
pip install -r requirements.txt
```

### 问题3: SPAQ测试失败

SPAQ测试是可选的。如果 `spaq-test/spaq_test.json` 不存在，实验仍会正常运行，只是不会报告SPAQ指标。

---

## 推荐工作流程

1. **首次运行**: 使用方法一（自动运行所有实验）
2. **结果分析**: 查看对比报告，找出最佳架构
3. **深入实验**: 对最佳架构进行更详细的超参数调优
4. **记录结果**: 更新 `record.md` 文件

---

## 训练稳定性修复

所有实验都包含以下三个关键修复，确保训练稳定性：

1. **Filter Iterator Exhaustion Bug**: 修复了 `filter()` 对象被多次使用导致的参数列表为空的问题
2. **Backbone Learning Rate Decay**: Backbone的学习率现在也会按计划衰减
3. **Optimizer State Preservation**: Adam优化器的momentum状态在epoch间得到保留，不再每epoch重新创建

这些修复确保了：
- 训练过程更加稳定
- 测试指标（SRCC/PLCC）不会在早期达到峰值后下降
- 模型能够持续改进而不是过早过拟合
