# 架构对比实验指南

本指南帮助您运行和对比三个不同的架构：
1. **ResNet-50** (原始架构)
2. **Swin Transformer** (无 Ranking Loss)
3. **Swin Transformer + Ranking Loss**

---

## 方法一：自动运行所有实验（推荐）

使用统一脚本自动运行所有三个实验并生成对比报告：

```bash
chmod +x run_architecture_comparison.sh
./run_architecture_comparison.sh
```

**功能：**
- 自动切换分支
- 依次运行三个实验
- 自动收集结果
- 生成对比报告
- 恢复原始分支

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

# 运行 Swin Transformer
./run_single_architecture.sh swin

# 运行 Swin Transformer + Ranking Loss
./run_single_architecture.sh swin-ranking
```

---

## 方法三：手动运行（完全控制）

### 实验1: ResNet-50

```bash
git checkout master
python train_test_IQA.py \
    --dataset koniq-10k \
    --epochs 10 \
    --train_test_num 1 \
    --batch_size 96 \
    --train_patch_num 20 \
    --test_patch_num 20
```

### 实验2: Swin Transformer

```bash
git checkout swin-transformer-backbone
python train_swin.py \
    --dataset koniq-10k \
    --epochs 10 \
    --train_test_num 1 \
    --batch_size 96 \
    --train_patch_num 20 \
    --test_patch_num 20 \
    --ranking_loss_alpha 0
```

### 实验3: Swin Transformer + Ranking Loss

```bash
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
| `--ranking_loss_alpha` | `0.3` | Ranking Loss权重（仅实验3） |
| `--ranking_loss_margin` | `0.1` | Ranking Loss边界（仅实验3） |

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

**Swin Transformer:**
- 查找: `Best test SRCC`

**Swin Transformer + Ranking Loss:**
- 查找: `Best test SRCC`

### 对比表格模板

| 架构 | 最佳Epoch | Test SRCC | Test PLCC | 论文基准 | 超出幅度 |
|------|-----------|-----------|-----------|----------|----------|
| ResNet-50 | ? | ? | ? | 0.906 / 0.917 | ? |
| Swin Transformer | ? | ? | ? | 0.906 / 0.917 | ? |
| Swin Transformer + Ranking Loss | ? | ? | ? | 0.906 / 0.917 | ? |

---

## 注意事项

1. **分支切换**: 脚本会自动切换分支，但请确保所有分支都已拉取最新代码
2. **训练时间**: 每个实验大约需要数小时，请确保有足够时间
3. **磁盘空间**: 每个实验会生成checkpoint，确保有足够空间
4. **GPU内存**: 如果GPU内存不足，可以减小 `--batch_size`
5. **结果保存**: 所有checkpoint保存在 `checkpoints/` 目录下，带时间戳的文件夹

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

### 问题1: 分支不存在
```bash
# 确保所有分支都已拉取
git fetch origin
git checkout -b master origin/master
git checkout -b swin-transformer-backbone origin/swin-transformer-backbone
git checkout -b ranking-loss origin/ranking-loss
```

### 问题2: 训练脚本不存在
确保在正确的分支上有对应的训练脚本：
- `master`: `train_test_IQA.py`
- `swin-transformer-backbone`: `train_swin.py`
- `ranking-loss`: `train_swin.py`

### 问题3: 依赖问题
确保所有分支的依赖都已安装：
```bash
pip install -r requirements.txt
```

---

## 推荐工作流程

1. **首次运行**: 使用方法一（自动运行所有实验）
2. **结果分析**: 查看对比报告，找出最佳架构
3. **深入实验**: 对最佳架构进行更详细的超参数调优
4. **记录结果**: 更新 `record.md` 文件

