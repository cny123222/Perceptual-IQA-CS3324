# 跨数据集测试指南 (Cross-Dataset Testing Guide)

## 简介

`cross_dataset_test.py` 是一个专门用于评估训练好的IQA模型在多个数据集上性能的脚本。

### 支持的数据集

1. **KonIQ-10k Test Set** - 在KonIQ-10k训练集上训练，在测试集上评估
2. **SPAQ** - 跨数据集泛化能力测试
3. **KADID-10K** - 跨数据集泛化能力测试
4. **AGIQA-3K** - AI生成图像质量评估

## 核心特性

✅ **与训练时完全一致的评估方法**
- 使用相同的 SRCC (Spearman Rank Correlation) 计算方法
- 使用相同的 PLCC (Pearson Linear Correlation) 计算方法
- 使用相同的图像预处理和patch策略

✅ **支持不同模型大小**
- Swin-Tiny (~28M 参数)
- Swin-Small (~50M 参数)
- Swin-Base (~88M 参数)

✅ **灵活的测试选项**
- 可选择测试特定数据集或全部数据集
- 可选择CenterCrop（可复现）或RandomCrop（原论文方法）
- 可自定义patch数量

## 基本用法

### 1. 测试所有数据集（推荐）

```bash
python cross_dataset_test.py \
  --checkpoint checkpoints/koniq-10k-swin_20251218_232111/best_model_srcc_0.9236_plcc_0.9406.pkl \
  --model_size tiny \
  --test_patch_num 20
```

### 2. 测试特定数据集

```bash
# 只测试KonIQ和SPAQ
python cross_dataset_test.py \
  --checkpoint checkpoints/best_model.pkl \
  --model_size small \
  --datasets koniq spaq
```

### 3. 使用RandomCrop（与原论文一致，但结果不可复现）

```bash
python cross_dataset_test.py \
  --checkpoint checkpoints/best_model.pkl \
  --model_size tiny \
  --test_random_crop
```

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|-----|------|------|
| `--checkpoint` | checkpoint文件路径 | `checkpoints/best_model.pkl` |
| `--model_size` | 模型大小 (tiny/small/base) | `tiny` |

### 可选参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--datasets` | `koniq spaq kadid agiqa` | 要测试的数据集列表 |
| `--test_patch_num` | `20` | 每张图像的patch数量 |
| `--test_random_crop` | `False` | 是否使用RandomCrop（默认CenterCrop） |
| `--base_dir` | 脚本所在目录 | 数据集基础目录 |

## 如何找到最佳checkpoint

### 方法1：查看训练日志

```bash
# 查看某次训练的最佳结果
grep "best model saved" logs/swin_multiscale_ranking_alpha0_20251218_232111.log
```

### 方法2：查看checkpoint目录

```bash
# 列出所有checkpoint目录
ls -la checkpoints/

# 查看特定目录中的模型文件（按SRCC排序）
ls -lt checkpoints/koniq-10k-swin_20251218_232111/
```

### 方法3：使用最新的最佳模型

```bash
# 找到最新的checkpoint目录
LATEST_CHECKPOINT=$(ls -t checkpoints/ | head -1)

# 找到该目录中SRCC最高的模型
BEST_MODEL=$(ls checkpoints/$LATEST_CHECKPOINT/best_model_srcc_*.pkl | sort -V | tail -1)

echo "Best checkpoint: $BEST_MODEL"

# 运行测试
python cross_dataset_test.py \
  --checkpoint "$BEST_MODEL" \
  --model_size tiny
```

## 输出说明

### 1. 屏幕输出

脚本会显示：
- 模型加载信息
- 每个数据集的测试进度
- 每个数据集的详细结果
- 最终汇总表格
- 平均性能指标

示例输出：

```
============================================================
FINAL RESULTS SUMMARY
============================================================
Model: Swin-Tiny
Checkpoint: best_model_srcc_0.9236_plcc_0.9406.pkl
Test patch num: 20
------------------------------------------------------------
Dataset              Images     SRCC       PLCC      
------------------------------------------------------------
KonIQ-10k Test Set   2010       0.9236     0.9406    
SPAQ                 11125      0.8876     0.9012    
KADID-10K            10125      0.8456     0.8678    
AGIQA-3K             2982       0.7823     0.8134    
============================================================

Average across 4 datasets:
  Average SRCC: 0.8598
  Average PLCC: 0.8808
============================================================
```

### 2. JSON结果文件

脚本会自动保存结果到JSON文件：

**文件名格式**：`cross_dataset_results_{model_size}_{checkpoint_name}.json`

**示例**：`cross_dataset_results_tiny_best_model_srcc_0.9236_plcc_0.9406.json`

**JSON内容**：

```json
{
  "checkpoint": "checkpoints/best_model.pkl",
  "model_size": "tiny",
  "test_patch_num": 20,
  "test_random_crop": false,
  "results": {
    "KonIQ-10k Test Set": {
      "srcc": 0.9236,
      "plcc": 0.9406,
      "num_images": 2010
    },
    "SPAQ": {
      "srcc": 0.8876,
      "plcc": 0.9012,
      "num_images": 11125
    },
    ...
  }
}
```

## 常见问题

### Q1: 为什么推荐使用CenterCrop而不是RandomCrop？

**A**: CenterCrop的结果是**确定性的**、**可复现的**，适合：
- 论文撰写（需要固定的数值）
- 模型对比（确保公平比较）
- 调试和验证

RandomCrop虽然是原论文的方法，但每次运行结果会略有不同（±0.001左右）。

### Q2: test_patch_num应该设置多少？

**A**: 建议与训练时保持一致：
- 训练时用的20 → 测试时也用20
- 训练时用的25 → 测试时也用25

更多的patch会：
- ✅ 提高预测稳定性
- ✅ 略微提高SRCC/PLCC
- ❌ 增加测试时间

### Q3: 如何解释跨数据集的性能下降？

**A**: 跨数据集性能下降是正常的：

| 训练集 | 测试集 | 典型性能 | 原因 |
|-------|--------|---------|------|
| KonIQ | KonIQ | 0.92-0.93 | 同分布 |
| KonIQ | SPAQ | 0.87-0.90 | 不同相机、场景 |
| KonIQ | KADID | 0.84-0.87 | 合成失真 vs 自然失真 |
| KonIQ | AGIQA | 0.78-0.82 | AI生成图像 vs 自然图像 |

**降低性能下降的方法**：
1. 使用更大的模型（Tiny → Small → Base）
2. 在多个数据集上联合训练
3. 使用数据增强提高泛化能力

### Q4: 如何对比不同配置的模型？

**A**: 创建一个批处理脚本：

```bash
#!/bin/bash
# compare_models.sh

# Config 1: Swin-Tiny baseline
python cross_dataset_test.py \
  --checkpoint checkpoints/config1/best_model.pkl \
  --model_size tiny \
  --test_patch_num 20

# Config 2: Swin-Small
python cross_dataset_test.py \
  --checkpoint checkpoints/config2/best_model.pkl \
  --model_size small \
  --test_patch_num 20

# Config 3: Swin-Small + Ranking Loss
python cross_dataset_test.py \
  --checkpoint checkpoints/config3/best_model.pkl \
  --model_size small \
  --test_patch_num 20
```

运行：`bash compare_models.sh`

然后对比生成的JSON文件。

## 实际应用示例

### 示例1：测试最新的Swin-Tiny模型

```bash
# Config 1 (最佳Swin-Tiny配置)
python cross_dataset_test.py \
  --checkpoint checkpoints/koniq-10k-swin_20251218_232111/best_model_srcc_0.9236_plcc_0.9406.pkl \
  --model_size tiny \
  --test_patch_num 20
```

### 示例2：测试Swin-Small模型（Ranking Loss alpha=0.5）

```bash
# 假设你的Swin-Small模型在这个目录
SWIN_SMALL_DIR="checkpoints/koniq-10k-swin-ranking-alpha0.5_20251219_195314"
BEST_SMALL=$(ls $SWIN_SMALL_DIR/best_model_srcc_*.pkl | sort -V | tail -1)

python cross_dataset_test.py \
  --checkpoint "$BEST_SMALL" \
  --model_size small \
  --test_patch_num 20
```

### 示例3：只测试SPAQ和KADID（快速验证）

```bash
python cross_dataset_test.py \
  --checkpoint checkpoints/best_model.pkl \
  --model_size tiny \
  --datasets spaq kadid \
  --test_patch_num 20
```

## 性能参考

根据我们的实验，以下是典型的性能范围：

### Swin-Tiny (~28M 参数)

| 数据集 | SRCC | PLCC |
|-------|------|------|
| KonIQ-10k | 0.920-0.924 | 0.938-0.942 |
| SPAQ | 0.880-0.895 | 0.895-0.905 |
| KADID-10K | 0.840-0.860 | 0.860-0.875 |
| AGIQA-3K | 0.775-0.795 | 0.805-0.820 |

### Swin-Small (~50M 参数)

| 数据集 | SRCC | PLCC |
|-------|------|------|
| KonIQ-10k | 0.928-0.932 | 0.942-0.946 |
| SPAQ | 0.890-0.905 | 0.905-0.915 |
| KADID-10K | 0.850-0.870 | 0.870-0.885 |
| AGIQA-3K | 0.785-0.805 | 0.815-0.830 |

**注意**：这些只是估计值，实际结果取决于训练配置、数据增强、正则化等因素。

## 与训练代码的一致性保证

本脚本确保与 `HyperIQASolver_swin.py` 中的测试方法**完全一致**：

1. ✅ **相同的数据预处理**：
   - Resize到(512, 384)
   - CenterCrop/RandomCrop到224x224
   - 相同的Normalize参数

2. ✅ **相同的patch策略**：
   - 每张图像生成 `test_patch_num` 个patch
   - 预测时对所有patch的结果取平均

3. ✅ **相同的相关系数计算**：
   - 使用 `scipy.stats.spearmanr()` 计算SRCC
   - 使用 `scipy.stats.pearsonr()` 计算PLCC
   - 先reshape后averaging，再计算相关系数

4. ✅ **相同的模型配置**：
   - 使用 `models_swin.HyperNet` 和 `models_swin.TargetNet`
   - 相同的multi-scale fusion、dropout、drop_path配置

## 故障排除

### 错误1：FileNotFoundError: JSON file not found

**原因**：数据集路径不正确或JSON文件缺失

**解决**：
```bash
# 检查数据集是否存在
ls -la koniq-10k/koniq_test.json
ls -la spaq-test/spaq_test.json
ls -la kadid-test/kadid_test.json
ls -la agiqa-test/agiqa_test.json

# 如果缺失，检查符号链接
ls -la | grep test
```

### 错误2：RuntimeError: Error(s) in loading state_dict

**原因**：checkpoint与model_size不匹配

**解决**：确认checkpoint是用哪个模型大小训练的：
```bash
# 查看训练日志
grep "Loading Swin Transformer" logs/YOUR_LOG_FILE.log
# 会显示 TINY/SMALL/BASE

# 然后使用对应的 --model_size
```

### 错误3：CUDA out of memory

**原因**：显存不足

**解决**：
```bash
# 方法1：减少batch_size（脚本默认是1，已经最小）
# 方法2：减少test_patch_num
python cross_dataset_test.py ... --test_patch_num 10

# 方法3：使用CPU（会很慢）
export CUDA_VISIBLE_DEVICES=""
python cross_dataset_test.py ...
```

## 更多帮助

查看完整的参数列表：

```bash
python cross_dataset_test.py --help
```

## 相关文件

- 训练脚本：`train_swin.py`
- 模型定义：`models_swin.py`
- 数据加载器：`data_loader.py`
- Solver：`HyperIQASolver_swin.py`
- 实验记录：`record.md`
- 第一阶段总结：`PHASE1_SUMMARY.md`

