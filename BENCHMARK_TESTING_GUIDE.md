# Benchmark Testing Guide

本指南说明如何测试各个baseline模型在四个数据集上的性能。

## 测试数据集

所有模型将在以下4个数据集上进行跨数据集测试：
- **KonIQ-10k**: 自然场景图像质量评估
- **SPAQ**: 智能手机拍摄图像质量评估  
- **KADID-10K**: 人工失真图像质量评估
- **AGIQA-3K**: AI生成图像质量评估

## 1. HyperIQA Baseline

HyperIQA是我们改进工作的基础模型，使用ResNet-50作为backbone。

### 预训练模型
- 路径: `pretrained/koniq_pretrained.pkl`
- 训练数据集: KonIQ-10k
- Backbone: ResNet-50

### 运行测试

```bash
cd /root/Perceptual-IQA-CS3324
python test_pretrained_correct.py
```

### 测试配置
- **Patch数量**: 25 (每张图像随机crop 25次)
- **Transform**: Resize(512, 384) → RandomCrop(224) → Normalize
- **输出**: 
  - 控制台输出详细结果
  - `logs/pretrained_correct_results.txt` - 文本格式结果

### 预期运行时间
- 约 20-30 分钟（取决于GPU性能）

---

## 2. StairIQA Baseline

StairIQA使用"阶梯式"设计，为不同数据集训练独立的输出头。

### 预训练模型
- 路径: `benchmarks/StairIQA/pretrained/ResNet_staircase_50-EXP1-Koniq10k.pkl`
- 训练数据集: KonIQ-10k
- Backbone: ResNet-50 with Staircase design
- 模型大小: 122MB

### 运行测试

```bash
cd /root/Perceptual-IQA-CS3324
python test_stairiqa_pretrained.py
```

### 测试配置
- **测试方法**: FiveCrop (5个crops取平均)
- **Transform**: Resize(384) → FiveCrop(320) → Normalize
- **输出头**: 使用Koniq10k的输出头（索引3）进行跨数据集测试
- **输出**:
  - 控制台输出详细结果
  - `logs/stairiqa_pretrained_results.txt` - 文本格式结果
  - `logs/stairiqa_pretrained_results.json` - JSON格式结果

### 预期运行时间
- 约 15-25 分钟（FiveCrop比随机crop略快）

---

## 3. DBCNN Baseline

DBCNN (Deep Bilinear CNN) 使用双线性卷积网络。

### 预训练模型
- 路径: `benchmarks/DBCNN/dbcnn/models/`
- 格式: MATLAB .mat文件
- **注意**: DBCNN需要MATLAB环境运行

### 运行测试

```bash
cd /root/Perceptual-IQA-CS3324/benchmarks/DBCNN/dbcnn
# 需要在MATLAB中运行
matlab -nodisplay -nosplash -r "run_exp"
```

---

## 评估指标

所有模型使用以下两个指标进行评估：

### SRCC (Spearman Rank Correlation Coefficient)
- 范围: [-1, 1]
- 衡量预测质量分数与真实分数的**排序相关性**
- 更高表示更好

### PLCC (Pearson Linear Correlation Coefficient)  
- 范围: [-1, 1]
- 衡量预测质量分数与真实分数的**线性相关性**
- 更高表示更好

---

## 结果汇总

测试完成后，所有结果将保存在 `logs/` 目录下：

```
logs/
├── pretrained_correct_results.txt      # HyperIQA结果
├── stairiqa_pretrained_results.txt     # StairIQA结果（文本）
└── stairiqa_pretrained_results.json    # StairIQA结果（JSON）
```

---

## 注意事项

### 1. Git追踪
预训练模型文件已在`.gitignore`中排除，不会被提交：
- `pretrained/`
- `benchmarks/StairIQA/pretrained/`
- `benchmarks/DBCNN/dbcnn/models/`

### 2. 数据盘符号链接
数据集文件夹已链接到数据盘，确保有足够空间：
- `agiqa-test` → `/root/autodl-tmp/Perceptual-IQA-CS3324-data/datasets/agiqa-test`
- `kadid-test` → `/root/autodl-tmp/Perceptual-IQA-CS3324-data/datasets/kadid-test`
- `koniq-10k` → `/root/autodl-tmp/Perceptual-IQA-CS3324-data/datasets/koniq-10k`
- `koniq-test` → `/root/autodl-tmp/Perceptual-IQA-CS3324-data/datasets/koniq-test`
- `spaq-test` → `/root/autodl-tmp/Perceptual-IQA-CS3324-data/datasets/spaq-test`

### 3. 随机种子
所有测试脚本都设置了固定随机种子（42），确保结果可复现。

### 4. GPU要求
- 建议使用CUDA GPU进行测试
- 最小显存要求: 4GB
- 推荐显存: 8GB+

---

## 快速对比测试

如果想快速对比所有baseline模型，可以依次运行：

```bash
# 1. HyperIQA
python test_pretrained_correct.py

# 2. StairIQA  
python test_stairiqa_pretrained.py

# 3. 查看结果
cat logs/pretrained_correct_results.txt
cat logs/stairiqa_pretrained_results.txt
```

---

## 故障排除

### 问题: 找不到模型文件
**解决**: 确认预训练模型已正确放置在指定路径

### 问题: CUDA out of memory
**解决**: 
- 减少batch size（已设为1）
- 使用更小的模型
- 清理GPU缓存: `torch.cuda.empty_cache()`

### 问题: 数据集路径错误
**解决**: 检查符号链接是否正确创建:
```bash
ls -la | grep test
```

---

## 更新日志

- **2024-12-22**: 创建测试指南，添加HyperIQA和StairIQA测试脚本

