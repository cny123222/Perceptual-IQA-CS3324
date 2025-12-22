# QualiCLIP Pre-training Experiment Plan

## 当前状态
- ✅ Phase 1: 代码实现完成
- 🔄 Phase 2: 预训练进行中 (Epoch 1/10)
- ⏳ Phase 3: 等待预训练完成
- ⏳ Phase 4: 对比实验
- ⏳ Phase 5: 结果分析

---

## 预训练完成后的操作流程

### Step 1: 验证预训练结果

```bash
# 检查训练日志
tail -100 /root/Perceptual-IQA-CS3324/logs/qualiclip_pretrain_run.log

# 检查保存的checkpoint
ls -lh /root/Perceptual-IQA-CS3324/checkpoints/qualiclip_pretrain_*/
```

**预期输出：**
- `swin_base_epoch5.pkl` (~353MB)
- `swin_base_epoch10.pkl` (~353MB)
- 训练loss应该稳定下降

---

### Step 2: 运行对比实验

#### 实验A: 基线（无预训练）

```bash
cd /root/Perceptual-IQA-CS3324

python train_swin.py \
    --database koniq10k \
    --model_name swin_base_baseline \
    --batch_size 8 \
    --epochs 30 \
    --lr 1e-4 \
    2>&1 | tee logs/train_baseline.log
```

**预计训练时间：** ~3-4小时（30 epochs）

#### 实验B: 使用QualiCLIP预训练

```bash
cd /root/Perceptual-IQA-CS3324

# 使用预训练权重
python train_swin.py \
    --database koniq10k \
    --model_name swin_base_qualiclip \
    --batch_size 8 \
    --epochs 30 \
    --lr 1e-4 \
    --pretrained_encoder checkpoints/qualiclip_pretrain_20251223_000041/swin_base_epoch10.pkl \
    --lr_encoder_pretrained 1e-5 \
    2>&1 | tee logs/train_qualiclip.log
```

**关键参数说明：**
- `--pretrained_encoder`: 预训练权重路径（修改为实际路径）
- `--lr_encoder_pretrained 1e-5`: encoder使用更小学习率（differential learning rate）
- `--lr 1e-4`: HyperNet使用正常学习率

**预计训练时间：** ~3-4小时（30 epochs）

---

### Step 3: 跨数据集泛化评估

测试在未见过的数据集上的表现：

```bash
# 测试基线模型
python test_swin.py \
    --model_path checkpoints/swin_base_baseline/best_model.pkl \
    --test_datasets spaq kadid agiqa

# 测试QualiCLIP预训练模型
python test_swin.py \
    --model_path checkpoints/swin_base_qualiclip/best_model.pkl \
    --test_datasets spaq kadid agiqa
```

---

### Step 4: 整理实验结果

#### 4.1 收集关键指标

创建结果对比表格：

| 实验组 | KonIQ-10k SRCC | KonIQ-10k PLCC | SPAQ SRCC | KADID SRCC | AGIQA SRCC |
|--------|----------------|----------------|-----------|------------|------------|
| 基线（无预训练） | ? | ? | ? | ? | ? |
| QualiCLIP预训练 | ? | ? | ? | ? | ? |
| **提升** | ? | ? | ? | ? | ? |

#### 4.2 分析改进点

重点关注：
1. **训练数据集（KonIQ-10k）表现**
   - SRCC/PLCC是否提升？
   - 收敛速度是否更快？
   - 最终性能是否更好？

2. **跨数据集泛化能力**
   - SPAQ（自然场景）
   - KADID-10K（人工失真）
   - AGIQA-3K（AI生成图像）
   - 预训练是否提高了泛化性能？

3. **训练稳定性**
   - Loss曲线是否更平滑？
   - 是否减少了过拟合？

---

## 预期效果

根据QualiCLIP论文和我们的设计，预期改进：

✅ **训练数据集（KonIQ-10k）：**
- SRCC: +0.01 ~ +0.03
- PLCC: +0.01 ~ +0.03
- 训练更稳定，收敛更快

✅ **跨数据集泛化：**
- SPAQ: +0.02 ~ +0.05
- KADID: +0.01 ~ +0.03
- AGIQA: +0.02 ~ +0.05
- 自监督学习的优势应该在跨数据集场景更明显

---

## 时间线估计

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| ✅ Phase 1 | 代码实现 | 已完成 |
| 🔄 Phase 2 | 自监督预训练 (10 epochs) | ~1.2小时 |
| ⏳ Phase 3 | 基线训练 (30 epochs) | ~3-4小时 |
| ⏳ Phase 4 | 预训练模型微调 (30 epochs) | ~3-4小时 |
| ⏳ Phase 5 | 跨数据集评估 | ~1小时 |
| ⏳ Phase 6 | 结果分析与文档 | ~1小时 |
| **总计** | | **~10-12小时** |

---

## 自动化脚本

可以创建一个脚本自动运行所有实验：

```bash
# run_qualiclip_experiments.sh
#!/bin/bash

# 等待预训练完成...

# 1. 基线实验
echo "========== Running Baseline Experiment =========="
python train_swin.py \
    --database koniq10k \
    --model_name swin_base_baseline \
    --batch_size 8 \
    --epochs 30 \
    --lr 1e-4 \
    2>&1 | tee logs/train_baseline.log

# 2. 预训练实验
echo "========== Running QualiCLIP Pre-trained Experiment =========="
PRETRAIN_PATH=$(ls -t checkpoints/qualiclip_pretrain_*/swin_base_epoch10.pkl | head -1)
python train_swin.py \
    --database koniq10k \
    --model_name swin_base_qualiclip \
    --batch_size 8 \
    --epochs 30 \
    --lr 1e-4 \
    --pretrained_encoder $PRETRAIN_PATH \
    --lr_encoder_pretrained 1e-5 \
    2>&1 | tee logs/train_qualiclip.log

# 3. 跨数据集评估
echo "========== Cross-dataset Evaluation =========="
# TODO: Add test commands

echo "All experiments completed!"
```

---

## 结果文档模板

实验完成后，更新 `QUALICLIP_RESULTS.md`：

```markdown
# QualiCLIP Self-Supervised Pre-training Results

## 实验设置
- **预训练数据集**: KonIQ-10k train split (7046 images)
- **预训练epochs**: 10
- **微调数据集**: KonIQ-10k train split
- **微调epochs**: 30
- **Backbone**: Swin-Base

## 主要结果

### KonIQ-10k测试集
| Method | SRCC | PLCC |
|--------|------|------|
| Baseline (无预训练) | X.XXX | X.XXX |
| QualiCLIP (预训练) | X.XXX | X.XXX |
| **改进** | +X.XXX | +X.XXX |

### 跨数据集泛化
| Dataset | Baseline SRCC | QualiCLIP SRCC | 改进 |
|---------|---------------|----------------|------|
| SPAQ | X.XXX | X.XXX | +X.XXX |
| KADID-10K | X.XXX | X.XXX | +X.XXX |
| AGIQA-3K | X.XXX | X.XXX | +X.XXX |

## 分析与结论
...
```

---

## 注意事项

⚠️ **重要提醒：**

1. **预训练权重路径**: 确保在运行实验B时，使用正确的预训练权重路径
2. **学习率设置**: 预训练encoder使用1e-5，避免破坏学到的特征
3. **实验命名**: 使用不同的`--model_name`，避免覆盖
4. **日志保存**: 使用`tee`命令同时输出到终端和日志文件
5. **GPU内存**: 如果OOM，可以减小batch_size到4

---

## 下一步（实验完成后）

1. 📊 整理实验结果到 `QUALICLIP_RESULTS.md`
2. 📈 绘制training curves对比图
3. 📝 更新 `BENCHMARK_RESULTS.md` 添加QualiCLIP结果
4. 🔀 Merge `feature/qualiclip-pretrain` 分支到 `master`
5. 📢 准备实验报告/论文材料

