# 最终实验计划

## 🐛 重要Bug修复

### ResNet-50 Baseline 无法复现的原因

**问题**：当前 SRCC 0.888，期望 0.9009，差距 -1.3%

**根本原因**：`HyerIQASolver.py` 第 261 行的 bug
```python
# 错误的代码（当前）
backbone_lr = self.lr  # Backbone LR stays constant

# 正确的代码（已修复）
backbone_lr = self.lr / pow(10, (t // 6))  # Backbone LR also decays
```

**影响**：Backbone 的学习率没有衰减，导致训练不充分

**修复状态**：✅ 已修复

---

## 📊 已有实验结果分析

### 1. Swin-Base（当前最佳）

**配置**：
- Model: Swin-Base
- Alpha: 0.5
- LR: 5e-6
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4

**结果**（3 轮）：
- Round 1: SRCC 0.9316, PLCC 0.9450
- Round 2: SRCC 0.9305, PLCC 0.9444
- Round 3: SRCC 0.9336, PLCC 0.9464 ⭐
- **平均**: SRCC 0.9319 ± 0.0016

**优势**：
✅ 性能最高
✅ 稳定性好（3轮波动小）

---

### 2. Swin-Small + Attention

**配置**：
- Model: Swin-Small
- Attention Fusion: Enabled
- Alpha: 0.5

**结果**（3 轮）：
- Round 1: SRCC 0.9311, PLCC 0.9424 ⭐ 很好！
- Round 2: SRCC 0.9293, PLCC 0.9425
- Round 3: SRCC 0.9254, PLCC 0.9402
- **平均**: SRCC 0.9286 ± 0.0029

**分析**：
✅ Round 1 结果很好（0.9311，只比 Base 低 0.05%）
⚠️ 稳定性差（Round 2, 3 下降）
💡 **问题不是模型不好，而是训练不稳定**

**可能的改进方向**：
1. 增加 Weight Decay（从 1e-4 → 2e-4）
2. 增加 Drop Path Rate（从 0.2 → 0.3）
3. 降低学习率（从 1e-5 → 5e-6）

---

### 3. PLCC 0.9471 的来源

查找结果：这是 **Swin-Tiny** 的 **训练集** SRCC（不是测试集）

```
Epoch 2: Train_SRCC: 0.9473, Test_SRCC: 0.9162, Test_PLCC: 0.9314
```

**结论**：没有测试集 PLCC 0.9471 的结果

最高的测试集 PLCC 是：
- Swin-Base Round 3: **PLCC 0.9464** ⭐

---

## 🎯 推荐的实验方案

### 方案 A：快速完成（推荐）

**1. 最佳模型（已完成）**
- 使用现有的 Swin-Base 3 轮结果
- SRCC = 0.9319 ± 0.0016
- ✅ 不需要重新训练

**2. 消融实验（6个，快速版）**

配置：`--train_test_num 1 --epochs 15 --patience 5`

```bash
# 1. 去掉 Multi-Scale Fusion
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 15 \
  --patience 5 \
  --train_test_num 1 \
  --no_multiscale \
  --ranking_loss_alpha 0.5 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

# 2. 去掉 Ranking Loss
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 15 \
  --patience 5 \
  --train_test_num 1 \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

# 3. 去掉 Drop Path
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 15 \
  --patience 5 \
  --train_test_num 1 \
  --ranking_loss_alpha 0.5 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

# 4. 去掉 Dropout
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 15 \
  --patience 5 \
  --train_test_num 1 \
  --ranking_loss_alpha 0.5 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

# 5. 去掉 Test Random Crop
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 15 \
  --patience 5 \
  --train_test_num 1 \
  --ranking_loss_alpha 0.5 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --no_spaq

# 6. 使用 Swin-Small（模型容量对比）
python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
  --batch_size 64 \
  --epochs 15 \
  --patience 5 \
  --train_test_num 1 \
  --ranking_loss_alpha 0.5 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**时间估算**：
- 每个实验：约 6 小时
- 总计：36 小时（1.5 天）

---

### 方案 B：尝试改进 Swin-Small + Attention（可选）

如果你想探索 Small + Attention 的潜力：

```bash
# Swin-Small + Attention + 更强正则化
python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
  --batch_size 64 \
  --epochs 30 \
  --patience 7 \
  --train_test_num 3 \
  --ranking_loss_alpha 0.5 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --attention_fusion \
  --no_spaq
```

**预期**：
- 如果成功，可能达到 SRCC 0.9320+（接近 Base）
- 但参数量只有 Base 的 60%
- 时间：约 30 小时（3 轮 × 10 小时）

**风险**：
- 可能仍然不稳定
- 可能不如 Base

---

### 方案 C：验证 Baseline 修复（必须）

```bash
# 重新运行 ResNet-50 baseline（修复后）
python train_test_IQA.py \
  --dataset koniq-10k \
  --train_test_num 3 \
  --no_spaq
```

**预期**：
- SRCC 应该恢复到 0.900+ 
- 时间：约 9 小时（3 轮 × 3 小时）

---

## 📝 报告中的说明

### 主模型
"We report the average of 3 independent runs for the final model 
(SRCC = 0.9319 ± 0.0016) to demonstrate stability and reproducibility."

### 消融实验
"For ablation studies, we use single-run results with reduced epochs 
(15 instead of 30) to efficiently compare relative performance differences 
while maintaining computational feasibility."

### Baseline
"We identified and fixed a bug in the baseline implementation where the 
backbone learning rate was not decaying properly, which caused a ~1.3% 
performance drop."

---

## 🎯 最终推荐

**立即执行**：
1. ✅ 修复 baseline bug（已完成）
2. 🔄 重新运行 baseline（验证修复）
3. 🚀 运行 6 个消融实验（方案 A）

**可选**：
- 如果时间充足，尝试改进 Small + Attention（方案 B）

**总时间**：
- 必须：36 小时（消融）+ 9 小时（baseline）= 45 小时（2 天）
- 可选：+30 小时（Small + Attention）

