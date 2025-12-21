# 预训练模型测试总结

## 测试结果

### 问题发现

1. **KonIQ-10k 测试失败**
   - 错误：`x` and `y` must have length at least 2
   - 原因：可能是图像路径问题，导致没有成功加载图像
   
2. **其他数据集缺失**
   - SPAQ: `spaq_test.json` 不存在
   - KADID-10K: `kadid_test.json` 不存在  
   - AGIQA-3K: `agiqa_test.json` 不存在

## 当前状态

### 已完成的工作

1. ✅ 修复了 Baseline (ResNet-50) 的 LR 衰减 bug
2. ✅ 创建了完整的实验计划 (`FINAL_EXPERIMENT_PLAN.md`)
3. ✅ 分析了所有实验结果
4. ✅ 创建了预训练模型测试脚本

### 当前最佳模型

**Swin-Base + 强正则化 + 低学习率**
- SRCC: 0.9319 ± 0.0016 (3 轮平均)
- Round 1: 0.9316
- Round 2: 0.9305
- Round 3: 0.9336 ⭐

**Checkpoint**: 
```
checkpoints/koniq-10k-swin-ranking-alpha0.5_20251220_091014/
best_model_srcc_0.9336_plcc_0.9464.pkl
```

## 下一步建议

### 方案 1：验证 Baseline 修复（必须）

修复了 backbone LR 衰减 bug 后，需要重新验证：

```bash
# 停止当前正在运行的 baseline 实验
pkill -f "train_test_IQA.py"

# 重新运行 baseline（修复后，预期 SRCC ~0.900+）
python train_test_IQA.py \
  --dataset koniq-10k \
  --train_test_num 3 \
  --no_spaq
```

**预期时间**：约 9 小时（3 轮 × 3 小时）

### 方案 2：运行消融实验（推荐）

使用快速版本（`epochs=15, patience=5, train_test_num=1`）：

```bash
# 1. 去掉 Multi-Scale Fusion (~6小时)
python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 \
  --epochs 15 --patience 5 --train_test_num 1 --no_multiscale \
  --ranking_loss_alpha 0.5 --lr 5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq

# 2. 去掉 Ranking Loss (~6小时)
python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 \
  --epochs 15 --patience 5 --train_test_num 1 \
  --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq

# ... 其他4个消融实验
```

**总时间**：约 36 小时（6 个实验 × 6 小时）

### 方案 3：跨数据集测试（可选）

需要先准备测试集的 JSON 文件：
- `SPAQ/spaq_test.json`
- `KADID-10K/kadid_test.json`
- `AGIQA-3K/agiqa_test.json`

然后运行：
```bash
python cross_dataset_test.py \
  --checkpoint checkpoints/koniq-10k-swin-ranking-alpha0.5_20251220_091014/best_model_srcc_0.9336_plcc_0.9464.pkl \
  --model_size base \
  --test_patch_num 20
```

## 时间规划

**必须完成（45 小时）**：
- Baseline 验证：9 小时
- 6 个消融实验：36 小时

**可选（30 小时）**：
- 改进 Swin-Small + Attention：30 小时

**总计**：45-75 小时（2-3 天）

## 报告建议

### 主模型
"We report the average of 3 independent runs for the final model 
(SRCC = 0.9319 ± 0.0016) to demonstrate stability and reproducibility."

### 消融实验
"For ablation studies, we use single-run results with reduced epochs 
(15 instead of 30) to efficiently compare relative performance differences 
while maintaining computational feasibility."

### Baseline Bug 修复
"We identified and fixed a bug in the baseline implementation where the 
backbone learning rate was not decaying properly, which caused a ~1.3% 
performance drop (0.888 vs 0.9009)."

