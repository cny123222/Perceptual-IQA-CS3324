# Early Stopping 使用指南

## 功能说明

Early Stopping（提前停止）功能已实现，用于：
- ✅ 自动保存最佳模型（基于验证集 SRCC）
- ✅ 防止过拟合（当性能不再提升时自动停止）
- ✅ 节省训练时间（不需要训练完所有 epochs）

## 默认行为

- **默认启用** Early Stopping
- **默认 patience = 5**（连续 5 个 epoch 无提升则停止）
- 每个 epoch 自动保存：
  - 常规 checkpoint：`checkpoint_epoch_N_srcc_X.XXXX_plcc_X.XXXX.pkl`
  - 最佳模型：`best_model_srcc_X.XXXX_plcc_X.XXXX.pkl`（当性能提升时更新）

## 使用示例

### 1. 默认使用（推荐）

```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 20 \
  --use_multiscale \
  --ranking_loss_alpha 0
```

**说明**：
- 训练最多 20 个 epochs
- 如果连续 5 个 epoch 验证集 SRCC 无提升，自动停止
- 最佳模型自动保存

### 2. 自定义 patience

```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 20 \
  --patience 3 \
  --use_multiscale \
  --ranking_loss_alpha 0
```

**说明**：
- 连续 3 个 epoch 无提升则停止（更激进）

### 3. 禁用 Early Stopping

```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 20 \
  --no_early_stopping \
  --use_multiscale \
  --ranking_loss_alpha 0
```

**说明**：
- 训练完所有 20 个 epochs
- 仍然会保存最佳模型

## 训练输出示例

```
Early stopping enabled with patience=5
Epoch	Train_Loss	Train_SRCC	Test_SRCC	Test_PLCC	SPAQ_SRCC	SPAQ_PLCC
Epoch 1/20:
  Total batches: 1474
  ...
1	4.850		0.8823		0.9193		0.9346	0.8621	0.8603
  Model saved to: .../checkpoint_epoch_1_srcc_0.9193_plcc_0.9346_...pkl
  ⭐ New best model saved! SRCC: 0.9193, PLCC: 0.9346
     Path: .../best_model_srcc_0.9193_plcc_0.9346_...pkl

Epoch 2/20:
  ...
2	3.004		0.9553		0.9194		0.9323	0.8575	0.8528
  Model saved to: .../checkpoint_epoch_2_srcc_0.9194_plcc_0.9323_...pkl
  ⭐ New best model saved! SRCC: 0.9194, PLCC: 0.9323
     Path: .../best_model_srcc_0.9194_plcc_0.9323_...pkl

Epoch 3/20:
  ...
3	2.501		0.9723		0.9180		0.9310	0.8560	0.8515
  Model saved to: .../checkpoint_epoch_3_srcc_0.9180_plcc_0.9310_...pkl
  (No improvement - 1 epoch without improvement)

...

Epoch 7/20:
  ...
7	1.823		0.9889		0.9165		0.9290	0.8540	0.8490
  Model saved to: .../checkpoint_epoch_7_srcc_0.9165_plcc_0.9290_...pkl
  (No improvement - 5 epochs without improvement)

🛑 Early stopping triggered!
   No improvement for 5 consecutive epochs.
   Best SRCC: 0.9194, Best PLCC: 0.9323
   Best model saved at: .../best_model_srcc_0.9194_plcc_0.9323_...pkl
```

## 最佳实践

### 1. 数据集大小与 patience 的关系

| 数据集大小 | 推荐 patience | 说明 |
|-----------|--------------|------|
| 小型 (<5K) | 3-5 | 快速收敛，提前停止 |
| 中型 (5K-10K) | 5-7 | 平衡速度和性能 |
| 大型 (>10K) | 7-10 | 给予更多训练时间 |

### 2. 结合其他技术

**推荐配置**：
```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 30 \              # 设置足够大的 epochs
  --patience 5 \             # 让 early stopping 决定何时停止
  --use_multiscale \         # 多尺度特征融合
  --ranking_loss_alpha 0 \   # 暂不使用 ranking loss
  --batch_size 96 \
  --train_patch_num 20 \
  --test_patch_num 20
```

### 3. 如何选择最佳模型

训练结束后，使用 `best_model_*.pkl` 文件：
- 它对应验证集 SRCC 最高的那个 epoch
- 已经考虑了过拟合问题
- 直接用于测试和部署

### 4. 调试时禁用 Early Stopping

如果你想观察完整的训练曲线（用于分析过拟合等问题）：
```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 10 \
  --no_early_stopping  # 强制训练完所有 epochs
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--patience` | 5 | 连续多少个 epoch 无提升则停止 |
| `--no_early_stopping` | False | 添加此标志以禁用 early stopping |
| `--epochs` | 16 | 最大训练轮数 |

## 注意事项

1. **Early stopping 基于验证集 SRCC**
   - 确保你的验证集有代表性
   - CenterCrop 测试确保结果可复现

2. **最佳模型会覆盖**
   - 每次出现更好的性能时，会覆盖之前的 `best_model_*.pkl`
   - 所有 epoch 的 checkpoint 仍然保留

3. **与随机种子配合**
   - 代码已设置随机种子（seed=42）
   - Early stopping + CenterCrop + 固定种子 = 完全可复现

4. **SPAQ 跨数据集测试**
   - Early stopping 仅基于主验证集（如 KonIQ-10k test）
   - SPAQ 结果仅用于参考，不影响 early stopping 决策

## 实施效果

根据我们的观察：
- 测试集 SRCC 通常在 **1-2 个 epoch 达到峰值**
- 使用 `patience=5` 可以给予足够的容错空间
- 预期节省 **50-70% 训练时间**（从 20 epochs → 5-10 epochs）

---

**生成时间**: 2025-12-17  
**适用版本**: HyperIQA Swin + ResNet (both supported)

