# Learning Rate Scheduler 使用指南

## 功能说明

已实现灵活的学习率调度器，支持三种模式：
- ✅ **CosineAnnealingLR**（默认，推荐）：平滑的余弦退火
- ✅ **Step Decay**（原始方法）：每6个epoch除以10
- ✅ **Constant LR**：固定学习率，不衰减

## 为什么使用 CosineAnnealingLR？

### 原始方法的问题

**Step Decay** (每6个epoch ÷10):
```
Epoch 0-5:  LR = 1e-4
Epoch 6-11: LR = 1e-5  ⬇ 突然下降
Epoch 12+:  LR = 1e-6  ⬇ 过早过小
```

❌ **缺点**:
- 学习率**突然下降**，可能导致训练不稳定
- 前6个epoch学习率固定，可能过大或过小
- 12 epoch后学习率已经很小（1e-6），后续训练几乎无效

### CosineAnnealingLR 的优势

```
Epoch 0:  LR = 1e-4  (max)
Epoch 5:  LR ≈ 5e-5  (平滑下降)
Epoch 10: LR ≈ 2e-5
Epoch 15: LR ≈ 5e-6
Epoch 20: LR ≈ 1e-6  (min)
```

✅ **优点**:
1. **平滑过渡**：学习率连续下降，避免突变
2. **早期适度**：初期有足够的探索空间
3. **后期精细**：末期可以做细微调整
4. **灵活控制**：通过 `T_max` 控制衰减速度

## 使用方法

### 1️⃣ 默认使用（推荐）

```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 20 \
  --ranking_loss_alpha 0
```

**输出**:
```
Learning rate scheduler: CosineAnnealingLR (T_max=20, eta_min=1e-6)
Early stopping enabled with patience=5
...
Epoch 1: ...
  Learning rates: HyperNet=0.000095, Backbone=0.000095
Epoch 2: ...
  Learning rates: HyperNet=0.000081, Backbone=0.000081
```

**说明**: 
- 默认使用 CosineAnnealingLR
- 学习率从初始值平滑降到 1e-6
- 每个epoch后打印当前学习率

---

### 2️⃣ 使用原始 Step Decay

```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 20 \
  --lr_scheduler step
```

**说明**: 
- 恢复原始的阶梯式衰减
- 每6个epoch学习率除以10
- 适合复现原文结果

---

### 3️⃣ 禁用学习率调度（固定LR）

```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 20 \
  --no_lr_scheduler
```

**说明**: 
- 学习率保持初始值不变
- 适合调试或特殊实验

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--lr_scheduler` | `cosine` | 调度器类型：`cosine` / `step` / `none` |
| `--no_lr_scheduler` | False | 禁用学习率调度（等价于 `--lr_scheduler none`） |
| `--lr` | 1e-4 | 初始学习率 |
| `--lr_ratio` | 10 | HyperNet 与 Backbone 的学习率比例 |

**注意**: 
- `--lr_scheduler none` 和 `--no_lr_scheduler` 效果相同
- CosineAnnealingLR 的 `eta_min` 固定为 1e-6（可以在代码中修改）

---

## 学习率曲线对比

### CosineAnnealingLR (T_max=20)

```
1e-4 |●
     |  ●
     |    ●
     |      ●
     |        ●
     |          ●
     |            ●
     |              ●
     |                ●
1e-6 |__________________|●
     0                   20 epochs
```

**特点**: 平滑下降，后期仍有调整空间

### Step Decay (原始)

```
1e-4 |●●●●●●|
     |      |
1e-5 |      ●●●●●●|
     |            |
1e-6 |____________●●●●●●
     0  6        12     18 epochs
```

**特点**: 阶梯式，突然下降

---

## 实验建议

### 场景 A: 首次训练新模型

**推荐配置**:
```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 30 \
  --lr_scheduler cosine \  # 默认
  --patience 7
```

**理由**: 
- CosineAnnealingLR 更平滑，适合探索
- 更多 epochs 给予充分时间
- 较大 patience 避免过早停止

---

### 场景 B: 复现原文结果

**推荐配置**:
```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 16 \
  --lr_scheduler step \    # 使用原始方法
  --no_early_stopping      # 训练完所有 epochs
```

**理由**: 
- 与原文保持一致
- 可以公平对比

---

### 场景 C: 快速实验/调试

**推荐配置**:
```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 10 \
  --no_lr_scheduler \      # 固定 LR
  --patience 3
```

**理由**: 
- 固定LR简化变量
- 小 patience 快速迭代

---

## 高级：自定义学习率调度

如果需要更复杂的调度策略，可以修改 `HyperIQASolver_swin.py`:

### 示例 1: Warmup + Cosine

```python
# 在 train() 函数中
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

warmup_epochs = 2
warmup_scheduler = LambdaLR(
    self.solver, 
    lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs
)
cosine_scheduler = CosineAnnealingLR(
    self.solver, 
    T_max=self.epochs - warmup_epochs, 
    eta_min=1e-6
)
self.scheduler = SequentialLR(
    self.solver,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs]
)
```

### 示例 2: ReduceLROnPlateau（基于验证集）

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

self.scheduler = ReduceLROnPlateau(
    self.solver, 
    mode='max',      # 最大化 SRCC
    factor=0.5,      # LR × 0.5
    patience=3,      # 3 epochs 无提升
    verbose=True
)

# 在验证后调用
self.scheduler.step(test_srcc)
```

---

## 常见问题

### Q1: CosineAnnealingLR 会让训练更慢吗？

**A**: 不会。计算开销几乎为零，只是改变学习率的更新方式。

### Q2: 为什么我的学习率没有变化？

**A**: 检查：
1. 确认没有使用 `--no_lr_scheduler`
2. 检查输出中的 "Learning rate scheduler" 信息
3. 确认每个 epoch 后有打印学习率

### Q3: CosineAnnealingLR 和 Early Stopping 冲突吗？

**A**: 不冲突。Early Stopping 基于验证集性能，学习率调度是独立的。两者配合使用效果更好。

### Q4: 可以中途改变调度器吗？

**A**: 不建议。如果需要，可以：
1. 停止训练
2. 加载 checkpoint
3. 用不同的 `--lr_scheduler` 重新启动

---

## 实验记录模板

建议记录以下信息以便对比：

```markdown
## Experiment: [实验名称]

**配置**:
- Dataset: koniq-10k
- Epochs: 30
- LR Scheduler: cosine / step / none
- Initial LR: 1e-4
- Patience: 5

**结果**:
- Best SRCC: X.XXXX (Epoch N)
- Best PLCC: X.XXXX
- Total epochs trained: N (stopped by early stopping / full)

**观察**:
- [记录训练过程中的发现]
```

---

## 性能对比（示例）

| 配置 | Best SRCC | Best PLCC | Epochs | 说明 |
|------|-----------|-----------|--------|------|
| Cosine (T=30, patience=5) | 0.9194 | 0.9323 | 7 | 早期达峰，稳定 |
| Step (original) | 0.9180 | 0.9310 | 12 | 需要更多epochs |
| Constant LR | 0.9150 | 0.9280 | 15 | 性能略低 |

*注：以上数据为示例，实际结果以你的实验为准*

---

**生成时间**: 2025-12-17  
**适用版本**: HyperIQA Swin + ResNet (both supported)

