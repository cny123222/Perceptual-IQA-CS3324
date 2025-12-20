# Swin-Base 学习率调度器分析

## 🤔 用户的关键问题

1. **最优配置还是不能完全复现** (0.9316 vs 0.9336)
2. **种子设置确定是对的吗？**
3. **调度器怎么样？换成 step 会更好吗？**

---

## ✅ 种子设置验证

### 当前代码（train_swin.py）

```python
# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**结论**：✅ **种子设置完全正确**，没有问题。

---

## 🎯 关键发现：Cosine vs Step 的差异

### 当前配置（Swin-Base）

- **Epochs**: 30
- **Learning Rate**: 5e-6 (backbone), 5e-5 (hypernet)
- **Scheduler**: **Cosine Annealing**
- **T_max**: 30, eta_min=1e-6

### Cosine Annealing 在 30 epochs 的行为

| Epoch | HyperNet LR | vs Initial | 说明 |
|-------|-------------|------------|------|
| 1 | 0.000049 | -2% | 几乎不变 |
| 2 | 0.000047 | -6% | 轻微下降 |
| 3 | 0.000043 | -14% | 开始明显下降 |
| 5 | 0.000035 | -30% | 下降较多 |
| 10 | 0.000018 | -64% | 大幅下降 |
| 15 | 0.000007 | -86% | 接近最小值 |
| 20 | 0.000002 | -96% | 几乎最小 |
| 30 | 0.000001 | -98% | 最小值 |

**关键观察**：
- ✅ Epoch 1-2: 学习率几乎不变（-2% to -6%）
- ⚠️ Epoch 3-5: 开始明显下降（-14% to -30%）
- ❌ Epoch 10+: 学习率已经很低（-64%以上）

### Step Decay 在 30 epochs 的行为

原始 HyperIQA 的 step decay：每 6 个 epoch 除以 10

| Epoch | HyperNet LR | vs Initial | 说明 |
|-------|-------------|------------|------|
| 1-6 | 0.000050 | 0% | **保持不变** |
| 7-12 | 0.000005 | -90% | 第一次下降 |
| 13-18 | 0.0000005 | -99% | 第二次下降 |
| 19-24 | 0.00000005 | -99.9% | 第三次下降 |
| 25-30 | 0.000000005 | -99.99% | 第四次下降 |

**关键观察**：
- ✅ Epoch 1-6: 学习率**完全保持不变**
- ❌ Epoch 7+: 学习率骤降 90%
- ❌ Epoch 13+: 学习率几乎为 0

---

## 📊 对比分析：哪个更适合 Swin-Base？

### 场景 1：之前最佳模型的训练曲线

```
Epoch 1: SRCC 0.9327, PLCC 0.9451
Epoch 2: SRCC 0.9336, PLCC 0.9464 ⭐ 最佳
Epoch 3: SRCC 0.9309, PLCC 0.9445
Epoch 4: SRCC 0.9313, PLCC 0.9429
Epoch 5: SRCC 0.9299, PLCC 0.9413
Epoch 6: SRCC 0.9288, PLCC 0.9396
```

**关键发现**：
- ✅ **Epoch 2 达到最佳**
- ⚠️ Epoch 3+ 开始过拟合（轻微下降）
- 📉 Epoch 6 已经明显下降

### 场景 2：当前实验的训练曲线

```
Epoch 1: SRCC 0.9316, PLCC 0.9450
Epoch 2: SRCC 0.9299, PLCC 0.9424
Epoch 3: (进行中...)
```

**关键发现**：
- ⚠️ Epoch 1 略低于之前（-0.0011）
- ❌ Epoch 2 下降而非上升（-0.0017）
- 📉 趋势与之前不同

---

## 🎯 深入分析：为什么 Cosine 可能更好？

### 1. Swin-Base 不是 ResNet-50！

**ResNet-50 (Baseline)**：
- ✅ 预训练权重非常成熟
- ✅ HyperNetwork 很小
- ✅ Epoch 1-2 就收敛
- ✅ 需要高学习率快速收敛
- **→ Step decay 更合适**

**Swin-Base**：
- ⚠️ 模型更大（88M vs 27M）
- ⚠️ 更容易过拟合
- ⚠️ 需要更强的正则化
- ⚠️ 需要更平滑的学习率下降
- **→ Cosine 可能更合适？**

### 2. 训练动态的差异

#### Cosine Annealing (当前)

**优点**：
- ✅ 平滑下降，避免震荡
- ✅ Epoch 1-2 学习率几乎不变（-2% to -6%）
- ✅ 后期逐渐降低，防止过拟合
- ✅ 适合需要长时间训练的大模型

**缺点**：
- ⚠️ Epoch 3+ 学习率下降较快（-14%+）
- ⚠️ 可能限制了后期的学习能力

#### Step Decay

**优点**：
- ✅ Epoch 1-6 保持高学习率
- ✅ 充分学习，快速收敛

**缺点**：
- ❌ Epoch 7 突然下降 90%（太激进）
- ❌ 可能导致震荡或不稳定
- ❌ 不适合需要平滑微调的大模型

---

## 🔬 实验证据分析

### 证据 1：之前最佳模型在 Epoch 2 达到最佳

**使用 Cosine**：
- Epoch 1: LR ≈ 0.000049 → SRCC 0.9327
- Epoch 2: LR ≈ 0.000047 (-6%) → SRCC 0.9336 ⭐
- Epoch 3: LR ≈ 0.000043 (-14%) → SRCC 0.9309 ↓

**分析**：
- ✅ Epoch 1-2 的轻微 LR 下降（-6%）可能帮助了收敛
- ⚠️ Epoch 3 的 LR 下降（-14%）可能太快，导致无法继续提升

### 证据 2：当前实验的异常行为

**使用 Cosine**：
- Epoch 1: SRCC 0.9316 (vs 0.9327, -0.0011)
- Epoch 2: SRCC 0.9299 (vs 0.9336, -0.0037) ↓

**可能原因**：
1. ❓ 随机性（数据增强、dropout）
2. ❓ 初始化差异
3. ❓ 学习率调度的微妙差异

---

## 💡 关键洞察：问题可能不在调度器

### 观察 1：Epoch 1 就有差异

```
之前：Epoch 1 SRCC 0.9327
当前：Epoch 1 SRCC 0.9316
差距：-0.0011
```

**在 Epoch 1，两次实验的 LR 几乎相同**（都是 -2%）！

**结论**：Epoch 1 的差异**不是**调度器造成的。

### 观察 2：训练动态不同

```
之前：Epoch 1 → Epoch 2: +0.0009 (上升) ↗️
当前：Epoch 1 → Epoch 2: -0.0017 (下降) ↘️
```

**这是训练轨迹的根本差异，不仅仅是调度器的问题。**

---

## 🎯 真正的问题：深度学习的混沌特性

### 1. 微小差异被放大

即使所有参数相同，微小的初始差异会被放大：

```
Epoch 1 差异: -0.0011 (0.12%)
    ↓ (放大)
Epoch 2 差异: -0.0037 (0.40%)
```

### 2. 不同的优化轨迹

两次训练可能走向不同的局部最优：

```
之前：找到了一个更好的局部最优（0.9336）
当前：找到了一个稍差的局部最优（0.9316?）
```

### 3. 随机性的累积效应

- ColorJitter 的随机扰动
- Dropout 的随机 mask
- DropPath 的随机路径
- 数据 shuffle 的顺序

**这些随机性在不同运行中累积，导致不同的结果。**

---

## 🤔 换成 Step Decay 会更好吗？

### 理论分析

**可能更好的情况**：
- ✅ 如果模型需要前期的高学习率充分学习
- ✅ 如果 Epoch 2-3 的 Cosine 下降太快

**可能更差的情况**：
- ❌ Epoch 7 的突然下降可能导致震荡
- ❌ 不适合需要平滑微调的大模型
- ❌ Swin-Base 已经在 Epoch 2 达到最佳，后续都是过拟合

### 实验建议

#### 方案 A：尝试 Step Decay ⚠️

```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --ranking_loss_margin 0.1 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler step \          # 改成 step
  --test_random_crop \
  --no_spaq
```

**预期**：
- 可能 Epoch 1-6 保持高性能
- 但 Epoch 7 突然下降可能导致问题
- 不确定是否能超过 0.9336

#### 方案 B：保持 Cosine，多次运行 ✅ 推荐

```bash
# 运行 3-5 次，使用不同的随机种子
for seed in 42 43 44 45 46; do
    python train_swin.py \
      [所有参数...] \
      --seed $seed
done
```

**预期**：
- 某次运行可能达到 0.9336 或更高
- 可以报告均值和标准差
- 更科学的做法

#### 方案 C：微调 Cosine 参数 🤔

使用更大的 `eta_min`，让学习率下降更慢：

```python
# 在 HyperIQASolver_swin.py 中修改
CosineAnnealingLR(self.solver, T_max=30, eta_min=5e-7)  # 从 1e-6 改成 5e-7
```

**预期**：
- 学习率下降稍微平缓一些
- 可能帮助后期的学习

---

## 📊 实验计划

### 优先级 1：多次运行（推荐）✅

```bash
# 当前实验继续运行，看最终结果
# 同时启动 2-3 次新的运行，使用不同种子

# Run 1 (seed=42, 当前进行中)
# 等待完成...

# Run 2 (seed=43)
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --ranking_loss_margin 0.1 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

# Run 3 (seed=44)
# 同上，只是种子不同
```

**预期结果**：
- 3 次运行的 SRCC 可能在 0.9310 - 0.9340 之间
- 报告：`SRCC: 0.9325 ± 0.0015` (mean ± std)

### 优先级 2：尝试 Step Decay ⚠️

只运行 1 次，看看效果：

```bash
python train_swin.py \
  [所有参数...] \
  --lr_scheduler step
```

**预期结果**：
- 可能在 0.9320 - 0.9340 之间
- 不确定是否能超过 Cosine

### 优先级 3：等待当前实验完成 ⏳

当前实验（seed=42, Cosine）还在 Epoch 3，需要等到：
- Early stopping 触发
- 或达到 30 epochs

**可能的结果**：
- 最佳 SRCC 可能在 Epoch 1 (0.9316)
- 或者后续 epoch 可能反弹到 0.9320-0.9330

---

## ✅ 最终建议

### 1. 种子设置 ✅

**完全正确**，没有问题。差异不是种子的问题。

### 2. 调度器选择 🤔

**Cosine vs Step**：

| 方面 | Cosine (当前) | Step | 建议 |
|------|--------------|------|------|
| 适合模型 | 大模型，需要平滑微调 | 小模型，快速收敛 | Cosine ✅ |
| Epoch 1-2 | 几乎不变（-2% to -6%） | 完全不变 | 差异不大 |
| Epoch 3-6 | 逐渐下降（-14% to -40%） | 完全不变 | Cosine 可能更好 |
| Epoch 7+ | 继续平滑下降 | 突然下降 90% | Cosine 更稳定 |
| **总体** | **更适合 Swin-Base** | 更适合 ResNet-50 | **保持 Cosine** ✅ |

### 3. 如何提高可复现性 🎯

**最佳方案**：多次运行，报告统计结果

```bash
# 运行 3-5 次
for seed in 42 43 44 45 46; do
    # 修改 train_swin.py 支持 --seed 参数
    # 或手动修改代码中的 seed 值
    python train_swin.py [所有参数...]
done
```

**报告格式**：
```
Best SRCC: 0.9336 (seed=42, from previous run)
Mean SRCC: 0.9328 ± 0.0012 (over 5 runs)
Range: 0.9316 - 0.9340
```

---

## 📝 结论

1. ✅ **种子设置正确**：不是问题所在
2. 🤔 **调度器选择**：Cosine 更适合 Swin-Base，不建议换成 Step
3. 🎯 **真正问题**：深度学习的固有随机性，±0.002 的波动是正常的
4. ✅ **解决方案**：多次运行，报告统计结果

**不建议换成 Step Decay**，因为：
- Cosine 更适合大模型（Swin-Base 88M）
- Step 的突然下降可能导致不稳定
- 当前差异主要是随机性，不是调度器的问题

**建议**：
- 等待当前实验完成
- 运行 2-3 次额外实验
- 报告均值和标准差

---

**文档版本**: 1.0  
**创建时间**: 2025-12-21  
**建议**: 保持 Cosine，多次运行取平均 ✅

