# 实验运行指南：最佳配置与消融实验

## 📋 目录
1. [最佳模型训练](#1-最佳模型训练)
2. [Baseline 对照实验](#2-baseline-对照实验)
3. [核心消融实验](#3-核心消融实验)
4. [实验结果记录](#4-实验结果记录)

---

## 1. 最佳模型训练

### 🏆 完整配置（推荐）

**模型**：Swin-Base + Multi-Scale + Ranking Loss + ColorJitter + Strong Regularization

**预期性能**：
- SRCC: **0.9336**
- PLCC: **0.9464**
- 训练时间：约 3-4 小时（NVIDIA A100/RTX 3090）

**完整命令**：
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
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**注意事项**：
- `--no_spaq`：跳过 SPAQ 跨数据集测试（节省时间，后续可单独测试）
- `--test_random_crop`：使用 RandomCrop 测试（匹配原论文，但可复现性稍差）
- ColorJitter 自动启用（在 `data_loader.py` 中，koniq-10k 训练集默认开启）
- Weight decay = 2e-4（在代码中设置，需要确认）

---

## 2. Baseline 对照实验

### 实验 0：原始 HyperIQA (ResNet-50)

**目的**：建立性能 baseline，证明代码实现正确

**预期性能**：
- SRCC: ~0.9009
- PLCC: ~0.9170

**命令**：
```bash
python train_test_IQA.py \
  --dataset koniq-10k \
  --epochs 10 \
  --batch_size 96 \
  --train_patch_num 20 \
  --test_patch_num 20
```

**说明**：
- 使用原始 ResNet-50 骨干网络
- 10 epochs 足够（通常 epoch 1-2 达到最佳）
- 这是所有改进的对比基准

---

## 3. 核心消融实验

### 消融 1：Swin-Base Baseline（无特殊技巧）

**目的**：验证仅替换骨干网络的效果

**控制变量**：
- ✅ Swin-Base 骨干网络
- ✅ Multi-Scale 特征融合（默认启用）
- ❌ 无 Ranking Loss (alpha=0)
- ❌ 无 ColorJitter（需要暂时禁用）
- ❌ 弱正则化（drop_path=0.1, dropout=0.2）
- ❌ 无 LR Scheduling

**预期性能**：SRCC ~0.925-0.930

**命令**：
```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0 \
  --lr 1e-5 \
  --drop_path_rate 0.1 \
  --dropout_rate 0.2 \
  --lr_scheduler none \
  --test_random_crop \
  --no_spaq
```

**注意**：需要临时修改 `data_loader.py`，在 koniq-10k 训练部分注释掉 ColorJitter：
```python
# 临时注释掉这一行（第49行）
# torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
```

---

### 消融 2：+ Ranking Loss (alpha=0.5)

**目的**：验证 Ranking Loss 的贡献

**新增**：
- ✅ Ranking Loss (alpha=0.5)

**其他保持与消融1相同**

**预期提升**：+0.2~0.3% SRCC

**命令**：
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
  --lr 1e-5 \
  --drop_path_rate 0.1 \
  --dropout_rate 0.2 \
  --lr_scheduler none \
  --test_random_crop \
  --no_spaq
```

**注意**：仍需禁用 ColorJitter

---

### 消融 3：+ ColorJitter

**目的**：验证 ColorJitter 数据增强的贡献

**新增**：
- ✅ ColorJitter（恢复 data_loader.py 中的设置）

**其他保持与消融2相同**

**预期提升**：+0.2~0.3% SRCC

**命令**：
```bash
# 与消融2命令相同，但恢复 data_loader.py 中的 ColorJitter
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
  --lr 1e-5 \
  --drop_path_rate 0.1 \
  --dropout_rate 0.2 \
  --lr_scheduler none \
  --test_random_crop \
  --no_spaq
```

**注意**：确保 `data_loader.py` 第49行的 ColorJitter 已取消注释

---

### 消融 4：+ 强正则化

**目的**：验证强正则化策略的贡献

**新增**：
- ✅ Strong Regularization:
  - drop_path_rate: 0.1 → 0.3 (3x)
  - dropout_rate: 0.2 → 0.4 (2x)

**其他保持与消融3相同**

**预期提升**：+0.3~0.5% SRCC（主要防止过拟合）

**命令**：
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
  --lr 1e-5 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler none \
  --test_random_crop \
  --no_spaq
```

---

### 消融 5：+ Cosine LR Scheduling + 降低学习率

**目的**：验证学习率策略的贡献

**新增**：
- ✅ Cosine LR Scheduling
- ✅ Lower LR: 1e-5 → 5e-6 (0.5x)

**其他保持与消融4相同**

**预期提升**：+0.1~0.2% SRCC（训练更稳定）

**命令**：
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
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**结果**：这应该达到最佳性能 SRCC 0.9336 ✅

---

## 消融实验快速参考表

| 实验 | 骨干 | 多尺度 | Rank Loss | ColorJitter | 强正则 | Cosine LR | 低学习率 | 预期 SRCC |
|-----|------|--------|-----------|-------------|--------|-----------|----------|-----------|
| Baseline | ResNet-50 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0.9009 |
| 消融1 | Swin-Base | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ~0.925 |
| 消融2 | Swin-Base | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ~0.927 |
| 消融3 | Swin-Base | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ~0.930 |
| 消融4 | Swin-Base | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ~0.933 |
| **消融5（最佳）** | Swin-Base | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **0.9336** |

**提升累计**：
- Swin-Base: +1.6~2.1%
- + Rank Loss: +0.2~0.3%
- + ColorJitter: +0.2~0.3%
- + Strong Reg: +0.3~0.5%
- + Cosine LR: +0.1~0.2%
- **总提升**: +3.4% (0.9009 → 0.9336)

---

## 4. 实验结果记录

### 推荐的结果记录格式

创建表格记录所有实验结果：

| 实验ID | 配置描述 | SRCC | PLCC | Train Time | 备注 |
|--------|---------|------|------|------------|------|
| EXP-0 | ResNet-50 Baseline | 0.9009 | 0.9170 | ~30min | 原始HyperIQA |
| EXP-1 | Swin-Base Basic | - | - | ~3h | 消融1 |
| EXP-2 | + Ranking Loss | - | - | ~3h | 消融2 |
| EXP-3 | + ColorJitter | - | - | ~3h | 消融3 |
| EXP-4 | + Strong Reg | - | - | ~3h | 消融4 |
| EXP-5 | + Cosine LR (最佳) | 0.9336 | 0.9464 | ~3h | 消融5 ✅ |

---

## 5. 补充实验（可选）

### 实验 A：多尺度融合消融

**目的**：证明多尺度特征融合的贡献

**对照**：单尺度（仅用 Stage 4）vs 多尺度（4个 stages）

**命令（单尺度）**：
```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --no_multiscale \
  --batch_size 32 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --lr 5e-6 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**预期**：SRCC ~0.925（-0.8~1.0%）

---

### 实验 B：Ranking Loss Alpha 调优

**目的**：证明 alpha=0.5 是最优值

**对照**：alpha = 0, 0.3, 0.5, 0.7, 1.0

**命令模板**：
```bash
# 替换 --ranking_loss_alpha [VALUE]
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha [0/0.3/0.5/0.7/1.0] \
  --lr 5e-6 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**已知结果**：
- alpha=0.0: SRCC 0.9307
- alpha=0.3: SRCC 0.9303
- alpha=0.5: SRCC 0.9336 ✅ 最优
- alpha=0.7: 未测试
- alpha=1.0: 未测试

---

### 实验 C：正则化强度调优

**目的**：证明强正则化的必要性

**对照**：
- 弱正则化：drop_path=0.1, dropout=0.2
- 中等：drop_path=0.2, dropout=0.3
- 强正则化：drop_path=0.3, dropout=0.4 ✅

**命令（弱正则化）**：
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
  --lr 5e-6 \
  --drop_path_rate 0.1 \
  --dropout_rate 0.2 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**预期**：过拟合，测试 SRCC ~0.928

---

### 实验 D：模型大小对比

**目的**：展示模型容量的影响

**对照**：Tiny vs Small vs Base

**命令（Tiny）**：
```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size tiny \
  --batch_size 96 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --lr 1e-5 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**命令（Small）**：
```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
  --batch_size 64 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --lr 1e-5 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**已知结果**：
- Tiny: SRCC 0.9236, PLCC 0.9361
- Small: SRCC 0.9303, PLCC 0.9444
- Base: SRCC 0.9336, PLCC 0.9464 ✅

---

## 6. 运行建议

### 时间规划

假设单实验 3 小时，推荐运行顺序：

**第一阶段（核心消融，必须）**：
1. Baseline (0.5h) → 验证代码正确性
2. 消融1-5（5x3h=15h）→ 完整消融链

**第二阶段（补充实验，可选）**：
3. 实验A（3h）→ 多尺度贡献
4. 实验D（2x3h=6h）→ 模型大小对比

**总时间**：约 24-30 小时（可并行运行多个 GPU）

### 并行策略

如果有多块 GPU：
```bash
# GPU 0: Baseline + 消融1
CUDA_VISIBLE_DEVICES=0 python train_test_IQA.py [...] &
CUDA_VISIBLE_DEVICES=0 python train_swin.py [...消融1...] &

# GPU 1: 消融2 + 消融3
CUDA_VISIBLE_DEVICES=1 python train_swin.py [...消融2...] &
CUDA_VISIBLE_DEVICES=1 python train_swin.py [...消融3...] &

# GPU 2: 消融4 + 消融5
CUDA_VISIBLE_DEVICES=2 python train_swin.py [...消融4...] &
CUDA_VISIBLE_DEVICES=2 python train_swin.py [...消融5...] &
```

### 结果验证

每个实验结束后检查：
1. 日志文件（`logs/` 目录）
2. 最佳 SRCC/PLCC
3. 训练曲线（是否过拟合？）
4. 收敛 epoch（是否 early stopping？）

---

## 7. 注意事项

### ⚠️ ColorJitter 控制

- **消融1-2**：需要暂时禁用 ColorJitter
  - 编辑 `data_loader.py` 第49行，注释掉 ColorJitter
- **消融3-5**：需要启用 ColorJitter
  - 确保 `data_loader.py` 第49行 ColorJitter 未注释

**建议**：创建两个版本的 data_loader.py
```bash
# 保存原版本
cp data_loader.py data_loader_with_jitter.py

# 创建无 ColorJitter 版本
sed 's/torchvision.transforms.ColorJitter/#torchvision.transforms.ColorJitter/' \
    data_loader.py > data_loader_no_jitter.py

# 使用时切换
cp data_loader_no_jitter.py data_loader.py  # 消融1-2
cp data_loader_with_jitter.py data_loader.py  # 消融3-5
```

### ⚠️ Weight Decay

当前 weight_decay 在代码中硬编码，需要检查：
```bash
grep "weight_decay" HyperIQASolver_swin.py
```

如果需要修改，可能需要：
1. 在 `train_swin.py` 添加 `--weight_decay` 参数
2. 在 `HyperIQASolver_swin.py` 传递给优化器

### ⚠️ 随机种子

确保 `train_swin.py` 中种子已设置：
```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 8. 预期实验报告表格

### 表1：主要改进对比

| 模型 | SRCC | PLCC | 参数量 | FLOPs | vs Baseline |
|------|------|------|--------|-------|-------------|
| ResNet-50 (Baseline) | 0.9009 | 0.9170 | 25.6M | ~12G | - |
| **Swin-Base (最佳)** | **0.9336** | **0.9464** | **88.8M** | **~18G** | **+3.40%** |

### 表2：消融实验结果

| 实验 | 配置 | SRCC | PLCC | 贡献 |
|------|------|------|------|------|
| 消融1 | Swin-Base Basic | 0.925* | 0.937* | +1.6% |
| 消融2 | + Ranking Loss | 0.927* | 0.939* | +0.2% |
| 消融3 | + ColorJitter | 0.930* | 0.942* | +0.3% |
| 消融4 | + Strong Reg | 0.933* | 0.946* | +0.3% |
| **消融5** | **+ Cosine LR** | **0.9336** | **0.9464** | **+0.1%** |

*预期值，需实际运行验证

### 表3：Ranking Loss Alpha 调优

| Alpha | SRCC | PLCC | 说明 |
|-------|------|------|------|
| 0.0 | 0.9307 | 0.9447 | 纯 L1 Loss |
| 0.3 | 0.9303 | 0.9435 | 权重过低 |
| **0.5** | **0.9336** | **0.9464** | **最优** ✅ |
| 0.7 | -* | -* | 待测试 |
| 1.0 | -* | -* | 待测试 |

---

## 9. 快速启动脚本

创建 `run_ablations.sh`：

```bash
#!/bin/bash

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

# Baseline
echo "Running Baseline (ResNet-50)..."
python train_test_IQA.py \
  --dataset koniq-10k \
  --epochs 10 \
  --batch_size 96 \
  --train_patch_num 20 \
  --test_patch_num 20

# 暂时禁用 ColorJitter
echo "Disabling ColorJitter for Ablation 1-2..."
cp data_loader.py data_loader_backup.py
sed -i 's/torchvision.transforms.ColorJitter/#torchvision.transforms.ColorJitter/' data_loader.py

# 消融1
echo "Running Ablation 1 (Swin-Base Basic)..."
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0 \
  --lr 1e-5 \
  --drop_path_rate 0.1 \
  --dropout_rate 0.2 \
  --lr_scheduler none \
  --test_random_crop \
  --no_spaq

# 消融2
echo "Running Ablation 2 (+ Ranking Loss)..."
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --lr 1e-5 \
  --drop_path_rate 0.1 \
  --dropout_rate 0.2 \
  --lr_scheduler none \
  --test_random_crop \
  --no_spaq

# 恢复 ColorJitter
echo "Enabling ColorJitter for Ablation 3-5..."
cp data_loader_backup.py data_loader.py

# 消融3
echo "Running Ablation 3 (+ ColorJitter)..."
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --lr 1e-5 \
  --drop_path_rate 0.1 \
  --dropout_rate 0.2 \
  --lr_scheduler none \
  --test_random_crop \
  --no_spaq

# 消融4
echo "Running Ablation 4 (+ Strong Regularization)..."
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --lr 1e-5 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler none \
  --test_random_crop \
  --no_spaq

# 消融5（最佳）
echo "Running Ablation 5 (+ Cosine LR - BEST)..."
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --lr 5e-6 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq

echo "All ablation experiments completed!"
```

**使用方法**：
```bash
chmod +x run_ablations.sh
./run_ablations.sh
```

---

**文档版本**: 1.0  
**最后更新**: December 20, 2025  
**预计总实验时间**: 约 24-30 小时（单 GPU）

**重要提醒**：
1. ✅ 运行前检查 `data_loader.py` 中 ColorJitter 设置
2. ✅ 确认 weight_decay 参数（代码中或命令行）
3. ✅ 验证随机种子已设置（train_swin.py）
4. ✅ 每个实验结束后记录结果到表格

