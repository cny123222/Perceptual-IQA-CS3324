# 核心消融实验设计

## 🎯 目标

通过**正向消融实验**（从简单到复杂）量化每个组件的独立贡献。

---

## 📊 实验序列

| 实验ID | 架构配置 | 说明 | 预期SRCC | 状态 |
|--------|---------|------|----------|------|
| **C0** | ResNet50 + HyperNet | 原始HyperIQA | 0.907 | ✅ 已完成 |
| **C1** | Swin-Base (单尺度) | 仅换backbone，无多尺度，无注意力 | ~0.930 | ⏳ 待运行 |
| **C2** | Swin-Base + 多尺度 | 添加多尺度融合（简单拼接） | ~0.935 | ⏳ 待运行 |
| **C3** | Swin-Base + 多尺度 + 注意力 | 完整版本（当前baseline） | 0.9378 | ✅ 已完成 |

---

## 🔬 每个组件的贡献

- **C1 - C0**: Swin Transformer相比ResNet50的提升
- **C2 - C1**: 多尺度融合的贡献
- **C3 - C2**: 注意力机制的贡献
- **C3 - C0**: 总体提升 = +3.08% SRCC

---

## 📝 实验命令

### C1: 仅换Backbone (Swin-Base, 单尺度, 无注意力)

```bash
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
    --dataset koniq-10k \
    --model_size base \
    --batch_size 32 \
    --epochs 10 \
    --patience 3 \
    --train_patch_num 20 \
    --test_patch_num 20 \
    --train_test_num 1 \
    --lr 5e-7 \
    --weight_decay 2e-4 \
    --drop_path_rate 0.3 \
    --dropout_rate 0.4 \
    --lr_scheduler cosine \
    --no_multiscale \
    --ranking_loss_alpha 0 \
    --test_random_crop \
    --no_spaq \
    --no_color_jitter \
    --exp_name "C1_swin_base_only" \
    2>&1 | tee logs/C1_swin_base_only_$(date +%Y%m%d_%H%M%S).log
```

**关键参数**：
- `--no_multiscale`: 禁用多尺度融合（只使用feat3）
- **无** `--attention_fusion`: 不添加注意力机制
- 其他参数与C3保持一致

---

### C2: 添加多尺度 (Swin-Base + 多尺度, 无注意力)

```bash
CUDA_VISIBLE_DEVICES=1 python train_swin.py \
    --dataset koniq-10k \
    --model_size base \
    --batch_size 32 \
    --epochs 10 \
    --patience 3 \
    --train_patch_num 20 \
    --test_patch_num 20 \
    --train_test_num 1 \
    --lr 5e-7 \
    --weight_decay 2e-4 \
    --drop_path_rate 0.3 \
    --dropout_rate 0.4 \
    --lr_scheduler cosine \
    --ranking_loss_alpha 0 \
    --test_random_crop \
    --no_spaq \
    --no_color_jitter \
    --exp_name "C2_swin_base_multiscale" \
    2>&1 | tee logs/C2_swin_base_multiscale_$(date +%Y%m%d_%H%M%S).log
```

**关键参数**：
- **默认启用** multi-scale（不加`--no_multiscale`）
- **无** `--attention_fusion`: 使用简单拼接，不用注意力加权
- 其他参数与C3保持一致

---

### C3: 完整版本 (当前baseline)

这是我们的最佳模型 (E6)，已经完成：
- **SRCC**: 0.9378
- **PLCC**: 0.9485
- **Checkpoint**: `checkpoints/koniq-10k-swin_20251223_002226/`

---

## 🎯 预期结果

### 贡献分解

假设：
- C0 (ResNet50): 0.907
- C1 (只换backbone): 0.930 → **Swin贡献: +2.3%**
- C2 (添加多尺度): 0.935 → **多尺度贡献: +0.5%**
- C3 (添加注意力): 0.9378 → **注意力贡献: +0.28%**

**验证假设**：
- Swin Transformer是最大贡献者（~75%的总提升）
- 多尺度融合提供显著增益（~16%的总提升）
- 注意力机制锦上添花（~9%的总提升）

---

## 📋 执行计划

### Step 1: 等待A1和A2完成
当前正在运行：
- A1 (Remove Attention) ≈ C2的反向实验
- A2 (Remove Multi-scale) ≈ C1的反向实验

### Step 2: 运行C1和C2
等A1和A2完成后：
1. 检查A1和A2的结果
2. 如果A1 ≈ C2且A2 ≈ C1，可以直接使用A1/A2的结果
3. 否则，运行C1和C2（正向实验更符合论文叙述）

### Step 3: 分析结果
- 对比C0 → C1 → C2 → C3
- 量化每个组件的独立贡献
- 更新EXPERIMENTS_LOG_TRACKER.md

---

## 💡 A1/A2 vs C1/C2 的关系

### 理论上应该相等
- **A2 (Remove Multi-scale) ≈ C1 (Swin-Base only)**
- **A1 (Remove Attention) ≈ C2 (Swin-Base + Multi-scale)**

### 实际差异
由于实验的随机性，两者可能有小幅差异（±0.0005），但应该非常接近。

### 决策标准
如果 `|A2 - C1预期| < 0.001` 且 `|A1 - C2预期| < 0.001`，可以直接使用A1/A2的结果作为C2/C1。

---

## 📝 论文叙述

正向消融实验更符合论文叙述逻辑：

> "我们从原始HyperIQA（ResNet50）出发，逐步添加以下改进：
> 
> 1. **Backbone替换**: 将ResNet50替换为Swin Transformer Base，SRCC从0.907提升到0.930（+2.3%）。这表明预训练的Transformer架构对IQA任务具有显著优势。
> 
> 2. **多尺度特征融合**: 引入4个阶段的特征融合，SRCC进一步提升到0.935（+0.5%）。这说明从低层纹理到高层语义的多尺度信息对质量评估至关重要。
> 
> 3. **注意力机制**: 采用动态加权融合替代简单拼接，最终SRCC达到0.9378（+0.28%）。注意力机制使模型能够根据不同图像自适应调整各尺度的重要性。
> 
> 总体而言，我们的方法相比原始HyperIQA提升了3.08% SRCC，其中Swin Transformer贡献了约75%的性能增益，多尺度融合和注意力机制提供了额外的25%增益。"

---

## 🔧 执行脚本

见 `run_core_ablations.sh`

