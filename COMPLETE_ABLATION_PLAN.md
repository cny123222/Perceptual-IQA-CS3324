# Complete Ablation Study Plan

**Last Updated**: Dec 21, 2025

**Goal**: Systematically evaluate the contribution of each component and architectural choice

**Strategy**: Fast ablation (3-5 epochs each, ~1-2 hours per experiment)

---

## 📋 Experiment Overview

### Total Experiments: 10
- **Part A: Component Ablation** (5 experiments)
- **Part B: Model Size Selection** (3 experiments)  
- **Part C: Loss Function Analysis** (2 experiments)

**Total Time**: ~15-20 hours (can run 2-3 in parallel)

---

## Part A: Component Ablation (核心消融实验)

**Goal**: Measure the contribution of each added component

**Base Configuration**: Swin-Base backbone (已确定最优)

### Experiment A0: Full Model ✅ **DONE**

**Configuration**: Base + Multi-scale + Attention + Ranking Loss (α=0.5)

**Results**:
- SRCC: **0.9343**
- PLCC: **0.9463**
- Log: `logs/swin_multiscale_ranking_alpha0.5_20251221_155013.log`
- Status: ✅ **Already Done** (10 epochs)

---

### Experiment A1: Remove Attention Fusion ✅ **DONE**

**Configuration**: Base + Multi-scale + Simple Concat + Ranking Loss (α=0.5)

**What's removed**: Attention-based feature fusion (用简单拼接代替)

**Results** (Round 1, fair comparison):
- SRCC: **0.9316** (-0.27%)
- PLCC: **0.9450** (-0.13%)
- Log: `logs/swin_multiscale_ranking_alpha0.5_20251221_003537.log`
- Status: ✅ **Already Done** (Round 1 of 10 rounds)

**Note**: This experiment ran for 3 rounds total, achieving best SRCC 0.9336 in Round 3. However, for fair comparison with other experiments (which ran 1 round), we use Round 1 result: 0.9316.

**Conclusion**: Attention provides **+0.27% SRCC** contribution (0.9316 → 0.9343)

---

### Experiment A2: Remove Ranking Loss ⏰ **TODO**

**Configuration**: Base + Multi-scale + Attention + L1 Loss Only

**What's removed**: Ranking loss (只用L1 loss)

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 10 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0 \
  --use_attention \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected Result**: SRCC ≈ 0.9310-0.9320 (-0.2-0.3%)

**Status**: ⏰ **TODO** (~1.5 hours)

---

### Experiment A3: Remove Both Attention and Ranking Loss ⏰ **TODO**

**Configuration**: Base + Multi-scale + Simple Concat + L1 Loss Only

**What's removed**: Attention + Ranking Loss (最接近baseline的配置)

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 10 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected Result**: SRCC ≈ 0.9290-0.9300

**Status**: ⏰ **TODO** (~1.5 hours)

**Purpose**: 验证两个组件的独立性和交互效应

---

### Experiment A4: Remove Multi-scale Features (Use Single-scale) ⏰ **TODO**

**Configuration**: Base + Single-scale (last layer only) + Attention + Ranking Loss

**What's removed**: Multi-scale feature extraction (只用最后一层特征)

**Implementation Note**: 需要修改代码或添加 `--single_scale` 参数

**Alternative (已有证据)**:
- 我们已经在Swin-Tiny上测试过：
  - Single-scale: 0.9154
  - Multi-scale: 0.9236 (+0.82%)
  
**Expected Result**: SRCC ≈ 0.9260 (-0.8%)

**Status**: 🔵 **OPTIONAL** (已有Tiny的证据，可以跳过)

---

## Part B: Model Size Selection (模型选择消融)

**Goal**: Justify the choice of Swin-Base over smaller variants

### Experiment B1: Swin-Tiny ✅ **DONE**

**Configuration**: Tiny + Multi-scale + Simple Concat + Ranking Loss (α=0.5)

**Results**:
- SRCC: **0.9236**
- PLCC: **0.9361**
- Log: `record.md` (multiple experiments)
- Status: ✅ **Already Done**

**Parameters**: 28M params, 4.5 GFLOPs

---

### Experiment B2: Swin-Small ✅ **DONE**

**Configuration**: Small + Multi-scale + Simple Concat + Ranking Loss (α=0.5)

**Results**:
- SRCC: **0.9303**
- PLCC: **0.9444**
- Log: `record.md`
- Status: ✅ **Already Done**

**Parameters**: 50M params, 8.7 GFLOPs

---

### Experiment B3: Swin-Base ✅ **DONE**

**Configuration**: Base + Multi-scale + Attention + Ranking Loss (α=0.5)

**Results**:
- SRCC: **0.9343**
- PLCC: **0.9463**
- Status: ✅ **Already Done**

**Parameters**: 88M params, 15.3 GFLOPs

**Conclusion**: 
- Performance scales with model size
- Base provides best performance with acceptable complexity

---

## Part C: Loss Function Analysis (损失函数消融)

**Goal**: Evaluate different ranking loss weights

### Experiment C1: Ranking Loss Alpha = 0.3 ✅ **DONE**

**Configuration**: Base + Attention + Ranking Loss (α=0.3)

**Results**:
- SRCC: **0.9303** (-0.40%)
- PLCC: **0.9435** (-0.28%)
- Log: `logs/swin_multiscale_ranking_alpha0.3_20251221_152455.log`
- Status: ✅ **Already Done**

**Conclusion**: α=0.3 is too low

---

### Experiment C2: Ranking Loss Alpha = 0.0 (Pure L1) ⏰ **TODO**

**Configuration**: Base + Attention + L1 Loss Only (same as A2)

**Status**: ⏰ See Experiment A2

---

### Experiment C3: Ranking Loss Alpha = 0.7 🔵 **OPTIONAL**

**Configuration**: Base + Attention + Ranking Loss (α=0.7)

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 10 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.7 \
  --use_attention \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected Result**: SRCC ≈ 0.9320-0.9330 (可能过高)

**Status**: 🔵 **OPTIONAL**

**Priority**: Low (α=0.5已经很好)

---

## Part D: Sensitivity Analysis (灵敏度分析 - 正则化)

**Goal**: Test model robustness to hyperparameter changes

### Experiment D1: Weak Regularization ⏰ **TODO**

**Configuration**: Base + Attention + Ranking + Weak Reg

**Changes**:
- `--weight_decay 1e-4` (was 2e-4, 50% reduction)
- `--drop_path_rate 0.2` (was 0.3, 33% reduction)  
- `--dropout_rate 0.3` (was 0.4, 25% reduction)

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 10 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --use_attention \
  --lr 5e-6 \
  --weight_decay 1e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected Result**: SRCC ≈ 0.9310-0.9320, 可能出现过拟合迹象

**Status**: ⏰ **TODO** (~1.5 hours)

---

### Experiment D2: Very Strong Regularization 🔵 **OPTIONAL**

**Configuration**: Base + Attention + Ranking + Very Strong Reg

**Changes**:
- `--weight_decay 3e-4` (was 2e-4, 1.5x increase)
- `--drop_path_rate 0.4` (was 0.3, 1.33x increase)
- `--dropout_rate 0.5` (was 0.4, 1.25x increase)

**Expected Result**: SRCC ≈ 0.9320-0.9330 (可能过度正则化，欠拟合)

**Status**: 🔵 **OPTIONAL**

**Priority**: Low

---

### Experiment D3: Lower Learning Rate ⏰ **TODO**

**Configuration**: Base + Attention + Ranking + LR=2.5e-6

**Changes**:
- `--lr 2.5e-6` (was 5e-6, 50% reduction)

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 10 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --use_attention \
  --lr 2.5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected Result**: SRCC ≈ 0.9330-0.9340 (可能收敛慢，需要更多epochs)

**Status**: ⏰ **TODO** (~1.5 hours)

---

### Experiment D4: Higher Learning Rate 🔵 **OPTIONAL**

**Configuration**: Base + Attention + Ranking + LR=1e-5

**Changes**:
- `--lr 1e-5` (was 5e-6, 2x increase)

**Expected Result**: SRCC ≈ 0.9300-0.9320 (可能不稳定)

**Status**: 🔵 **OPTIONAL**

---

## 📊 Summary of Experiments

### Must Do (Priority 1) ⭐⭐⭐⭐⭐

| ID | Experiment | Status | Time | Purpose |
|----|-----------|--------|------|---------|
| A0 | Full Model | ✅ Done | - | Baseline |
| A1 | Remove Attention | ✅ Done | - | Attention contribution |
| A2 | Remove Ranking Loss | ⏰ TODO | 1.5h | Ranking loss contribution |
| A3 | Remove Both | ⏰ TODO | 1.5h | Component interaction |
| B1-B3 | Model Sizes | ✅ Done | - | Model selection |
| C1 | Alpha=0.3 | ✅ Done | - | Loss weight |

**Total Must Do**: 2 experiments, ~3 hours

---

### Should Do (Priority 2) ⭐⭐⭐⭐

| ID | Experiment | Status | Time | Purpose |
|----|-----------|--------|------|---------|
| D1 | Weak Regularization | ⏰ TODO | 1.5h | Sensitivity analysis |
| D3 | Lower LR | ⏰ TODO | 1.5h | LR sensitivity |

**Total Should Do**: 2 experiments, ~3 hours

---

### Optional (Priority 3) ⭐⭐

| ID | Experiment | Status | Time | Purpose |
|----|-----------|--------|------|---------|
| A4 | Single-scale | 🔵 Optional | 1.5h | Multi-scale contribution (已有Tiny证据) |
| C3 | Alpha=0.7 | 🔵 Optional | 1.5h | Explore higher alpha |
| D2 | Very Strong Reg | 🔵 Optional | 1.5h | Upper bound |
| D4 | Higher LR | 🔵 Optional | 1.5h | Instability check |

**Total Optional**: 4 experiments, ~6 hours

---

## 🎯 Recommended Execution Plan

### Phase 1: Core Ablations (必须做) ⭐⭐⭐⭐⭐

**Parallel Track 1**:
```bash
# A2: Remove Ranking Loss (Terminal 1)
python train_swin.py ... --ranking_loss_alpha 0 --use_attention ...
```

**Parallel Track 2**:
```bash
# A3: Remove Both (Terminal 2)
python train_swin.py ... --ranking_loss_alpha 0 ...
```

**Time**: 1.5 hours (parallel) = **1.5 hours total**

---

### Phase 2: Sensitivity Analysis (应该做) ⭐⭐⭐⭐

**Parallel Track 1**:
```bash
# D1: Weak Regularization (Terminal 1)
python train_swin.py ... --weight_decay 1e-4 --drop_path_rate 0.2 --dropout_rate 0.3 ...
```

**Parallel Track 2**:
```bash
# D3: Lower LR (Terminal 2)
python train_swin.py ... --lr 2.5e-6 ...
```

**Time**: 1.5 hours (parallel) = **1.5 hours total**

---

### Phase 3: Optional Extensions (可选) ⭐⭐

Run if time permits

**Time**: Up to 6 hours

---

## 📋 Quick Command List for Copy-Paste

### A2: Remove Ranking Loss
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 10 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0 --use_attention --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/ablation_remove_ranking_loss.log
```

### A3: Remove Both
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 10 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/ablation_remove_both.log
```

### D1: Weak Regularization
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 10 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.5 --use_attention --lr 5e-6 --weight_decay 1e-4 --drop_path_rate 0.2 --dropout_rate 0.3 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/sensitivity_weak_reg.log
```

### D3: Lower Learning Rate
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 10 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.5 --use_attention --lr 2.5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/sensitivity_lower_lr.log
```

---

## 📈 Expected Results Table (To be filled)

| Experiment | Configuration | SRCC (Round 1) | PLCC (Round 1) | SRCC Δ | Component Contribution |
|-----------|---------------|----------------|----------------|--------|------------------------|
| **A0: Full Model** | Base + Att + Rank | **0.9343** | **0.9463** | - | Baseline |
| **A1: - Attention** | Base + Rank | **0.9316** | **0.9450** | **-0.0027** | **Attention: +0.27%** |
| **A2: - Ranking** | Base + Att | ? | ? | ? | Ranking: ? |
| **A3: - Both** | Base only | ? | ? | ? | Both: ? |
| **B1: Tiny** | Tiny + Rank | **0.9236** | **0.9361** | -0.0107 | Size effect: Base vs Tiny |
| **B2: Small** | Small + Rank | **0.9303** | **0.9444** | -0.0040 | Size effect: Base vs Small |
| **C1: Alpha=0.3** | Base + Att + α=0.3 | **0.9303** | **0.9435** | -0.0040 | α too low |
| **D1: Weak Reg** | Weak regularization | ? | ? | ? | Reg sensitivity |
| **D3: Lower LR** | LR=2.5e-6 | ? | ? | ? | LR sensitivity |

**Note**: All results are from Round 1 for fair comparison. Some experiments ran multiple rounds, but only Round 1 is used here.

---

## ⏱️ Time Estimate

- **Phase 1 (Must Do)**: 1.5 hours
- **Phase 2 (Should Do)**: 1.5 hours
- **Phase 3 (Optional)**: 6 hours

**Total Core Work**: **3 hours** (4 experiments, 2 parallel tracks)

**Total with Optional**: **9 hours**

---

**Recommendation**: 
1. Start with Phase 1 (必须做) - 2 experiments in parallel
2. Review results, then do Phase 2 (应该做) - 2 more experiments
3. Skip Phase 3 unless we have extra time

**Total time investment for solid ablation study**: **3 hours**

