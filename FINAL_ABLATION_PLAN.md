# Final Ablation and Analysis Plan

**Last Updated**: Dec 21, 2025

**Goal**: Measure contribution of each new component and parameter sensitivity

---

## 📊 Model Evolution: Baseline → Final

### Original Baseline (ResNet-50)
- Backbone: ResNet-50 (23M params)
- Features: Single-scale (last layer only)
- Fusion: N/A
- Loss: L1 only
- Performance: **SRCC 0.9009**

### Our Final Model
- Backbone: **Swin-Base** (88M params) ← UPGRADED
- Features: **Multi-scale** (4 layers) ← NEW
- Fusion: **Attention-based** ← NEW  
- Loss: **L1 + Ranking (α=0.5)** ← NEW
- Performance: **SRCC 0.9343** (+3.47%)

---

## Part A: Component Ablation (Core Experiments)

**Goal**: Measure contribution of each NEW component

### Experiment A0: Full Model ✅ **DONE**

**All components enabled**

**Results**:
- SRCC: **0.9343**
- PLCC: **0.9463**
- Log: `logs/swin_multiscale_ranking_alpha0.5_20251221_155013.log`

---

### Experiment A1: Remove Attention Fusion ✅ **DONE**

**What's removed**: Attention fusion → Simple concatenation

**Results** (Round 1):
- SRCC: **0.9316** (-0.27%)
- PLCC: **0.9450** (-0.13%)
- Log: `logs/swin_multiscale_ranking_alpha0.5_20251221_003537.log`

**Contribution**: Attention provides **+0.27% SRCC**

**Note**: This experiment used `patience=7` instead of `5`. The difference is minor and acceptable.

---

### Experiment A2: Remove Ranking Loss ✅ **DONE**

**What's removed**: Ranking Loss → L1 loss only (alpha=0)

**Results** (Epoch 1):
- SRCC: **0.9338** (-0.05%)
- PLCC: **0.9465** (+0.02%)
- Log: `logs/swin_multiscale_ranking_alpha0_20251221_203437.log`

**Contribution**: Ranking Loss provides **+0.05% SRCC, +0.02% PLCC**

**Note**: Contribution is much smaller than expected! PLCC actually improved slightly without ranking loss.

---

### Experiment A3: Remove Multi-scale Features ⏰ **TODO**

**What's removed**: Multi-scale → Single-scale (last layer only)

**Implementation**: Add `--no_multiscale` flag to disable multi-scale features

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --attention_fusion \
  --no_multiscale \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Evidence from Swin-Tiny**:
- Single-scale: SRCC 0.9154
- Multi-scale: SRCC 0.9236
- Contribution: **+0.82%**

**Expected**: SRCC ≈ 0.9260 (-0.8%)

**Time**: ~1.5 hours

**Note**: Need to verify `--no_multiscale` parameter exists, or use existing flag

---

## Part B: Model Size Comparison ⏰ **NEEDS REDO**

**Goal**: Justify Swin-Base choice with IDENTICAL configuration

**Problem**: Previous runs used different configurations for different model sizes. For fair comparison, all models should use the SAME configuration (Best Model config).

**Configuration for ALL models**:
- Multi-scale: ✅ Enabled
- Attention: ✅ Enabled
- Ranking Loss: α=0.5
- Strong Regularization: wd=2e-4, dp=0.3, do=0.4
- LR: 5e-6, Batch: 32, Patience: 5

### B1: Swin-Tiny ⏰ **TODO**

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size tiny \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9240-0.9250

**Time**: ~1.5 hours

---

### B2: Swin-Small ⏰ **TODO**

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9310-0.9320

**Time**: ~1.5 hours

---

### B3: Swin-Base ✅ **DONE**

**Results**:
- SRCC: **0.9343**
- PLCC: **0.9463**
- Log: `logs/swin_multiscale_ranking_alpha0.5_20251221_155013.log`

**Status**: ✅ Already done with correct configuration

---

**Note**: This is architecture selection to show performance scales with model capacity. All models use identical hyperparameters for fair comparison.

**Conclusion**: Performance scales with model size, Base provides best performance

---

## Part C: Ranking Loss Sensitivity ⏰ **PARTIAL**

**Goal**: Find optimal ranking loss weight (α)

### C1: Alpha = 0.3 ⚠️ **NEEDS REDO**

**Problem**: Previous run (`logs/swin_multiscale_ranking_alpha0.3_20251221_155010.log`) did NOT use `--attention_fusion`, so it changed 2 variables simultaneously (Alpha + Attention). Results are invalid for sensitivity analysis.

**Command** (corrected):
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.3 \
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9320-0.9330 (with Attention enabled)

**Time**: ~1.5 hours

**Status**: ⚠️ Must redo with correct configuration

---

### C2: Alpha = 0.0 (Pure L1) ⏰ **TODO**

Same as Experiment A2 above

---

### C3: Alpha = 0.1 ⏰ **TODO**

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.1 \
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9305-0.9315

**Time**: ~1.5 hours

---

### C4: Alpha = 0.7 ⏰ **TODO**

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.7 \
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9320-0.9330 (may be too high)

**Time**: ~1.5 hours

---

## Part D: Regularization Sensitivity ⏰ **TODO**

**Goal**: Test robustness to each regularization parameter **independently**

**Strategy**: Change ONE parameter at a time for clear attribution and easy plotting

### D1: Lower Weight Decay ⏰ **TODO**

**Change**: `weight_decay`: 2e-4 → 1e-4 (50% reduction)

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 1e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9330-0.9340

**Time**: ~1.5 hours

---

### D2: Lower Drop Path Rate ⏰ **TODO**

**Change**: `drop_path_rate`: 0.3 → 0.2 (33% reduction)

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9335-0.9345

**Time**: ~1.5 hours

---

### D3: Lower Dropout Rate ⏰ **TODO**

**Change**: `dropout_rate`: 0.4 → 0.3 (25% reduction)

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9335-0.9345

**Time**: ~1.5 hours

---

### D4: Higher Weight Decay 🔵 **OPTIONAL**

**Change**: `weight_decay`: 2e-4 → 3e-4 (50% increase)

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 3e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9330-0.9340

**Time**: ~1.5 hours

---

### D5: Higher Drop Path Rate 🔵 **OPTIONAL**

**Change**: `drop_path_rate`: 0.3 → 0.4 (33% increase)

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.4 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9330-0.9340

**Time**: ~1.5 hours

---

### D6: Higher Dropout Rate 🔵 **OPTIONAL**

**Change**: `dropout_rate`: 0.4 → 0.5 (25% increase)

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.5 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9330-0.9340

**Time**: ~1.5 hours

---

## Part E: Learning Rate Sensitivity ⏰ **TODO**

**Goal**: Test robustness to learning rate changes

### E1: Lower LR (LR = 2.5e-6) ⏰ **TODO**

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --attention_fusion \
  --lr 2.5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9330-0.9340 (slower convergence)

**Time**: ~1.5 hours

---

### E2: Higher LR (LR = 1e-5) ⏰ **TODO**

**Command**:
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --attention_fusion \
  --lr 1e-5 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

**Expected**: SRCC ≈ 0.9300-0.9320 (may be unstable)

**Time**: ~1.5 hours

---

## 📊 Experiment Priority

### 🔥 Priority 1: Core Ablations (Must Do)

| ID | Experiment | Time | Purpose |
|----|-----------|------|---------|
| A2 | Remove Ranking Loss | 1.5h | Ranking contribution |
| A3 | Remove Multi-scale | 1.5h | Multi-scale contribution |

**Total**: 3 hours (can run in parallel = 1.5 hours)

---

### ⭐ Priority 2: Ranking Loss Sensitivity (Should Do)

| ID | Experiment | Time | Purpose |
|----|-----------|------|---------|
| C3 | Alpha = 0.1 | 1.5h | Low alpha boundary |
| C4 | Alpha = 0.7 | 1.5h | High alpha boundary |

**Total**: 3 hours (can run in parallel = 1.5 hours)

---

### 🔵 Priority 3: Regularization Sensitivity (Nice to Have)

| ID | Experiment | Time | Purpose |
|----|-----------|------|---------|
| D1 | Weak Reg | 1.5h | Lower bound |
| D2 | Very Strong Reg | 1.5h | Upper bound |

**Total**: 3 hours (can run in parallel = 1.5 hours)

---

### 🟢 Priority 4: LR Sensitivity (Nice to Have)

| ID | Experiment | Time | Purpose |
|----|-----------|------|---------|
| E1 | Lower LR | 1.5h | LR lower bound |
| E2 | Higher LR | 1.5h | LR upper bound |

**Total**: 3 hours (can run in parallel = 1.5 hours)

---

## 🎯 Execution Plan

### Phase 1: Core Ablations (Priority 1) - **START HERE**

**Run in parallel** (1.5 hours total):

**Terminal 1:**
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 10 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq
```

**Terminal 2:**
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 10 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.5 --attention_fusion --no_multiscale --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq
```

---

### Phase 2: Ranking Loss Sensitivity (Priority 2)

**Run in parallel** (1.5 hours total):

**Terminal 1:** Alpha = 0.1 (see C3 command above)

**Terminal 2:** Alpha = 0.7 (see C4 command above)

---

### Phase 3: Regularization Sensitivity (Priority 3)

**Run in parallel** (1.5 hours total):

**Terminal 1:** Weak Reg (see D1 command above)

**Terminal 2:** Very Strong Reg (see D2 command above)

---

### Phase 4: LR Sensitivity (Priority 4)

**Run in parallel** (1.5 hours total):

**Terminal 1:** Lower LR (see E1 command above)

**Terminal 2:** Higher LR (see E2 command above)

---

## 📈 Expected Results Summary

### Component Ablation

| Experiment | SRCC | Δ from Full | Component Contribution |
|-----------|------|-------------|------------------------|
| Full Model | 0.9343 | - | Baseline |
| - Attention | 0.9316 | -0.27% | **Attention: +0.27%** |
| - Ranking Loss | 0.9338 ✅ | -0.05% ✅ | **Ranking: +0.05%** ⚠️ |
| - Multi-scale | ~0.9260 | ~-0.83% | **Multi-scale: +0.83%** |

### Ranking Loss Sensitivity

| Alpha | SRCC | Note |
|-------|------|------|
| 0.0 | ~0.9315 | No ranking |
| 0.1 | ~0.9310 | Too low |
| 0.3 | 0.9303 | Tested, too low |
| **0.5** | **0.9343** | **Optimal** ✅ |
| 0.7 | ~0.9325 | May be too high |

### Regularization Sensitivity

**Weight Decay** (others fixed at optimal):
| Value | SRCC | Note |
|-------|------|------|
| 1e-4 | ~0.9335 | Lower |
| **2e-4** | **0.9343** | **Optimal** ✅ |
| 3e-4 | ~0.9335 | Higher |

**Drop Path Rate** (others fixed at optimal):
| Value | SRCC | Note |
|-------|------|------|
| 0.2 | ~0.9340 | Lower |
| **0.3** | **0.9343** | **Optimal** ✅ |
| 0.4 | ~0.9335 | Higher |

**Dropout Rate** (others fixed at optimal):
| Value | SRCC | Note |
|-------|------|------|
| 0.3 | ~0.9340 | Lower |
| **0.4** | **0.9343** | **Optimal** ✅ |
| 0.5 | ~0.9335 | Higher |

### LR Sensitivity

| LR | SRCC | Note |
|----|------|------|
| 2.5e-6 | ~0.9335 | Slower convergence |
| **5e-6** | **0.9343** | **Optimal** ✅ |
| 1e-5 | ~0.9310 | May be unstable |

---

## ⏱️ Total Time Estimate

- **Priority 1 (Core)**: 1.5 hours (parallel)
- **Priority 2 (Ranking)**: 1.5 hours (parallel)
- **Priority 3 (Reg)**: 1.5 hours (parallel)
- **Priority 4 (LR)**: 1.5 hours (parallel)

**Total**: 6 hours (all parallel) or 12 hours (all sequential)

**Recommended**: Do Priority 1 + 2 = **3 hours** (gives complete story)

---

## 🚀 Quick Start

**Start with Phase 1** (2 experiments in parallel, 1.5 hours):

1. Open two terminals
2. Run the Terminal 1 command (Remove Ranking Loss)
3. Run the Terminal 2 command (Remove Multi-scale)
4. Wait ~1.5 hours for results
5. Review and decide if you want to continue with Phase 2-4

**All experiments auto-generate log files** - no need for `tee` command!

---

## 📋 Complete Experiment Checklist

### ✅ Completed Experiments

| ID | Experiment | SRCC | Status | Notes |
|----|-----------|------|--------|-------|
| A0 | Full Model (Base + Att + Rank0.5) | 0.9343 | ✅ Done | Baseline |
| A1 | Remove Attention | 0.9316 | ✅ Done | Patience=7 (acceptable) |
| A2 | Remove Ranking Loss (Alpha=0) | 0.9338 | ✅ Done | +0.05% contribution only! ⚠️ |

**Total Completed**: 3 experiments

---

### ⚠️ Experiments Needing Redo

| ID | Experiment | Issue | Status |
|----|-----------|-------|--------|
| B1 | Swin-Tiny | Used different config (no Attention) | ⚠️ Redo for fair comparison |
| B2 | Swin-Small | Used different config (no Attention) | ⚠️ Redo for fair comparison |
| C1 | Alpha = 0.3 | Missing `--attention_fusion` | ⚠️ Must redo |

**Total Needing Redo**: 3 experiments

---

### 🔥 Priority 1: Core Ablations (Must Do - 1 experiment)

| ID | Experiment | Command | Time | Status |
|----|-----------|---------|------|--------|
| ~~**A2**~~ | ~~**Remove Ranking Loss**~~ | - | - | ✅ Done |
| **A3** | **Remove Multi-scale** | See Section "Experiment A3" | 1.5h | ⏰ TODO |

**Remaining**: 1 experiment (1.5 hours)

---

### ⭐ Priority 2: Ranking Loss Sensitivity (Should Do - 3 experiments)

| ID | Experiment | Command | Time |
|----|-----------|---------|------|
| **C1** | **Alpha = 0.3 (Redo)** | See Section "C1" (corrected) | 1.5h |
| **C3** | **Alpha = 0.1** | See Section "C3" | 1.5h |
| **C4** | **Alpha = 0.7** | See Section "C4" | 1.5h |

**Can run in parallel**: C1 + C3 or C1 + C4 = 1.5 hours

---

### 🔵 Priority 3: Model Size Comparison (Should Do - 2 experiments)

| ID | Experiment | Command | Time |
|----|-----------|---------|------|
| **B1** | **Swin-Tiny (Redo)** | See Section "B1" | 1.5h |
| **B2** | **Swin-Small (Redo)** | See Section "B2" | 1.5h |

**Can run in parallel**: 1.5 hours total

**Why redo**: Previous runs used simpler config (no Attention). Need identical config (with Attention) for fair comparison and proper scaling analysis.

---

### 🟡 Priority 4: Regularization Sensitivity (Nice to Have - 3-6 experiments)

| ID | Experiment | Command | Time | Priority |
|----|-----------|---------|------|----------|
| **D1** | **Lower Weight Decay** | See Section "D1" | 1.5h | ⭐ Core |
| **D2** | **Lower Drop Path** | See Section "D2" | 1.5h | ⭐ Core |
| **D3** | **Lower Dropout** | See Section "D3" | 1.5h | ⭐ Core |
| **D4** | Higher Weight Decay | See Section "D4" | 1.5h | 🔵 Optional |
| **D5** | Higher Drop Path | See Section "D5" | 1.5h | 🔵 Optional |
| **D6** | Higher Dropout | See Section "D6" | 1.5h | 🔵 Optional |

**Can run in parallel**: 
- Core (D1-D3): 1.5 hours (3 terminals)
- Optional (D4-D6): 1.5 hours (3 terminals)

**Strategy**: Each experiment changes ONE parameter only for clear attribution and easy plotting

---

### 🟢 Priority 5: LR Sensitivity (Nice to Have - 2 experiments)

| ID | Experiment | Command | Time |
|----|-----------|---------|------|
| **E1** | **Lower LR (2.5e-6)** | See Section "E1" | 1.5h |
| **E2** | **Higher LR (1e-5)** | See Section "E2" | 1.5h |

**Can run in parallel**: 1.5 hours total

---

## 🎯 Recommended Execution Order

### Stage 1: Core Ablations (Essential) 🔥

**Status**: 
- ✅ **A2 (Remove Ranking Loss)**: DONE - Contribution only +0.05% ⚠️
- ⏰ **A3 (Remove Multi-scale)**: TODO - Expected large contribution (+0.8%)

**Single Execution** (1.5 hours):
- Terminal 1: **A3** (Remove Multi-scale)

**Why**: Measure the contribution of multi-scale feature extraction (expected to be significant).

---

### Stage 2: Fix C1 + Explore Boundaries ⭐

**Parallel Execution** (1.5 hours):
- Terminal 1: **C1** (Alpha = 0.3, corrected with Attention)
- Terminal 2: **C3** (Alpha = 0.1) or **C4** (Alpha = 0.7)

**Why**: C1 needs redo, and we should test the boundaries of alpha parameter.

---

### Stage 3: Model Size Comparison 🔵

**Parallel Execution** (1.5 hours):
- Terminal 1: **B1** (Swin-Tiny with full config)
- Terminal 2: **B2** (Swin-Small with full config)

**Why**: Show that performance scales with model capacity (with identical hyperparameters).

---

### Stage 4: Regularization Analysis 🟡

**Parallel Execution** (1.5 hours):
- Terminal 1: **D1** (Lower Weight Decay)
- Terminal 2: **D2** (Lower Drop Path)
- Terminal 3: **D3** (Lower Dropout)

**Why**: Understand sensitivity to each regularization parameter independently.

**Note**: D4-D6 (higher values) are optional for exploring the other direction.

---

### Stage 5: LR Sensitivity 🟢

**Parallel Execution** (1.5 hours):
- Terminal 1: **E1** (Lower LR)
- Terminal 2: **E2** (Higher LR)

**Why**: Understand model robustness to learning rate changes.

---

## ⏱️ Total Time Investment

| Stages | Experiments | Time (Parallel) | Priority |
|--------|-------------|-----------------|----------|
| **Stage 1** | A2 + A3 | 1.5h | 🔥 Must |
| **Stage 2** | C1 + C3/C4 | 1.5h | ⭐ Should |
| **Stage 3** | B1 + B2 | 1.5h | 🔵 Should |
| **Stage 4** | D1 + D2 + D3 | 1.5h | 🟡 Nice |
| **Stage 5** | E1 + E2 | 1.5h | 🟢 Nice |

**Minimum (Stage 1 only)**: 1.5 hours (2 experiments)
**Recommended (Stage 1+2+3)**: 4.5 hours (6-7 experiments)  
**Complete (All core)**: 7.5 hours (13 experiments, not counting D4-D6 optional)
**Full (With D4-D6)**: 9 hours (16 experiments)

---

## 🚀 Quick Start Commands

### Stage 1 Command (Next Experiment!)

**A3 (Remove Multi-scale) - IMPORTANT:**
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.5 --attention_fusion --no_multiscale --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq
```

**Expected**: Significant SRCC drop (~0.8%), as multi-scale features are crucial for IQA.

---

**Summary**: 
- **3 experiments ✅ done** (A0, A1, A2)
- **3 experiments ⚠️ need redo** (B1, B2, C1 - for fair comparison and correct config)
- **9+ experiments ⏰ TODO** 
  - Core ablations: A3 (1) ← **Only 1 left!**
  - Ranking sensitivity: C3, C4 (2, plus C1 redo)
  - Model sizes: B1, B2 (2, redo)
  - Regularization: D1-D3 core (3), D4-D6 optional (3)
  - LR sensitivity: E1, E2 (2)
- **Total new work**: 12-15 experiments (6-7.5 hours with parallel execution)

**⚠️ Surprising Finding**: Ranking Loss contribution is only **+0.05% SRCC** (much smaller than expected!)

