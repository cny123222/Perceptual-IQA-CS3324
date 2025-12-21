# Ablation Experiments - Component-by-Component Analysis

**Last Updated**: Dec 21, 2025

**Strategy**: Remove one component at a time from the full model to measure its contribution

---

## 📊 Model Evolution: Baseline → Final

### Original Baseline (ResNet-50)

**Architecture**: HyperIQA with ResNet-50

**Components**:
1. ✅ Backbone: ResNet-50 (23M params)
2. ✅ Feature Extraction: Single-scale (last layer only)
3. ❌ Feature Fusion: N/A (only one scale)
4. ✅ Loss Function: L1 (MAE) only
5. ✅ Regularization: Basic (weight_decay=1e-4, no dropout, no drop_path)

**Performance**: SRCC 0.9009, PLCC 0.9170

---

### Our Final Model (Best)

**Architecture**: HyperIQA with Swin Transformer Base + Enhancements

**Components**:
1. ✅ Backbone: **Swin Transformer Base** (88M params) ← **UPGRADED**
2. ✅ Feature Extraction: **Multi-scale** (4 layers) ← **ADDED**
3. ✅ Feature Fusion: **Attention-based** weighted fusion ← **ADDED**
4. ✅ Loss Function: **L1 + Ranking Loss** (α=0.5) ← **ADDED**
5. ✅ Regularization: **Strong** (wd=2e-4, dropout=0.4, drop_path=0.3) ← **ENHANCED**

**Performance**: SRCC 0.9343, PLCC 0.9463

**Total Improvement**: **+3.47% SRCC** (0.9009 → 0.9343)

---

## 🔬 Added Components Summary

Compared to the ResNet-50 baseline, we added/upgraded 4 key components:

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | **Swin Transformer Backbone** | Upgrade | ResNet-50 → Swin-Base (23M → 88M params) |
| 2 | **Multi-scale Features** | Added | Single-scale → 4-scale hierarchical features |
| 3 | **Attention Fusion** | Added | Simple concat → Attention-weighted fusion |
| 4 | **Ranking Loss** | Added | L1 only → L1 + Ranking Loss (α=0.5) |
| 5 | **Strong Regularization** | Enhanced | Basic → Strong (dropout, drop_path, weight_decay) |

**Note**: Component #1 (Backbone) is evaluated through model size comparison (Tiny/Small/Base), not ablation.

---

## 🎯 Ablation Experiment Design

**Methodology**: Subtractive approach - start with full model, remove ONE component at a time

**Fair Comparison**: All experiments run for the same duration (5 epochs, ~1.5 hours each)

---

## Experiment 0: Full Model (Baseline) ✅ **DONE**

**Configuration**: Base + Multi-scale + Attention + Ranking Loss + Strong Reg

**All Components Enabled**:
- ✅ Swin-Base backbone (88M)
- ✅ Multi-scale features (4 layers)
- ✅ Attention fusion
- ✅ Ranking Loss (α=0.5)
- ✅ Strong regularization (wd=2e-4, dp=0.3, do=0.4)

**Results**:
- SRCC: **0.9343**
- PLCC: **0.9463**
- Log: `logs/swin_multiscale_ranking_alpha0.5_20251221_155013.log`

**Status**: ✅ Completed

---

## Ablation 1: Remove Attention Fusion ✅ **DONE**

**What's removed**: Attention-based feature fusion

**What remains**:
- ✅ Swin-Base backbone
- ✅ Multi-scale features
- ❌ ~~Attention fusion~~ → Simple concatenation
- ✅ Ranking Loss (α=0.5)
- ✅ Strong regularization

**Results** (Round 1):
- SRCC: **0.9316** (-0.0027, -0.27%)
- PLCC: **0.9450** (-0.0013, -0.13%)
- Log: `logs/swin_multiscale_ranking_alpha0.5_20251221_003537.log`

**Component Contribution**: Attention fusion provides **+0.27% SRCC**

**Status**: ✅ Completed

---

## Ablation 2: Remove Ranking Loss ⏰ **TODO**

**What's removed**: Ranking Loss (use L1 loss only)

**What remains**:
- ✅ Swin-Base backbone
- ✅ Multi-scale features
- ✅ Attention fusion
- ❌ ~~Ranking Loss~~ → L1 loss only (α=0)
- ✅ Strong regularization

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
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/ablation_remove_ranking_loss.log
```

**Expected Result**: SRCC ≈ 0.9310-0.9320 (-0.2-0.3%)

**Time**: ~1.5 hours

**Status**: ⏰ TODO

---

## Ablation 3: Remove Strong Regularization ⏰ **TODO**

**What's removed**: Strong regularization (use basic/weak regularization)

**What remains**:
- ✅ Swin-Base backbone
- ✅ Multi-scale features
- ✅ Attention fusion
- ✅ Ranking Loss (α=0.5)
- ❌ ~~Strong regularization~~ → Weak regularization

**Changes**:
- `weight_decay`: 2e-4 → 1e-4 (50% reduction)
- `drop_path_rate`: 0.3 → 0.2 (33% reduction)
- `dropout_rate`: 0.4 → 0.3 (25% reduction)

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
  --attention_fusion \
  --lr 5e-6 \
  --weight_decay 1e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/ablation_remove_strong_reg.log
```

**Expected Result**: SRCC ≈ 0.9310-0.9320 (-0.2-0.3%), may show overfitting

**Time**: ~1.5 hours

**Status**: ⏰ TODO

---

## Ablation 4: Remove Multi-scale Features 🔵 **OPTIONAL**

**What's removed**: Multi-scale feature extraction (use single-scale)

**What remains**:
- ✅ Swin-Base backbone
- ❌ ~~Multi-scale features~~ → Single-scale (last layer only)
- ✅ Attention fusion
- ✅ Ranking Loss (α=0.5)
- ✅ Strong regularization

**Implementation Note**: Requires code modification to add `--no_multiscale` or `--single_scale` parameter

**Alternative Evidence**:
- We already tested this on Swin-Tiny:
  - Single-scale: SRCC 0.9154
  - Multi-scale: SRCC 0.9236
  - **Contribution: +0.82%**

**Expected Result**: SRCC ≈ 0.9260 (-0.8%)

**Status**: 🔵 OPTIONAL (we already have evidence from Tiny model)

---

## 📊 Model Size Comparison (Architecture Selection)

**Goal**: Justify the choice of Swin-Base backbone

These are NOT ablation experiments, but architecture selection experiments:

### Tiny vs Small vs Base

| Model | Params | FLOPs | SRCC | PLCC | Status |
|-------|--------|-------|------|------|--------|
| **Swin-Tiny** | 28M | 4.5G | 0.9236 | 0.9361 | ✅ Done |
| **Swin-Small** | 50M | 8.7G | 0.9303 | 0.9444 | ✅ Done |
| **Swin-Base** | 88M | 15.3G | **0.9343** | **0.9463** | ✅ Done |

**Configuration**: All use Multi-scale + Ranking Loss (α=0.5) + Strong Reg

**Conclusion**: 
- Performance scales with model size
- Base provides best performance with acceptable complexity
- Diminishing returns: Tiny→Small (+0.67%), Small→Base (+0.40%)

---

## 📈 Results Summary Table

| Experiment | Configuration | SRCC | PLCC | SRCC Δ | Component Contribution |
|-----------|---------------|------|------|--------|------------------------|
| **Exp 0: Full Model** | All components | **0.9343** | **0.9463** | - | Baseline |
| **Abl 1: - Attention** | Remove Attention | **0.9316** | **0.9450** | **-0.0027** | **Attention: +0.27%** |
| **Abl 2: - Ranking** | Remove Ranking Loss | ? | ? | ? | Ranking Loss: ? |
| **Abl 3: - Strong Reg** | Remove Strong Reg | ? | ? | ? | Strong Reg: ? |
| **Abl 4: - Multi-scale** | Remove Multi-scale | ~0.9260 | ~0.9400 | ~-0.0083 | Multi-scale: ~+0.83% |

**Note**: Abl 4 is optional since we have evidence from Tiny model

---

## 🎯 Execution Plan

### Phase 1: Core Ablations (Must Do) ⭐⭐⭐⭐⭐

Run these 2 experiments in parallel:

**Terminal 1: Ablation 2 (Remove Ranking Loss)**
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 10 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/ablation_remove_ranking_loss.log
```

**Terminal 2: Ablation 3 (Remove Strong Regularization)**
```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 10 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.5 --attention_fusion --lr 5e-6 --weight_decay 1e-4 --drop_path_rate 0.2 --dropout_rate 0.3 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/ablation_remove_strong_reg.log
```

**Time**: 1.5 hours (parallel execution)

---

### Phase 2: Optional Multi-scale Ablation 🔵

**Only if time permits** (we already have strong evidence from Tiny model):

```bash
# Requires adding --single_scale parameter to train_swin.py first
cd /root/Perceptual-IQA-CS3324 && python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 10 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.5 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --single_scale 2>&1 | tee logs/ablation_remove_multiscale.log
```

---

## ⏱️ Time Estimate

- **Phase 1 (Must Do)**: 2 experiments × 1.5 hours = **1.5 hours** (parallel)
- **Phase 2 (Optional)**: 1 experiment × 1.5 hours = **1.5 hours**

**Total Core Work**: **1.5 hours** (2 experiments in parallel)

**Total with Optional**: **3 hours** (if running all sequentially)

---

## 💡 Why This Design?

### ✅ Advantages of One-Component-at-a-Time:

1. **Clear Attribution**: Each experiment measures exactly one component's contribution
2. **No Interaction Effects**: Removing two components simultaneously makes it hard to separate their individual contributions
3. **Standard Practice**: This is the standard ablation study methodology in research papers
4. **Fair Comparison**: All experiments use the same base configuration except for the removed component

### ❌ Why Not Remove Multiple Components?

Example: "Remove Attention + Ranking Loss" experiment would measure:
- Attention contribution?
- Ranking Loss contribution?
- Their interaction effect?

We can't separate these! Better to measure each individually.

---

## 📝 Expected Final Results

After completing all ablations, we'll have:

| Component | Individual Contribution | Importance |
|-----------|------------------------|------------|
| **Swin-Base Backbone** | +3.34% (vs ResNet-50) | 🥇 Critical |
| **Multi-scale Features** | +0.83% (from Tiny evidence) | 🥈 Very Important |
| **Ranking Loss** | ~+0.3% (estimated) | 🥉 Important |
| **Attention Fusion** | +0.27% (measured) | 🏅 Important |
| **Strong Regularization** | ~+0.2% (estimated) | 🏅 Important |

**Total**: +3.47% SRCC improvement over ResNet-50 baseline

---

## 🚀 Ready to Start?

All commands are ready for copy-paste execution. Start with Phase 1 (2 experiments in parallel, 1.5 hours total).

**Next Step**: Run the two commands from Phase 1 in separate terminals!

