# Loss Function Comparison Experiments

**Date**: 2025-12-22  
**Purpose**: Compare 5 different loss functions for IQA training  
**Configuration**: Full model (Multi-scale + Attention), batch_size=32, epochs=5, no ColorJitter

---

## 📐 Loss Functions Implemented

### 1. **L1 Loss (MAE)** - Current Baseline ⭐
```
L_MAE = (1/N) Σ |Q_i - Q̂_i|
```
- **Status**: ✅ Already tested - SRCC **0.9354**
- **Characteristics**: Robust to outliers, simple, smooth gradients
- **Pros**: Well-established, stable training
- **Cons**: Doesn't directly optimize ranking

---

### 2. **L2 Loss (MSE)**
```
L_MSE = (1/N) Σ ||y_n - ŷ_n||²
```
- **Status**: ⏳ TO RUN
- **Characteristics**: Penalizes large errors more, smooth everywhere
- **Pros**: Differentiable at zero, faster convergence sometimes
- **Cons**: Sensitive to outliers
- **Expected**: SRCC ~0.930-0.935 (slightly worse due to outliers)

---

### 3. **SRCC Loss** (Spearman Correlation)
```
L_SRCC = 1 - Σ(v_n - v̄)(p_n - p̄) / √[Σ(v_n - v̄)² · Σ(p_n - p̄)²]
```
- **Status**: ⏳ TO RUN
- **Characteristics**: Directly optimizes the evaluation metric
- **Pros**: Target-metric optimization, ranking-aware
- **Cons**: May have unstable gradients, batch-dependent
- **Expected**: SRCC ~0.935-0.940 (potentially better, direct optimization)

---

### 4. **Rank Loss** (Pairwise Ranking)
```
L_rank^ij = max(0, |Q̂_i - Q̂_j| - e(Q̂_i, Q̂_j) · (Q_i - Q_j))
where e(Q̂_i, Q̂_j) = {1 if Q̂_i ≥ Q̂_j, -1 if Q̂_i < Q̂_j}
```
- **Status**: ⏳ TO RUN
- **Characteristics**: Explicitly learns relative ordering
- **Pros**: Direct ranking supervision, pairwise comparisons
- **Cons**: O(N²) complexity, slower training
- **Expected**: SRCC ~0.933-0.938 (good for ranking)

---

### 5. **Pairwise Fidelity Loss**
```
p^pred(A>B) = Φ((μ_A^pred - μ_B^pred) / √((σ_A^pred)² + (σ_B^pred)²))
L_fd = 1 - √[p(A>B)·p^pred(A>B)] - √[(1-p(A>B))·(1-p^pred(A>B))]
```
- **Status**: ⏳ TO RUN
- **Characteristics**: Considers opinion distribution/uncertainty
- **Pros**: Probabilistic, handles ambiguity
- **Cons**: Complex, requires distribution modeling
- **Expected**: SRCC ~0.932-0.937 (uncertainty-aware)
- **Note**: Simplified implementation (assumes uniform uncertainty)

---

## 🎯 Experiment Configuration

**Base Configuration** (same for all):
- Model: Swin-Base
- Multi-scale: ✅ Enabled
- Attention: ✅ Enabled
- Ranking Loss Alpha: 0 (no legacy ranking loss)
- ColorJitter: ❌ Disabled
- Batch Size: 32
- Epochs: 5
- Patience: 5
- LR: 5e-6
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4

**Only Variable**: `--loss_type` parameter

---

## 📋 Experiment Commands

### F1: L1 Loss (Baseline) ✅
**Status**: COMPLETE - SRCC 0.9354

```bash
# Already done - this is our current baseline
# Log: logs/swin_multiscale_ranking_alpha0_20251222_161625.log
```

---

### F2: L2 Loss (MSE)

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --loss_type l2 \
  --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 \
  --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter
```

**Expected Time**: ~1.7 hours  
**Expected SRCC**: 0.930-0.935

---

### F3: SRCC Loss

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --loss_type srcc \
  --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 \
  --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter
```

**Expected Time**: ~1.7 hours  
**Expected SRCC**: 0.935-0.940 (potentially best)

---

### F4: Rank Loss

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=2 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --loss_type rank \
  --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 \
  --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter
```

**Expected Time**: ~2.0 hours (slower due to pairwise computation)  
**Expected SRCC**: 0.933-0.938

---

### F5: Pairwise Fidelity Loss

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=3 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --loss_type pairwise \
  --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 \
  --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter
```

**Expected Time**: ~2.0 hours  
**Expected SRCC**: 0.932-0.937

---

## 🚀 Execution Strategy

### Option 1: All at Once (4 GPUs)
Run F2, F3, F4, F5 simultaneously on different GPUs.
**Total Time**: ~2 hours

```bash
# Terminal 1 (GPU 0) - L2
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --loss_type l2 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter

# Terminal 2 (GPU 1) - SRCC
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --loss_type srcc --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter

# Terminal 3 (GPU 2) - Rank
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=2 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --loss_type rank --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter

# Terminal 4 (GPU 3) - Pairwise
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=3 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --loss_type pairwise --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter
```

### Option 2: Sequential (1 GPU)
**Total Time**: ~7.4 hours

### Option 3: Priority Order (2 GPUs)
1. **First batch**: L2 + SRCC (~1.7h)
2. **Second batch**: Rank + Pairwise (~2h)
**Total Time**: ~3.7 hours

---

## 📊 Results Summary Table

| Loss Type | SRCC | PLCC | Δ SRCC | Time | Status | Notes |
|-----------|------|------|--------|------|--------|-------|
| **L1 (MAE)** | **0.9354** | 0.9448 | - | 1.7h | ✅ | Baseline |
| L2 (MSE) | - | - | - | - | ⏳ | Expected: slightly worse |
| SRCC | - | - | - | - | ⏳ | Expected: potentially best |
| Rank | - | - | - | - | ⏳ | Expected: good for ranking |
| Pairwise | - | - | - | - | ⏳ | Expected: uncertainty-aware |

---

## 🔬 Expected Findings

### Hypothesis 1: SRCC Loss Best
**Reason**: Directly optimizes the evaluation metric  
**Prediction**: SRCC Loss ≥ 0.9360

### Hypothesis 2: L1 vs L2
**Reason**: IQA scores may have outliers  
**Prediction**: L1 > L2 (by ~0.002-0.005)

### Hypothesis 3: Ranking-aware Losses
**Reason**: IQA is fundamentally a ranking problem  
**Prediction**: SRCC ≈ Rank > L1 > L2 > Pairwise

### Hypothesis 4: Training Stability
**Reason**: Batch-level vs sample-level optimization  
**Prediction**: L1 = L2 (most stable) > SRCC > Rank = Pairwise

---

## 📝 Key Questions to Answer

1. **Which loss function achieves the highest SRCC?**
2. **Is directly optimizing SRCC better than L1?**
3. **Do ranking-aware losses (SRCC, Rank) outperform point-wise losses (L1, L2)?**
4. **What's the trade-off between complexity and performance?**
5. **Should we use pairwise losses for IQA?**

---

## 🎯 For the Paper

This experiment will provide:
1. **Loss function ablation**: Quantify impact of loss choice
2. **Theoretical justification**: Why L1 is good (or if others are better)
3. **Practical insights**: Which loss to recommend
4. **Comparison table**: Easy visualization for readers

---

## ⏱️ Time Estimates

- **Per Experiment**: 1.7-2.0 hours
- **Total (4 new experiments)**: 7.4 hours sequential, 2 hours parallel (4 GPUs)
- **Total with B1/B2**: ~4 hours parallel (2 loss + 2 model size)

---

## 🚦 Priority Recommendation

### High Priority (Do First):
- **F2 (L2)**: Quick baseline comparison
- **F3 (SRCC)**: Most promising theoretically

### Medium Priority:
- **F4 (Rank)**: Interesting for ranking perspective

### Low Priority (Optional):
- **F5 (Pairwise)**: Complex, may not improve much

---

## 📌 Notes

1. All implementations carefully follow the formulas provided
2. L1 is the default (backward compatible)
3. Can combine with legacy ranking_loss_alpha if needed
4. SRCC and Rank losses may be slower due to batch-level computation
5. Pairwise loss uses simplified uncertainty (uniform assumption)

