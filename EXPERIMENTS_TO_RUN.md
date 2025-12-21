# 🔬 Complete Experiment List (Baseline: Alpha=0.3)

**Last Updated**: Dec 22, 2025 00:10

**New Best Model (Baseline)**:
- Configuration: Swin-Base + Multi-scale + Attention + Ranking(α=0.3) + Strong Reg
- Performance: **SRCC 0.9352, PLCC 0.9460** (Epoch 3)
- Checkpoint: `checkpoints/koniq-10k-swin-ranking-alpha0.3_20251221_215124/best_model_srcc_0.9352_plcc_0.9460.pkl`
- Log: `logs/swin_multiscale_ranking_alpha0.3_20251221_215123.log`

---

## 🎯 Experiments Overview

| Stage | Category | # Experiments | Est. Time | Can Parallel |
|-------|----------|---------------|-----------|--------------|
| **Stage 1** | Core Ablations (A1-A3) | 3 | 4.5h | ✅ Yes (3 parallel) |
| **Stage 2** | Ranking Sensitivity (C) | 3 | 4.5h | ✅ Yes (3 parallel) |
| **Stage 3** | Model Size (B) | 2 | 3h | ✅ Yes (2 parallel) |
| **Stage 4** | Regularization (D) | 6 | 9h | ✅ Yes (3 parallel × 2 batches) |
| **Stage 5** | Learning Rate (E) | 2 | 3h | ✅ Yes (2 parallel) |
| **Total** | | **16** | **24h** | **~8-10h with parallelism** |

---

## 📋 Stage 1: Core Ablations (MUST DO - 3 experiments)

**Goal**: Measure contribution of each component against new baseline (α=0.3)

### A1: Remove Attention Fusion

**What changes**: `--attention_fusion` → removed

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
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/ablation_a1_no_attention_alpha0.3_$(date +%Y%m%d_%H%M%S).log
```

**Expected**: SRCC ~0.9325 (loss ~0.27%)

---

### A2: Remove Ranking Loss

**What changes**: `--ranking_loss_alpha 0.3` → `--ranking_loss_alpha 0`

```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
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
  2>&1 | tee logs/ablation_a2_no_ranking_alpha0.3_$(date +%Y%m%d_%H%M%S).log
```

**Expected**: SRCC ~0.9347 (loss ~0.05%)

---

### A3: Remove Multi-scale Features

**What changes**: Add `--no_multiscale` flag

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
  --no_multiscale \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/ablation_a3_no_multiscale_alpha0.3_$(date +%Y%m%d_%H%M%S).log
```

**Expected**: SRCC ~0.9318 (loss ~0.34%)

---

## 📊 Stage 2: Ranking Loss Sensitivity (3 experiments)

**Goal**: Compare different alpha values (baseline is 0.3)

### C1: Alpha = 0.1

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
  --no_spaq \
  2>&1 | tee logs/sensitivity_c1_alpha0.1_$(date +%Y%m%d_%H%M%S).log
```

---

### C2: Alpha = 0.5

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
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/sensitivity_c2_alpha0.5_$(date +%Y%m%d_%H%M%S).log
```

**Note**: We already have this from previous runs (SRCC 0.9343), but re-running for consistency

---

### C3: Alpha = 0.7

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
  --no_spaq \
  2>&1 | tee logs/sensitivity_c3_alpha0.7_$(date +%Y%m%d_%H%M%S).log
```

---

## 🔷 Stage 3: Model Size Comparison (2 experiments)

**Goal**: Compare Tiny, Small, Base with consistent config (α=0.3)

### B1: Swin-Tiny

```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size tiny \
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
  --no_spaq \
  2>&1 | tee logs/model_size_b1_tiny_alpha0.3_$(date +%Y%m%d_%H%M%S).log
```

---

### B2: Swin-Small

```bash
cd /root/Perceptual-IQA-CS3324 && python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
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
  --no_spaq \
  2>&1 | tee logs/model_size_b2_small_alpha0.3_$(date +%Y%m%d_%H%M%S).log
```

---

## 🎛️ Stage 4: Regularization Sensitivity (6 experiments)

**Goal**: Vary ONE regularization parameter at a time

### D1: Lower Weight Decay (2e-4 → 1e-4)

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
  --weight_decay 1e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/reg_d1_wd1e-4_$(date +%Y%m%d_%H%M%S).log
```

---

### D2: Higher Weight Decay (2e-4 → 3e-4)

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
  --weight_decay 3e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/reg_d2_wd3e-4_$(date +%Y%m%d_%H%M%S).log
```

---

### D3: Lower Drop Path (0.3 → 0.2)

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
  --drop_path_rate 0.2 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/reg_d3_dp0.2_$(date +%Y%m%d_%H%M%S).log
```

---

### D4: Higher Drop Path (0.3 → 0.4)

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
  --drop_path_rate 0.4 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/reg_d4_dp0.4_$(date +%Y%m%d_%H%M%S).log
```

---

### D5: Lower Dropout (0.4 → 0.3)

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
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/reg_d5_do0.3_$(date +%Y%m%d_%H%M%S).log
```

---

### D6: Higher Dropout (0.4 → 0.5)

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
  --dropout_rate 0.5 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/reg_d6_do0.5_$(date +%Y%m%d_%H%M%S).log
```

---

## 📈 Stage 5: Learning Rate Sensitivity (2 experiments)

**Goal**: Test lower and higher learning rates

### E1: Lower LR (5e-6 → 2.5e-6)

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
  --lr 2.5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/lr_e1_lr2.5e-6_$(date +%Y%m%d_%H%M%S).log
```

---

### E2: Higher LR (5e-6 → 1e-5)

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
  --lr 1e-5 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  2>&1 | tee logs/lr_e2_lr1e-5_$(date +%Y%m%d_%H%M%S).log
```

---

## 🚀 Execution Strategy

### Option 1: Run in Parallel Batches (Recommended for overnight)

**Batch 1** (3 parallel, ~1.5h):
- A1 (Remove Attention)
- A2 (Remove Ranking)
- A3 (Remove Multi-scale)

**Batch 2** (3 parallel, ~1.5h):
- C1 (Alpha=0.1)
- C2 (Alpha=0.5)
- C3 (Alpha=0.7)

**Batch 3** (2 parallel, ~1.5h):
- B1 (Tiny)
- B2 (Small)

**Batch 4** (3 parallel, ~1.5h):
- D1 (WD=1e-4)
- D2 (WD=3e-4)
- D3 (DP=0.2)

**Batch 5** (3 parallel, ~1.5h):
- D4 (DP=0.4)
- D5 (DO=0.3)
- D6 (DO=0.5)

**Batch 6** (2 parallel, ~1.5h):
- E1 (LR=2.5e-6)
- E2 (LR=1e-5)

**Total time**: ~9 hours (overnight run)

### Option 2: Run All Sequentially (~24h)

Not recommended - too long

---

## 📝 Notes

1. All experiments use **5 epochs** for quick validation
2. All experiments use **patience=5** for early stopping
3. All experiments use **alpha=0.3** as baseline (except C experiments)
4. All experiments use **strong regularization** (wd=2e-4, dp=0.3, do=0.4)
5. Results will be collected and analyzed after completion


