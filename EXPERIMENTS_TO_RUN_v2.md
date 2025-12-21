# 🔬 Complete Experiment List (Baseline: Alpha=0.3) - v2

**Last Updated**: Dec 22, 2025 00:20  
**Hardware**: 2 × GPU (run 2 experiments in parallel)

**New Best Model (Baseline)**:
- Configuration: Swin-Base + Multi-scale + Attention + Ranking(α=0.3) + Strong Reg
- Performance: **SRCC 0.9352, PLCC 0.9460** (Epoch 3)
- Checkpoint: `checkpoints/koniq-10k-swin-ranking-alpha0.3_20251221_215124/best_model_srcc_0.9352_plcc_0.9460.pkl`

---

## 🎯 Experiments Overview

| Stage | Category | # Experiments | Est. Time | Parallel (2 GPUs) |
|-------|----------|---------------|-----------|-------------------|
| **Stage 1** | Core Ablations (A1-A3) | 3 | 1.5h + 1.5h = 3h | 2 + 1 |
| **Stage 2** | Ranking Sensitivity (C1-C3) | 3 | 1.5h + 1.5h = 3h | 2 + 1 |
| **Stage 3** | Model Size (B1-B2) | 2 | 1.5h | 2 |
| **Stage 4** | Regularization (D1-D6) | 6 | 4.5h | 2 × 3 batches |
| **Stage 5** | Learning Rate (E1-E5) | 5 | 3.75h | 2 + 2 + 1 |
| **Total** | | **19** | **15.75h** | **~16h with 2 GPUs** |

---

## 📋 Stage 1: Core Ablations (3 experiments, 3h)

### A1: Remove Attention Fusion

```bash
python train_swin.py \
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
  --no_spaq
```

---

### A2: Remove Ranking Loss

```bash
python train_swin.py \
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
  --no_spaq
```

---

### A3: Remove Multi-scale Features

```bash
python train_swin.py \
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
  --no_spaq
```

---

## 📊 Stage 2: Ranking Loss Sensitivity (3 experiments, 3h)

### C1: Alpha = 0.1

```bash
python train_swin.py \
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

---

### C2: Alpha = 0.5

```bash
python train_swin.py \
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
  --no_spaq
```

---

### C3: Alpha = 0.7

```bash
python train_swin.py \
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

---

## 🔷 Stage 3: Model Size Comparison (2 experiments, 1.5h)

### B1: Swin-Tiny

```bash
python train_swin.py \
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
  --no_spaq
```

---

### B2: Swin-Small

```bash
python train_swin.py \
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
  --no_spaq
```

---

## 🎛️ Stage 4: Regularization Sensitivity (6 experiments, 4.5h)

**Note**: Each experiment changes ONLY ONE parameter at a time

### D1: Weight Decay = 1e-4 (baseline: 2e-4)

**Changes**: wd 2e-4 → 1e-4 (others unchanged)

```bash
python train_swin.py \
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
  --no_spaq
```

---

### D2: Weight Decay = 3e-4 (baseline: 2e-4)

**Changes**: wd 2e-4 → 3e-4 (others unchanged)

```bash
python train_swin.py \
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
  --no_spaq
```

---

### D3: Drop Path = 0.2 (baseline: 0.3)

**Changes**: dp 0.3 → 0.2 (others unchanged)

```bash
python train_swin.py \
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
  --no_spaq
```

---

### D4: Drop Path = 0.4 (baseline: 0.3)

**Changes**: dp 0.3 → 0.4 (others unchanged)

```bash
python train_swin.py \
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
  --no_spaq
```

---

### D5: Dropout = 0.3 (baseline: 0.4)

**Changes**: do 0.4 → 0.3 (others unchanged)

```bash
python train_swin.py \
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
  --no_spaq
```

---

### D6: Dropout = 0.5 (baseline: 0.4)

**Changes**: do 0.4 → 0.5 (others unchanged)

```bash
python train_swin.py \
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
  --no_spaq
```

---

## 📈 Stage 5: Learning Rate Sensitivity (5 experiments, 3.75h)

**Baseline**: 5e-6  
**Test Range**: 0.5×, 0.75×, 1× (baseline), 1.5×, 2×

### E1: LR = 2.5e-6 (0.5× baseline)

```bash
python train_swin.py \
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
  --no_spaq
```

---

### E2: LR = 3.75e-6 (0.75× baseline)

```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.3 \
  --attention_fusion \
  --lr 3.75e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

### E3: LR = 5e-6 (1× baseline) ✅ ALREADY DONE

**This is the baseline alpha=0.3 experiment - no need to rerun**

---

### E4: LR = 7.5e-6 (1.5× baseline)

```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.3 \
  --attention_fusion \
  --lr 7.5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

### E5: LR = 1e-5 (2× baseline)

```bash
python train_swin.py \
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
  --no_spaq
```

---

## 🚀 Execution Strategy (2 GPUs)

### Batch-by-batch execution:

| Batch | Experiments | GPU0 | GPU1 | Time |
|-------|-------------|------|------|------|
| 1 | A1, A2 | A1 | A2 | 1.5h |
| 2 | A3 | A3 | - | 1.5h |
| 3 | C1, C2 | C1 | C2 | 1.5h |
| 4 | C3 | C3 | - | 1.5h |
| 5 | B1, B2 | B1 | B2 | 1.5h |
| 6 | D1, D2 | D1 | D2 | 1.5h |
| 7 | D3, D4 | D3 | D4 | 1.5h |
| 8 | D5, D6 | D5 | D6 | 1.5h |
| 9 | E1, E2 | E1 | E2 | 1.5h |
| 10 | E4, E5 | E4 | E5 | 1.5h |

**Total**: 10 batches × 1.5h = **15h**

**Note**: E3 (LR=5e-6) is the baseline experiment, already done!

---

## 📝 Summary

- **Total experiments to run**: 18 (19 - 1 already done)
- **Parallel capacity**: 2 GPUs
- **Estimated time**: ~15 hours
- **All experiments use 5 epochs** for quick validation
- **Logs saved automatically** by train_swin.py


