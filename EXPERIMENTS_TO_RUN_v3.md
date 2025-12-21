# 🔬 Complete Experiment List (Baseline: Alpha=0.3) - v3 (4-GPU Optimized)

**Date**: 2025-12-22  
**Baseline Model**: koniq-10k-swin_20251221_203438 (Alpha=0.3, SRCC=0.9352, PLCC=0.9471)

---

## 🎯 Experiment Overview

| Stage | Category | Experiments | Description |
|-------|----------|-------------|-------------|
| **Stage 1** | Core Ablations (A) | 3 | Remove one component at a time |
| **Stage 2** | Ranking Loss (C) | 3 | Test different alpha values |
| **Stage 3** | Model Size (B) | 2 | Compare Swin-Tiny/Small with Base |
| **Stage 4** | Regularization (D) | 3 | Weight Decay sensitivity only |
| **Stage 5** | Learning Rate (E) | 3 | LR sensitivity analysis |
| **Total** | | **14** | **~6 hours with 4 GPUs** |

---

## 📊 Baseline Configuration (Alpha=0.3)

```bash
# Baseline: Already completed ✅
# checkpoint: koniq-10k-swin_20251221_203438/best_model_srcc_0.9352_plcc_0.9471.pkl

Model: Swin-Base
Attention: Yes
Multi-scale: Yes (stages [1,2,3])
Ranking Loss Alpha: 0.3
Weight Decay: 2e-4
Drop Path: 0.3
Dropout: 0.4
Learning Rate: 5e-6
Patience: 5
Optimizer: AdamW

Round 1 Results:
  SRCC: 0.9352
  PLCC: 0.9471
  RMSE: 0.1846
```

---

## 🧪 Stage 1: Core Ablation Studies (A)

### A1: Remove Attention Fusion
**Purpose**: Quantify attention mechanism's contribution

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--attention_fusion` removed  
**Status**: ⏳ To Run  
**GPU**: 0  
**Expected**: SRCC < 0.9352 (attention should help)

---

### A2: Remove Ranking Loss (L1 Only)
**Purpose**: Quantify ranking loss contribution

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--ranking_loss_alpha 0`  
**Status**: ⏳ To Run  
**GPU**: 1  
**Expected**: SRCC < 0.9352 (ranking loss should help)

---

### A3: Remove Multi-scale Features (Single-scale)
**Purpose**: Quantify multi-scale fusion's contribution

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--multiscale` and `--multiscale_stages` removed  
**Status**: ⏳ To Run  
**GPU**: 2  
**Expected**: SRCC < 0.9352 (multi-scale should help)

---

## 🎚️ Stage 2: Ranking Loss Sensitivity (C)

### C1: Alpha = 0.1 (Lower)
**Purpose**: Test weaker ranking loss

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.1 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--ranking_loss_alpha 0.1`  
**Status**: ⏳ To Run  
**GPU**: 3  
**Expected**: SRCC ≈ 0.93 (too weak?)

---

### C2: Alpha = 0.5 (Higher)
**Purpose**: Test stronger ranking loss

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.5 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--ranking_loss_alpha 0.5`  
**Status**: ⏳ To Run  
**GPU**: 0  
**Expected**: SRCC ≈ 0.93 (may be too strong)

---

### C3: Alpha = 0.7 (Much Higher)
**Purpose**: Test very strong ranking loss

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.7 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--ranking_loss_alpha 0.7`  
**Status**: ⏳ To Run  
**GPU**: 1  
**Expected**: SRCC < 0.9352 (likely too strong)

---

## 📏 Stage 3: Model Size Comparison (B)

### B1: Swin-Tiny
**Purpose**: Compare smaller model

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size tiny \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--model_size tiny`  
**Status**: ⏳ To Run  
**GPU**: 2  
**Expected**: SRCC < 0.9352 (less capacity)

---

### B2: Swin-Small
**Purpose**: Compare medium model

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size small \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--model_size small`  
**Status**: ⏳ To Run  
**GPU**: 3  
**Expected**: SRCC ≈ 0.93 (between tiny and base)

---

## ⚖️ Stage 4: Regularization Sensitivity (D) - Weight Decay Only

### D1: Weight Decay = 5e-5 (0.25× baseline, very weak)
**Purpose**: Test very weak regularization

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 5e-5 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--weight_decay 5e-5`  
**Status**: ⏳ To Run  
**GPU**: 0  
**Expected**: May overfit, SRCC ≈ 0.93

---

### D2: Weight Decay = 1e-4 (0.5× baseline, weak)
**Purpose**: Test weak regularization

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 1e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--weight_decay 1e-4`  
**Status**: ⏳ To Run  
**GPU**: 1  
**Expected**: SRCC ≈ 0.935

---

### D3: Weight Decay = 2e-4 (Baseline) ✅
**Status**: Already completed  
**Result**: SRCC = 0.9352, PLCC = 0.9471

---

### D4: Weight Decay = 4e-4 (2× baseline, strong)
**Purpose**: Test strong regularization

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 4e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--weight_decay 4e-4`  
**Status**: ⏳ To Run  
**GPU**: 2  
**Expected**: May underfit slightly, SRCC ≈ 0.933

---

## 📈 Stage 5: Learning Rate Sensitivity (E)

### E1: LR = 2.5e-6 (0.5× baseline, conservative)
**Purpose**: Test slower learning

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 2.5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--lr 2.5e-6`  
**Status**: ⏳ To Run  
**GPU**: 3  
**Expected**: May not converge fully, SRCC ≈ 0.93

---

### E2: LR = 5e-6 (Baseline) ✅
**Status**: Already completed  
**Result**: SRCC = 0.9352, PLCC = 0.9471

---

### E3: LR = 7.5e-6 (1.5× baseline)
**Purpose**: Test faster learning

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 7.5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--lr 7.5e-6`  
**Status**: ⏳ To Run  
**GPU**: 0  
**Expected**: SRCC ≈ 0.935 (may be slightly unstable)

---

### E4: LR = 1e-5 (2× baseline, aggressive)
**Purpose**: Test aggressive learning

```bash
python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 1e-5 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5
```

**Changed**: `--lr 1e-5`  
**Status**: ⏳ To Run  
**GPU**: 1  
**Expected**: May be unstable, SRCC ≈ 0.93

---

## 🚀 Execution Plan (4 GPUs)

### Batch 1 (1.5 hours): Core Ablations Start
- **GPU 0**: A1 (Remove Attention)
- **GPU 1**: A2 (Remove Ranking Loss)
- **GPU 2**: A3 (Remove Multi-scale)
- **GPU 3**: C1 (Alpha=0.1)

### Batch 2 (1.5 hours): Ranking & Model Size
- **GPU 0**: C2 (Alpha=0.5)
- **GPU 1**: C3 (Alpha=0.7)
- **GPU 2**: B1 (Swin-Tiny)
- **GPU 3**: B2 (Swin-Small)

### Batch 3 (1.5 hours): Regularization
- **GPU 0**: D1 (WD=5e-5)
- **GPU 1**: D2 (WD=1e-4)
- **GPU 2**: D4 (WD=4e-4)
- **GPU 3**: E1 (LR=2.5e-6)

### Batch 4 (1.5 hours): Learning Rate
- **GPU 0**: E3 (LR=7.5e-6)
- **GPU 1**: E4 (LR=1e-5)

**Total Time: ~6 hours** (4 batches × 1.5h)

---

## 📋 Results Summary Template

After all experiments complete, results will be organized as:

### Core Ablations (A)
| Exp | Config | SRCC | PLCC | RMSE | Δ SRCC | Component Impact |
|-----|--------|------|------|------|--------|------------------|
| Baseline | Full Model | 0.9352 | 0.9471 | 0.1846 | - | - |
| A1 | No Attention | ? | ? | ? | ? | ? |
| A2 | No Ranking | ? | ? | ? | ? | ? |
| A3 | No Multi-scale | ? | ? | ? | ? | ? |

### Ranking Loss Sensitivity (C)
| Exp | Alpha | SRCC | PLCC | RMSE | Δ SRCC |
|-----|-------|------|------|------|--------|
| C1 | 0.1 | ? | ? | ? | ? |
| Baseline | 0.3 | 0.9352 | 0.9471 | 0.1846 | - |
| C2 | 0.5 | ? | ? | ? | ? |
| C3 | 0.7 | ? | ? | ? | ? |

### Model Size (B)
| Exp | Size | Params | FLOPs | SRCC | PLCC | RMSE |
|-----|------|--------|-------|------|------|------|
| B1 | Tiny | ~28M | ~4.5G | ? | ? | ? |
| B2 | Small | ~50M | ~8.7G | ? | ? | ? |
| Baseline | Base | ~88M | ~15.3G | 0.9352 | 0.9471 | 0.1846 |

### Regularization Sensitivity (D)
| Exp | Weight Decay | SRCC | PLCC | RMSE | Δ SRCC |
|-----|--------------|------|------|------|--------|
| D1 | 5e-5 (0.25×) | ? | ? | ? | ? |
| D2 | 1e-4 (0.5×) | ? | ? | ? | ? |
| Baseline | 2e-4 (1×) | 0.9352 | 0.9471 | 0.1846 | - |
| D4 | 4e-4 (2×) | ? | ? | ? | ? |

### Learning Rate Sensitivity (E)
| Exp | Learning Rate | SRCC | PLCC | RMSE | Δ SRCC |
|-----|---------------|------|------|------|--------|
| E1 | 2.5e-6 (0.5×) | ? | ? | ? | ? |
| Baseline | 5e-6 (1×) | 0.9352 | 0.9471 | 0.1846 | - |
| E3 | 7.5e-6 (1.5×) | ? | ? | ? | ? |
| E4 | 1e-5 (2×) | ? | ? | ? | ? |

---

## 📝 Notes

1. **All experiments use the same base configuration** except for the ONE parameter being tested
2. **Training logs** are automatically saved to `logs/` directory
3. **Checkpoints** are saved to `checkpoints/koniq-10k-swin-{config}_{timestamp}/`
4. **Early stopping** with patience=5 ensures efficient training
5. **Results** should be recorded in `VALIDATION_AND_ABLATION_LOG.md`

