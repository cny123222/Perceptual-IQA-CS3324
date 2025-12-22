# 🚀 Two GPU Experiments Plan

**Hardware**: 2 GPUs  
**Strategy**: Sequential batches, LR comparison first, then ablations  
**Total Time**: ~15 hours (7-8 batches)

---

## Phase 1: Learning Rate Comparison (NOW) ⚡

### GPU 0: LR = 1e-6
```bash
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

### GPU 1: LR = 5e-7
```bash
CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 5e-7 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Time**: ~3.4 hours  
**Decision**: Use whichever achieves higher median SRCC (expected: 1e-6)

---

## Phase 2: Core Ablations (After Phase 1)

Assume **LR = 1e-6** is best.

### Batch 1: Ablation Studies

#### GPU 0: A1 - Remove Attention
```bash
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**Note**: NO `--attention_fusion`

#### GPU 1: A2 - Remove Multi-scale
```bash
CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --no_multi_scale \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**Note**: Add `--no_multi_scale`

**Time**: ~3.4 hours

---

## Phase 3: Model Size Comparison

### Batch 2: Tiny and Small Models

#### GPU 0: B1 - Swin-Tiny
```bash
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size tiny \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

#### GPU 1: B2 - Swin-Small
```bash
CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.25 \
  --dropout_rate 0.35 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Time**: ~3.2 hours

---

## Timeline Summary

| Phase | Batch | Experiments | Time | Total |
|-------|-------|-------------|------|-------|
| 1 | LR Comparison | 1e-6 vs 5e-7 | 3.4h | 3.4h |
| 2 | Ablations | A1, A2 | 3.4h | 6.8h |
| 3 | Model Sizes | B1, B2 | 3.2h | 10h |

**Total Wall Time**: ~10 hours

---

## Experiment Order

1. ✅ **Phase 1** (Now): LR 1e-6 vs 5e-7 → Determine best LR
2. ✅ **Phase 2** (Next): A1 (No Attention) + A2 (No Multi-scale) → Quantify contributions
3. ✅ **Phase 3** (Final): B1 (Tiny) + B2 (Small) → Model size analysis

---

## Expected Results

| Experiment | Expected SRCC | Δ vs Best | Notes |
|------------|--------------|-----------|-------|
| **Best (LR 1e-6)** | **0.937** | - | Baseline |
| LR 5e-7 | 0.935 | -0.002 | Too low |
| A1 (No Attention) | 0.932 | -0.005 | Attention: +0.5% |
| A2 (No Multi-scale) | 0.930 | -0.007 | Multi-scale: +0.7% |
| B1 (Tiny) | 0.921 | -0.016 | Smaller capacity |
| B2 (Small) | 0.933 | -0.004 | Good trade-off |

