# All 10 Core Experiments - Simplified Model (No Ranking Loss)

**Date**: 2025-12-22 (MAJOR UPDATE)  
**Configuration**: batch_size=32, epochs=5, train_test_num=1, **--no_color_jitter**, **--ranking_loss_alpha 0**  
**Baseline**: Alpha=0 (SRCC **0.9354**, no ranking loss, no ColorJitter)  
**Time per Experiment**: ~1.7 hours (~20min/epoch)

## 🔥 IMPORTANT DISCOVERY!

**Ranking Loss is HARMFUL!** 
- ✅ Alpha=0 (no ranking loss): SRCC **0.9354** (Best!)
- ❌ Alpha=0.3 (with ranking loss): SRCC 0.9332 (worse by -0.0022)

**Decision**: 
- All experiments now use `--ranking_loss_alpha 0`
- Ranking loss sensitivity (C1-C3) moved to supplementary experiments
- Only 10 core experiments remaining

**Other Improvements**:
- ✅ **No ColorJitter**: 3x faster training (1.7h vs 3.2h)
- ✅ **Simpler model**: L1 loss only, no complex ranking loss
- ✅ **Better performance**: SRCC 0.9354 is our best result so far!

**Running Strategy**:
- **2-4 GPUs simultaneously**: Recommended! (~4-8 hours total)
- **1-2 GPUs**: Conservative approach (~8-17 hours total)

---

## A. Core Ablations (核心消融)

### A1: 移除Attention Fusion

**目的**: 验证Attention Fusion的贡献  
**预计时间**: ~1.7小时  
**GPU**: 0

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: Should show drop in SRCC (quantify attention contribution)

---

### A3: 移除Multi-scale Feature Fusion

**目的**: 验证Multi-scale的贡献  
**预计时间**: ~1.7小时  
**GPU**: 1

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --no_multiscale \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: Should show drop in SRCC (quantify multi-scale contribution)

---

## B. Model Size Comparison (模型大小对比)

### B1: Swin-Tiny Model

**目的**: 测试更小、更快的模型  
**预计时间**: ~1.7小时  
**GPU**: 2

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=2 python train_swin.py \
  --dataset koniq-10k \
  --model_size tiny \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: Likely lower SRCC due to less capacity

---

### B2: Swin-Large Model

**目的**: 测试更大、更强的模型  
**预计时间**: ~1.7小时  
**GPU**: 3

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=3 python train_swin.py \
  --dataset koniq-10k \
  --model_size large \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: May improve slightly or plateau (diminishing returns)

---

## D. Regularization Sensitivity (正则化敏感度)

### D1: Weight Decay = 1e-4 (Lower)

**目的**: 测试更低的Weight Decay  
**预计时间**: ~1.7小时  
**GPU**: 0

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 1e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: May overfit slightly (less regularization)

---

### D2: Weight Decay = 5e-4 (Higher)

**目的**: 测试更高的Weight Decay  
**预计时间**: ~1.7小时  
**GPU**: 1

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 5e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: May underfit slightly (too much regularization)

---

### D3: Drop Path Rate = 0.1 (Lower)

**目的**: 测试更低的Drop Path  
**预计时间**: ~1.7小时  
**GPU**: 2

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=2 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.1 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: May show different generalization behavior

---

### D4: Drop Path Rate = 0.5 (Higher)

**目的**: 测试更高的Drop Path  
**预计时间**: ~1.7小时  
**GPU**: 3

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=3 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.5 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: May show different training dynamics

---

## E. Learning Rate Sensitivity (学习率敏感度)

### E1: LR = 1e-6 (Very Low)

**目的**: 测试非常低的学习率  
**预计时间**: ~1.7小时  
**GPU**: 0

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: May train too slowly, not converge fully

---

### E2: LR = 3e-6 (Moderately Low)

**目的**: 测试中等偏低的学习率  
**预计时间**: ~1.7小时  
**GPU**: 1

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --lr 3e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: Should perform reasonably well

---

### E3: LR = 7e-6 (Moderately High)

**目的**: 测试中等偏高的学习率  
**预计时间**: ~1.7小时  
**GPU**: 2

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --lr 7e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: Should perform reasonably well

---

### E4: LR = 1e-5 (High)

**目的**: 测试较高的学习率  
**预计时间**: ~1.7小时  
**GPU**: 3

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --lr 1e-5 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

**Expected Result**: May be unstable or train too quickly

---

## ⏱️ Time Estimates

**Total Experiments**: 10  
**Per Experiment**: ~1.7 hours

**Sequential (1 GPU)**: ~17 hours  
**Parallel (2 GPUs)**: ~8.5 hours  
**Parallel (4 GPUs)**: ~4.25 hours ⚡ (RECOMMENDED)

---

## 🚀 Recommended Execution Plan

### Batch 1 (A + B groups) - Run all 4 simultaneously:
```bash
# Terminal 1
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # A1

# Terminal 2
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --no_multiscale --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # A3

# Terminal 3
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=2 python train_swin.py --dataset koniq-10k --model_size tiny --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # B1

# Terminal 4
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=3 python train_swin.py --dataset koniq-10k --model_size large --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # B2
```

**After Batch 1 completes (~1.7h)**, run Batch 2:

### Batch 2 (D group) - Run all 4 simultaneously:
```bash
# Terminal 1
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 1e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # D1

# Terminal 2
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 5e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # D2

# Terminal 3
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=2 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.1 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # D3

# Terminal 4
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=3 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.5 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # D4
```

**After Batch 2 completes (~1.7h)**, run Batch 3:

### Batch 3 (E group, first 2) - Run 2 simultaneously:
```bash
# Terminal 1
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 1e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # E1

# Terminal 2
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 3e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # E2
```

### Batch 4 (E group, last 2) - Run 2 simultaneously:
```bash
# Terminal 1
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 7e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # E3

# Terminal 2
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 1e-5 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # E4
```

**Total Time: ~6.8 hours** (4 batches: 1.7h + 1.7h + 1.7h + 1.7h)

---

## 📊 Monitoring Commands

```bash
# Watch GPU usage
watch -n 5 nvidia-smi

# Check CPU/Memory
htop

# Monitor latest log
tail -f logs/*.log

# Count running experiments
ps aux | grep train_swin.py | grep -v grep | wc -l
```

---

## ✅ Checklist

- [ ] A1 - Remove Attention
- [ ] A3 - Remove Multi-scale
- [ ] B1 - Model Tiny
- [ ] B2 - Model Large
- [ ] D1 - Weight Decay 1e-4
- [ ] D2 - Weight Decay 5e-4
- [ ] D3 - Drop Path 0.1
- [ ] D4 - Drop Path 0.5
- [ ] E1 - LR 1e-6
- [ ] E2 - LR 3e-6
- [ ] E3 - LR 7e-6
- [ ] E4 - LR 1e-5

**Total**: 10 experiments

---

## 📝 Notes

1. **All experiments use `--ranking_loss_alpha 0`** (no ranking loss)
2. **All experiments use `--no_color_jitter`** (3x faster)
3. Baseline SRCC is 0.9354 (best result so far!)
4. With 4 GPUs and no resource contention, can complete all in ~7 hours
5. Remember to update `EXPERIMENTS_LOG_TRACKER.md` after each experiment
