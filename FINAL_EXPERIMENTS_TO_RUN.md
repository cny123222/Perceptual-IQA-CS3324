# Final Experiments to Run - 10 Core Experiments

**Date**: 2025-12-22 (FINAL VERSION)  
**Configuration**: batch_size=32, epochs=5, train_test_num=1, **--no_color_jitter**, **--ranking_loss_alpha 0**  
**Baseline**: SRCC **0.9354** (no ranking loss, no ColorJitter)  
**Time per Experiment**: ~1.7 hours

## 🎯 Key Decisions

1. ✅ **No Ranking Loss** (Alpha=0): Better performance (0.9354 vs 0.9332)
2. ✅ **No ColorJitter**: 3x faster training, negligible performance loss
3. ✅ **Batch Size 32**: Stable and reliable (batch_size=96 doesn't provide speedup)
4. ✅ **10 Core Experiments**: Ranking loss sensitivity moved to future work

---

## Experiment List

### A. Core Ablations (3 experiments)
- A1: Remove Attention Fusion
- A3: Remove Multi-scale Features

### B. Model Size Comparison (2 experiments)
- B1: Swin-Tiny
- B2: Swin-Large

### D. Regularization Sensitivity (4 experiments)
- D1: Weight Decay = 1e-4
- D2: Weight Decay = 5e-4
- D3: Drop Path = 0.1
- D4: Drop Path = 0.5

### E. Learning Rate Sensitivity (4 experiments)
- E1: LR = 1e-6
- E2: LR = 3e-6
- E3: LR = 7e-6
- E4: LR = 1e-5

---

## 📋 All Commands (Copy & Paste)

### A1: Remove Attention Fusion
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

### A3: Remove Multi-scale Features
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --no_multiscale --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

### B1: Swin-Tiny Model
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size tiny --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

### B2: Swin-Large Model
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size large --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

### D1: Weight Decay = 1e-4
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 1e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

### D2: Weight Decay = 5e-4
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 5e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

### D3: Drop Path = 0.1
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.1 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

### D4: Drop Path = 0.5
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.5 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

### E1: LR = 1e-6
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --lr 1e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

### E2: LR = 3e-6
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --lr 3e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

### E3: LR = 7e-6
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --lr 7e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

### E4: LR = 1e-5
```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 \
  --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0 --lr 1e-5 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq --no_color_jitter
```

---

## 🚀 Parallel Execution Strategy

### Option 1: 4 GPUs (Recommended - ~4.25 hours total)

**Batch 1** (~1.7h):
```bash
# Terminal 1 (GPU 0)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # A1

# Terminal 2 (GPU 1)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --no_multiscale --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # A3

# Terminal 3 (GPU 2)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=2 python train_swin.py --dataset koniq-10k --model_size tiny --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # B1

# Terminal 4 (GPU 3)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=3 python train_swin.py --dataset koniq-10k --model_size large --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # B2
```

**Batch 2** (~1.7h):
```bash
# Terminal 1 (GPU 0)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 1e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # D1

# Terminal 2 (GPU 1)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 5e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # D2

# Terminal 3 (GPU 2)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=2 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.1 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # D3

# Terminal 4 (GPU 3)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=3 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.5 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # D4
```

**Batch 3** (~1.7h):
```bash
# Terminal 1 (GPU 0)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 1e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # E1

# Terminal 2 (GPU 1)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 3e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # E2

# Terminal 3 (GPU 2)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=2 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 7e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # E3

# Terminal 4 (GPU 3)
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=3 python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 1e-5 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter  # E4
```

**Total Time: ~5.1 hours** (3 batches × 1.7h)

---

### Option 2: 2 GPUs (~8.5 hours total)

Run 2 experiments at a time, alternating between GPU 0 and GPU 1.

---

### Option 3: 1 GPU (~17 hours total)

Run experiments sequentially on a single GPU.

---

## 📊 Monitoring

```bash
# Watch GPU usage
watch -n 5 nvidia-smi

# Check running experiments
ps aux | grep train_swin.py | grep -v grep

# Monitor latest log
tail -f logs/*.log

# Check progress
ls -lt checkpoints/
```

---

## ✅ Progress Checklist

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

**Total: 10 experiments**

---

## 📝 After Each Experiment

1. Check the log file for best SRCC/PLCC
2. Update `FINAL_EXPERIMENTS_RESULTS.md`
3. Mark the checkbox above as complete

---

## 🎯 Expected Results

All experiments should complete successfully with SRCC around 0.93-0.94 range, with variations showing the impact of each component/hyperparameter.

