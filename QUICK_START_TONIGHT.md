# 🚀 Quick Start Guide - Tonight's Experiments

**Date**: Dec 22, 2025  
**Baseline Updated**: Alpha=0.3 (SRCC 0.9352, PLCC 0.9460)

---

## 📋 What's Changing

**Old Baseline**: Alpha=0.5, SRCC 0.9343  
**New Baseline**: Alpha=0.3, SRCC 0.9352 (+0.09%)

**Why**:  
Alpha=0.3 achieved better Round 1 performance in complete training run

**Impact**:  
Need to redo core ablations with new baseline for fair comparison

---

## 🎯 Tonight's Mission

Run **16 experiments** in **6 parallel batches**  
Total time: **~9 hours** (perfect for overnight)

### Experiments List:
- **3** Core Ablations (A1-A3): Remove Attention, Ranking, Multi-scale
- **3** Ranking Sensitivity (C1-C3): Alpha 0.1, 0.5, 0.7
- **2** Model Size (B1-B2): Tiny, Small
- **6** Regularization (D1-D6): Weight decay, Drop path, Dropout variations
- **2** Learning Rate (E1-E2): 2.5e-6, 1e-5

---

## ⚡ Two Ways to Run

### Option 1: Tmux (Recommended - Easy to Monitor)

```bash
cd /root/Perceptual-IQA-CS3324
./run_experiments_tmux.sh
```

**Advantages**:
- ✅ Each experiment in separate tmux window
- ✅ Easy to monitor individual experiments
- ✅ Can attach/detach anytime
- ✅ Built-in GPU monitoring window

**How to use**:
```bash
# Attach to session
tmux attach -t ablation_exp

# List all windows
tmux list-windows -t ablation_exp

# Switch windows
Ctrl+b, w    # Interactive window selector
Ctrl+b, n    # Next window
Ctrl+b, p    # Previous window

# Detach from session
Ctrl+b, d
```

---

### Option 2: Sequential Batches (Simpler)

```bash
cd /root/Perceptual-IQA-CS3324
./run_all_experiments.sh
```

**Advantages**:
- ✅ Fire and forget
- ✅ Automatic batch management
- ✅ Clear progress reporting

**Disadvantages**:
- ❌ Can't easily monitor individual experiments
- ❌ If terminal disconnects, experiments stop

---

## 📊 Execution Timeline

### Parallel Execution (9 hours total):

| Time | Batch | Experiments | GPU Usage |
|------|-------|-------------|-----------|
| 00:00-01:30 | Batch 1 | A1, A2, A3 (Core Ablations) | 3 GPUs |
| 01:30-03:00 | Batch 2 | C1, C2, C3 (Ranking α) | 3 GPUs |
| 03:00-04:30 | Batch 3 | B1, B2 (Model Size) | 2 GPUs |
| 04:30-06:00 | Batch 4 | D1, D2, D3 (Reg Part 1) | 3 GPUs |
| 06:00-07:30 | Batch 5 | D4, D5, D6 (Reg Part 2) | 3 GPUs |
| 07:30-09:00 | Batch 6 | E1, E2 (Learning Rate) | 2 GPUs |

**If started at 11 PM → Done by 8 AM** ☀️

---

## 📁 Where to Find Results

All logs will be saved to:
```
/root/Perceptual-IQA-CS3324/logs/
```

### Log naming convention:
- Core Ablations: `ablation_a[1-3]_*_alpha0.3_TIMESTAMP.log`
- Ranking Sensitivity: `sensitivity_c[1-3]_alpha*.log`
- Model Size: `model_size_b[1-2]_*_alpha0.3_TIMESTAMP.log`
- Regularization: `reg_d[1-6]_*_TIMESTAMP.log`
- Learning Rate: `lr_e[1-2]_*_TIMESTAMP.log`

### Checkpoints will be saved to:
```
/root/Perceptual-IQA-CS3324/checkpoints/
```

---

## 🔍 Monitoring Progress

### Check GPU usage:
```bash
watch -n 1 nvidia-smi
```

### Count running experiments:
```bash
ps aux | grep train_swin.py | grep -v grep | wc -l
```

### Check latest log:
```bash
cd /root/Perceptual-IQA-CS3324/logs
ls -lt *.log | head -5
tail -f <latest_log_file>
```

### Monitor tmux windows:
```bash
tmux attach -t ablation_exp
# Navigate to "monitor" window to see GPU status
```

---

## ⚠️ Important Notes

1. **All experiments use 5 epochs** (quick validation, ~1.5h each)
2. **Early stopping enabled** (patience=5)
3. **All use alpha=0.3** as baseline (except C experiments)
4. **Consistent config** across all experiments for fair comparison

---

## 📈 Expected Results

Based on previous experiments with alpha=0.5:

### Core Ablations:
- **A1 (Remove Attention)**: ~0.9325 (-0.27%)
- **A2 (Remove Ranking)**: ~0.9347 (-0.05%)
- **A3 (Remove Multi-scale)**: ~0.9318 (-0.34%)

### Ranking Sensitivity:
- **C1 (Alpha=0.1)**: ~0.9340
- **C2 (Alpha=0.5)**: ~0.9343 (known)
- **C3 (Alpha=0.7)**: ~0.9330

### Model Size:
- **B1 (Tiny)**: ~0.91-0.92
- **B2 (Small)**: ~0.92-0.93

### Regularization & LR:
- Will show sensitivity curves for paper

---

## 🎯 Tomorrow Morning Tasks

1. **Check completion status**:
   ```bash
   ls -lh logs/*.log | wc -l  # Should be 16+ new logs
   ```

2. **Extract best SRCC from each**:
   ```bash
   for log in logs/ablation_*.log logs/sensitivity_*.log; do
       echo "$log:"
       grep "New best model" $log | tail -1
   done
   ```

3. **Update documentation**:
   - `VALIDATION_AND_ABLATION_LOG.md`
   - `EXPERIMENTS_TO_RUN.md` (mark as done)
   - Create results summary table

4. **Create visualizations**:
   - Component contribution bar chart
   - Ranking loss sensitivity curve
   - Regularization sensitivity curves

5. **Run cross-dataset testing** on new best model (if any)

---

## 🚨 Troubleshooting

### If experiments fail to start:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check dataset exists
ls -lh /root/Perceptual-IQA-CS3324/koniq-10k/
```

### If GPU out of memory:
- Reduce batch size from 32 to 16 in scripts
- Or run experiments sequentially (1 at a time)

### If tmux session gets messed up:
```bash
# Kill and restart
tmux kill-session -t ablation_exp
./run_experiments_tmux.sh
```

---

## 🎉 Ready to Go!

**Recommended command for tonight**:

```bash
cd /root/Perceptual-IQA-CS3324
./run_experiments_tmux.sh

# Then detach and go to sleep 😴
# Ctrl+b, d
```

**Good luck! 🚀**


