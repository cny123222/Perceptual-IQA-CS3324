# 🚀 Experiments Quick Start Guide

**Status**: Phase 1 RUNNING ✅  
**Hardware**: 2 × NVIDIA GeForce RTX 5090  
**Total Time**: ~10 hours (3 phases)

---

## 📊 Current Status

### ✅ Phase 1: LR Comparison (RUNNING)
- **GPU 0**: LR=1e-6 (10 rounds, 10 epochs)
- **GPU 1**: LR=5e-7 (10 rounds, 10 epochs)
- **Started**: 2024-12-23 00:09
- **ETA**: ~3.4 hours
- **Goal**: Determine optimal learning rate

### ⏳ Phase 2: Ablation Studies (WAITING)
- **A1**: Remove Attention Fusion
- **A2**: Remove Multi-scale Fusion
- **Time**: ~3.4 hours
- **Start**: After Phase 1 completes

### ⏳ Phase 3: Model Size Comparison (WAITING)
- **B1**: Swin-Tiny (28M params)
- **B2**: Swin-Small (50M params)
- **Time**: ~3.2 hours
- **Start**: After Phase 2 completes

---

## 🎯 Expected Results

| Experiment | Expected SRCC | Δ vs Best | Key Finding |
|------------|--------------|-----------|-------------|
| **Phase 1: LR=1e-6** | **0.937** | - | **Best LR** |
| Phase 1: LR=5e-7 | 0.935 | -0.002 | Too low |
| Phase 2: A1 (No Attention) | 0.932 | -0.005 | Attention: **+0.5%** |
| Phase 2: A2 (No Multi-scale) | 0.930 | -0.007 | Multi-scale: **+0.7%** |
| Phase 3: B1 (Tiny) | 0.921 | -0.016 | Capacity matters |
| Phase 3: B2 (Small) | 0.933 | -0.004 | Good trade-off |

---

## 🛠️ Monitoring & Control

### Real-time Monitoring
```bash
# One-time check
./monitor_experiments.sh

# Auto-refresh every 30 seconds
watch -n 30 ./monitor_experiments.sh

# Check specific log
tail -f phase1_lr1e6.out
tail -f phase1_lr5e7.out
```

### GPU Status
```bash
# Quick check
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

### Process Management
```bash
# Check running experiments
ps aux | grep train_swin.py

# Kill if needed (use PID from monitor script)
kill <PID>
```

---

## 🚀 Starting Next Phases

### After Phase 1 Completes:
```bash
# Check results first
./extract_all_results.sh

# Start Phase 2 (Ablations)
./run_phase2_ablations.sh
```

### After Phase 2 Completes:
```bash
# Check results
./extract_all_results.sh

# Start Phase 3 (Model sizes)
./run_phase3_model_sizes.sh
```

---

## 📝 Extract Results

### All Results at Once:
```bash
# Display to terminal
./extract_all_results.sh

# Save to file
./extract_all_results.sh > RESULTS_SUMMARY.txt
```

### Individual Results:
```bash
# Phase 1
grep "median SRCC" phase1_lr1e6.out
grep "median SRCC" phase1_lr5e7.out

# Phase 2 (after started)
grep "median SRCC" phase2_A1_no_attention.out
grep "median SRCC" phase2_A2_no_multiscale.out

# Phase 3 (after started)
grep "median SRCC" phase3_B1_tiny.out
grep "median SRCC" phase3_B2_small.out
```

---

## 📂 Output Files

### Log Files:
- `phase1_lr1e6.out` - GPU 0, LR=1e-6
- `phase1_lr5e7.out` - GPU 1, LR=5e-7
- `phase2_A1_no_attention.out` - A1 ablation
- `phase2_A2_no_multiscale.out` - A2 ablation
- `phase3_B1_tiny.out` - Tiny model
- `phase3_B2_small.out` - Small model

### Checkpoints:
```
checkpoints/
├── koniq-10k-swin_<timestamp>/  # Each experiment gets its own folder
│   └── best_model_srcc_0.9xxx_plcc_0.9xxx.pkl
```

### Training Logs (detailed):
```
logs/
└── swin_multiscale_ranking_alpha0_<timestamp>.log
```

---

## ⏱️ Timeline

| Time | Phase | Action |
|------|-------|--------|
| 00:09 | Phase 1 Start | ✅ Started automatically |
| ~03:40 | Phase 1 Done | Run `./run_phase2_ablations.sh` |
| ~07:10 | Phase 2 Done | Run `./run_phase3_model_sizes.sh` |
| ~10:20 | Phase 3 Done | Run `./extract_all_results.sh` |

---

## 🎯 Key Commands Summary

```bash
# Monitor progress
./monitor_experiments.sh

# Start Phase 2 (after Phase 1)
./run_phase2_ablations.sh

# Start Phase 3 (after Phase 2)
./run_phase3_model_sizes.sh

# Extract all results
./extract_all_results.sh

# Check GPU status
nvidia-smi

# Check specific log
tail -f phase1_lr1e6.out
```

---

## 📊 What to Do After All Phases Complete

1. **Extract Results**:
   ```bash
   ./extract_all_results.sh > FINAL_RESULTS.txt
   ```

2. **Update Tracker**:
   - Open `EXPERIMENTS_LOG_TRACKER.md`
   - Record all median SRCC/PLCC values
   - Add analysis and insights

3. **Prepare Paper Materials**:
   - Use `MODEL_IMPROVEMENTS_SUMMARY.md` as reference
   - Create ablation study table
   - Generate comparison figures

4. **Archive Best Checkpoints**:
   ```bash
   # Find best models
   ls -lh checkpoints/*/best_model*.pkl
   
   # Archive them
   tar -czf best_models.tar.gz checkpoints/*/best_model*.pkl
   ```

---

## 🆘 Troubleshooting

### Experiment Crashed?
```bash
# Check log file for errors
tail -50 phase1_lr1e6.out

# Common issues:
# - Out of memory → reduce batch_size
# - CUDA error → restart experiment
# - Data loading slow → check disk I/O
```

### GPU Not Utilized?
```bash
# Check if process is running
ps aux | grep train_swin.py

# Check GPU assignment
nvidia-smi

# Restart if needed
kill <PID>
./run_phase2_ablations.sh  # or whichever phase
```

### Need to Pause?
```bash
# Find PID
ps aux | grep train_swin.py

# Gracefully stop (Ctrl+C equivalent)
kill -INT <PID>

# Force kill if needed
kill -9 <PID>

# Restart later with same script
```

---

## 📧 Questions?

- Check `TWO_GPU_EXPERIMENTS.md` for detailed plan
- Check `FINAL_EXPERIMENTS_PLAN.md` for methodology
- Check `MODEL_IMPROVEMENTS_SUMMARY.md` for paper writing

---

**Last Updated**: 2024-12-23 00:15  
**Status**: Phase 1 Running, ~180 minutes remaining

