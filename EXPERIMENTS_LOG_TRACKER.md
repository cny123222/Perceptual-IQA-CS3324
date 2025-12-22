# Experiments Log Tracker - Round 2 (No ColorJitter) 🚀

**Purpose**: Track all ablation and sensitivity experiments with their log files and results.  
**Baseline**: Alpha=0.3, **NO ColorJitter** (SRCC 0.9350, PLCC 0.9460)  
**Configuration**: batch_size=32, epochs=5, train_test_num=1, **--no_color_jitter**  
**Started**: 2025-12-22  

## 🔥 Round 2 Changes
- ✅ All experiments use `--no_color_jitter`
- ✅ New baseline: SRCC 0.9350 (vs 0.9352 with ColorJitter, drop only -0.0002)
- ✅ Training time: ~1.7h per experiment (vs ~3.2h with ColorJitter)
- ✅ Fair comparison across all experiments
- ✅ Total 14 experiments to run

---

## Progress Overview

**Completed**: 1/14 (A4 - Baseline without ColorJitter ✅)  
**Running**: 0/14  
**Remaining**: 13/14

**Experiments List**:
- [x] A4 - Baseline (No ColorJitter) - **0.9350** ✅
- [ ] A1 - Remove Attention
- [ ] A2 - Remove Ranking Loss  
- [ ] A3 - Remove Multi-scale
- [ ] C1 - Alpha=0.1
- [ ] C2 - Alpha=0.5
- [ ] C3 - Alpha=0.7
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

---

## 📊 Baseline (Best Model)

### ⭐ NEW Baseline - Full Model WITHOUT ColorJitter (Alpha=0.3)

**Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0.3_20251222_135111.log`

**Configuration**:
- Model Size: base
- Multi-scale: ✅ True
- Attention Fusion: ✅ True
- **ColorJitter**: ❌ **Disabled**
- Ranking Loss Alpha: **0.3**
- Learning Rate: 5e-6
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4
- LR Scheduler: cosine
- Test Random Crop: ✅ True

**Results**:
- **SRCC**: **0.9350**
- **PLCC**: **0.9460**
- **Time**: ~1.7 hours
- **Checkpoint**: `checkpoints/koniq-10k-swin-ranking-alpha0.3_20251222_135111/best_model_srcc_0.9350_plcc_0.9460.pkl`
- **Status**: ✅ COMPLETE

**Notes**: This is the new baseline for all Round 2 experiments. Performance drop from ColorJitter removal is negligible (-0.0002 SRCC).

---

## 🔬 Part A: Core Ablations

### A1 - Remove Attention Fusion

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Attention Fusion: ❌ **False** (removed)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Quantify the contribution of attention-based multi-scale feature fusion.

---

### A2 - Remove Ranking Loss

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Ranking Loss Alpha: **0.0** (ranking loss disabled)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Quantify the contribution of ranking loss in addition to L1 loss.

---

### A3 - Remove Multi-scale Features

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Multi-scale: ❌ **False** (removed, using only last layer)
- Attention Fusion: N/A (no fusion needed for single scale)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Quantify the contribution of multi-scale feature extraction.

---

## 📈 Part C: Ranking Loss Sensitivity Analysis

**Purpose**: Understand how ranking_loss_alpha affects model performance.

---

### C1 - Alpha=0.1

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Ranking Loss Alpha: **0.1** (vs 0.3 baseline)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test lower ranking loss weight.

---

### C2 - Alpha=0.5

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Ranking Loss Alpha: **0.5** (vs 0.3 baseline)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test higher ranking loss weight.

---

### C3 - Alpha=0.7

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Ranking Loss Alpha: **0.7** (vs 0.3 baseline)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test even higher ranking loss weight.

---

## 🔍 Part B: Model Size Comparison

**Purpose**: Determine if a larger model provides better performance.

---

### B1 - Tiny Model

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Model Size: **tiny** (vs base)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test smaller, faster model.

---

### B2 - Large Model

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Model Size: **large** (vs base)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test larger, potentially more powerful model.

---

## ⚖️ Part D: Regularization Sensitivity Analysis

**Purpose**: Understand how regularization parameters affect model performance.

---

### D1 - Weight Decay = 1e-4

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Weight Decay: **1e-4** (vs 2e-4 baseline)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test lower weight decay.

---

### D2 - Weight Decay = 5e-4

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Weight Decay: **5e-4** (vs 2e-4 baseline)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test higher weight decay.

---

### D3 - Drop Path = 0.1

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Drop Path Rate: **0.1** (vs 0.3 baseline)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test lower drop path rate.

---

### D4 - Drop Path = 0.5

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Drop Path Rate: **0.5** (vs 0.3 baseline)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test higher drop path rate.

---

## 📉 Part E: Learning Rate Sensitivity Analysis

**Purpose**: Understand how learning rate affects model performance.

---

### E1 - LR = 1e-6

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Learning Rate: **1e-6** (vs 5e-6 baseline)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test very low learning rate.

---

### E2 - LR = 3e-6

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Learning Rate: **3e-6** (vs 5e-6 baseline)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test moderately low learning rate.

---

### E3 - LR = 7e-6

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Learning Rate: **7e-6** (vs 5e-6 baseline)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test moderately high learning rate.

---

### E4 - LR = 1e-5

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Learning Rate: **1e-5** (vs 5e-6 baseline)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **Time**: -
- **Log File**: -

**Purpose**: Test high learning rate.

---

## 📝 How to Update This Log

After each experiment completes:

1. Update the experiment status to ✅ COMPLETE
2. Fill in SRCC, PLCC, and Time
3. Add the log file path
4. Update the checkpoint path if needed
5. Update the progress checkboxes at the top

Example:
```markdown
### A1 - Remove Attention Fusion

**Status**: ✅ COMPLETE

**Results**:
- **SRCC**: 0.9320
- **PLCC**: 0.9440
- **Time**: 1.65h
- **Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0.3_20251222_XXXXXX.log`
```

---

## 🎯 Next Steps

1. Run experiments from `ALL_EXPERIMENTS_COMMANDS.md`
2. Monitor progress with `watch -n 30 nvidia-smi`
3. Use `tail -f logs/*.log` to check training progress
4. Update this file after each experiment completes

---

## ⏱️ Time Estimates

- **Per Experiment**: ~1.7 hours
- **Sequential (1 GPU)**: ~23.8 hours (14 experiments)
- **Parallel (2 GPUs)**: ~11.9 hours (7 experiments each)
- **Parallel (4 GPUs)**: ~6.8 hours (optimal scheduling)

**Recommendation**: Run 1-2 experiments at a time on separate GPUs to avoid resource contention.
