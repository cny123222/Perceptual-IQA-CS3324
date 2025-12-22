# Final Experiments Results - Ablation & Sensitivity Analysis

**Date**: 2025-12-22  
**Configuration**: batch_size=32, epochs=5, train_test_num=1, **--no_color_jitter**, **--ranking_loss_alpha 0**  
**Baseline**: SRCC **0.9354**, PLCC **0.9465** (no ranking loss, no ColorJitter)

---

## 📊 Baseline (Reference Model)

### Full Model (Best Configuration)

**Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0_20251222_161625.log`

**Configuration**:
- Model Size: base
- Multi-scale: ✅ Yes
- Attention Fusion: ✅ Yes
- Ranking Loss Alpha: 0 (no ranking loss)
- ColorJitter: ❌ No
- Learning Rate: 5e-6
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4

**Results**:
- **SRCC**: **0.9354** 🏆
- **PLCC**: **0.9465**
- **Training Time**: ~1.7 hours
- **Checkpoint**: `checkpoints/koniq-10k-swin_20251222_161625/best_model_srcc_0.9354_plcc_0.9465.pkl`

**Status**: ✅ COMPLETE

---

## 🔬 Part A: Core Ablations

### A1 - Remove Attention Fusion

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Attention Fusion: ❌ No (removed)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Quantify the contribution of attention-based multi-scale feature fusion.

**Expected**: SRCC drop, showing attention fusion is important.

---

### A3 - Remove Multi-scale Features

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Multi-scale: ❌ No (single-scale, last layer only)
- Attention Fusion: N/A (not applicable for single scale)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Quantify the contribution of multi-scale feature extraction.

**Expected**: SRCC drop, showing multi-scale features are important.

---

## 🔍 Part B: Model Size Comparison

### B1 - Swin-Tiny Model

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Model Size: tiny (vs base)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Test if a smaller model can achieve comparable performance.

**Expected**: SRCC drop due to reduced model capacity.

---

### B2 - Swin-Large Model

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Model Size: large (vs base)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Test if a larger model improves performance.

**Expected**: Similar or slightly better SRCC (diminishing returns).

---

## ⚖️ Part D: Regularization Sensitivity

### D1 - Weight Decay = 1e-4 (Lower)

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Weight Decay: 1e-4 (vs 2e-4)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Test impact of lower regularization.

**Expected**: May overfit slightly or perform similarly.

---

### D2 - Weight Decay = 5e-4 (Higher)

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Weight Decay: 5e-4 (vs 2e-4)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Test impact of higher regularization.

**Expected**: May underfit slightly or perform similarly.

---

### D3 - Drop Path = 0.1 (Lower)

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Drop Path Rate: 0.1 (vs 0.3)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Test impact of lower stochastic depth.

**Expected**: Different training dynamics, may overfit.

---

### D4 - Drop Path = 0.5 (Higher)

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Drop Path Rate: 0.5 (vs 0.3)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Test impact of higher stochastic depth.

**Expected**: Different training dynamics, may underfit.

---

## 📉 Part E: Learning Rate Sensitivity

### E1 - LR = 1e-6 (Very Low)

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Learning Rate: 1e-6 (vs 5e-6)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Test very low learning rate.

**Expected**: May not converge fully in 5 epochs.

---

### E2 - LR = 3e-6 (Moderately Low)

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Learning Rate: 3e-6 (vs 5e-6)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Test moderately low learning rate.

**Expected**: Should perform reasonably well.

---

### E3 - LR = 7e-6 (Moderately High)

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Learning Rate: 7e-6 (vs 5e-6)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Test moderately high learning rate.

**Expected**: Should perform reasonably well.

---

### E4 - LR = 1e-5 (High)

**Status**: ⏳ NOT STARTED

**Configuration**: Same as baseline except:
- Learning Rate: 1e-5 (vs 5e-6)

**Results**:
- SRCC: -
- PLCC: -
- Training Time: -
- Log File: -
- Checkpoint: -

**Purpose**: Test high learning rate.

**Expected**: May be unstable or converge too quickly.

---

## 📊 Summary Table (To be filled)

| Experiment | SRCC | PLCC | Δ SRCC | Δ PLCC | Time | Status |
|------------|------|------|--------|--------|------|--------|
| **Baseline** | **0.9354** | **0.9465** | - | - | 1.7h | ✅ |
| A1 (No Attention) | - | - | - | - | - | ⏳ |
| A3 (No Multi-scale) | - | - | - | - | - | ⏳ |
| B1 (Tiny) | - | - | - | - | - | ⏳ |
| B2 (Large) | - | - | - | - | - | ⏳ |
| D1 (WD=1e-4) | - | - | - | - | - | ⏳ |
| D2 (WD=5e-4) | - | - | - | - | - | ⏳ |
| D3 (DP=0.1) | - | - | - | - | - | ⏳ |
| D4 (DP=0.5) | - | - | - | - | - | ⏳ |
| E1 (LR=1e-6) | - | - | - | - | - | ⏳ |
| E2 (LR=3e-6) | - | - | - | - | - | ⏳ |
| E3 (LR=7e-6) | - | - | - | - | - | ⏳ |
| E4 (LR=1e-5) | - | - | - | - | - | ⏳ |

---

## 📝 How to Update

After each experiment completes:

1. Find the log file in `logs/` directory
2. Extract best SRCC and PLCC from the log
3. Update the experiment section above with:
   - SRCC and PLCC values
   - Training time
   - Log file path
   - Checkpoint path
4. Update the summary table
5. Mark status as ✅ COMPLETE

**Example**:
```bash
# Find best result in log
grep "New best model" logs/your_log_file.log | tail -1
```

---

## 🎯 Progress

**Completed**: 1/11 (Baseline only)  
**Running**: 0/11  
**Remaining**: 10/11

---

## 📈 Key Findings (To be filled after experiments)

### Core Ablations
- **Attention Fusion**: [To be determined]
- **Multi-scale Features**: [To be determined]

### Model Size
- **Tiny vs Base**: [To be determined]
- **Large vs Base**: [To be determined]

### Regularization
- **Weight Decay**: Optimal range [To be determined]
- **Drop Path**: Optimal range [To be determined]

### Learning Rate
- **Optimal LR**: [To be determined]
- **LR Sensitivity**: [To be determined]

---

## 🔬 Supplementary Experiments (Future Work)

### Ranking Loss Sensitivity (Not in core experiments)

We discovered that **ranking loss is harmful**:
- Alpha=0 (no ranking loss): SRCC **0.9354** ✅
- Alpha=0.3: SRCC 0.9332 (worse by -0.0022)

Therefore, ranking loss sensitivity analysis is moved to future supplementary work.

---

## 📌 Notes

1. All experiments use the same random seed (42) for reproducibility
2. All experiments use the same data split
3. Training time may vary slightly depending on GPU load
4. SRCC is the primary metric for IQA evaluation
5. Ranking loss experiments are excluded from core analysis

