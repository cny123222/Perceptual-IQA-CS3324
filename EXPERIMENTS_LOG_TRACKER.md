# Experiments Log Tracker

**Purpose**: Track all ablation and sensitivity experiments with their log files and results.  
**Baseline**: Alpha=0.3 (SRCC 0.9352, PLCC 0.9460)  
**Configuration**: batch_size=32, epochs=5, train_test_num=1

---

## 📊 Baseline (Best Model)

### Best - Full Model (Alpha=0.3)

**Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0.3_20251221_215123.log`

**Configuration**:
- Model Size: base
- Multi-scale: ✅ True
- Attention Fusion: ✅ True
- Ranking Loss Alpha: **0.3**
- Learning Rate: 5e-6
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4

**Results**:
- **SRCC**: 0.9352 ⭐
- **PLCC**: 0.9460
- **RMSE**: -
- **Checkpoint**: `checkpoints/koniq-10k-swin-ranking-alpha0.3_20251221_215124/best_model_srcc_0.9352_plcc_0.9460.pkl`
- **Status**: ✅ COMPLETE

**Notes**: This is the current best model and baseline for all comparisons.

---

## 🔬 A. Core Ablations

### A1 - Remove Attention Fusion

**Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0.3_20251222_123450.log`

**Configuration**:
- Model Size: base
- Multi-scale: ✅ True
- Attention Fusion: ❌ **False** (removed)
- Ranking Loss Alpha: 0.3
- Learning Rate: 5e-6
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4

**Results**:
- **SRCC**: - (in progress or interrupted)
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏸️ INCOMPLETE

**Purpose**: Quantify the contribution of attention-based multi-scale feature fusion.

---

### A2 - Remove Ranking Loss

**Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0_20251222_104715.log`

**Configuration**:
- Model Size: base
- Multi-scale: ✅ True
- Attention Fusion: ✅ True
- Ranking Loss Alpha: **0.0** (ranking loss disabled)
- Learning Rate: 5e-6
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4

**Results**:
- **SRCC**: - (in progress or interrupted)
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏸️ INCOMPLETE

**Purpose**: Quantify the contribution of ranking loss in addition to L1 loss.

---

### A3 - Remove Multi-scale Feature Fusion

**Log File**: (not started)

**Configuration**:
- Model Size: base
- Multi-scale: ❌ **False** (removed, single-scale)
- Attention Fusion: ✅ True
- Ranking Loss Alpha: 0.3
- Learning Rate: 5e-6
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Quantify the contribution of multi-scale feature fusion.

**Command**:
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
  --no_multiscale \
  --attention_fusion \
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

## 📈 C. Ranking Loss Sensitivity

### C1 - Alpha=0.1 (Lower)

**Log File**: (not started)

**Configuration**: Same as baseline except:
- Ranking Loss Alpha: **0.1** (lower)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Test lower ranking loss weight.

---

### C2 - Alpha=0.5 (Higher, Original Best)

**Log File**: (not started)

**Configuration**: Same as baseline except:
- Ranking Loss Alpha: **0.5** (original best from Round 1)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Compare with original best configuration.

---

### C3 - Alpha=0.7 (Much Higher)

**Log File**: (not started)

**Configuration**: Same as baseline except:
- Ranking Loss Alpha: **0.7** (much higher)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Test much higher ranking loss weight.

---

## 🏗️ B. Model Size Comparison

### B1 - Swin-Tiny (~28M params)

**Log File**: (not started)

**Configuration**: Same as baseline except:
- Model Size: **tiny**

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Test smaller model capacity.

---

### B2 - Swin-Small (~50M params)

**Log File**: (not started)

**Configuration**: Same as baseline except:
- Model Size: **small**

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Test medium model capacity.

---

## ⚙️ D. Weight Decay Sensitivity

### D1 - WD=5e-5 (Very Weak, 0.25×)

**Log File**: (not started)

**Configuration**: Same as baseline except:
- Weight Decay: **5e-5** (very weak)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Test very weak regularization.

---

### D2 - WD=1e-4 (Weak, 0.5×)

**Log File**: (not started)

**Configuration**: Same as baseline except:
- Weight Decay: **1e-4** (weak)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Test weak regularization.

---

### D4 - WD=4e-4 (Strong, 2×)

**Log File**: (not started)

**Configuration**: Same as baseline except:
- Weight Decay: **4e-4** (strong)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Test strong regularization.

---

## 📊 E. Learning Rate Sensitivity

### E1 - LR=2.5e-6 (Conservative, 0.5×)

**Log File**: (not started)

**Configuration**: Same as baseline except:
- Learning Rate: **2.5e-6** (conservative)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Test conservative learning rate.

---

### E3 - LR=7.5e-6 (Faster, 1.5×)

**Log File**: (not started)

**Configuration**: Same as baseline except:
- Learning Rate: **7.5e-6** (faster)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Test faster learning rate.

---

### E4 - LR=1e-5 (Aggressive, 2×)

**Log File**: (not started)

**Configuration**: Same as baseline except:
- Learning Rate: **1e-5** (aggressive)

**Results**:
- **SRCC**: -
- **PLCC**: -
- **RMSE**: -
- **Status**: ⏳ NOT STARTED

**Purpose**: Test aggressive learning rate.

---

## 📋 Summary Table

| Exp | Description | Log File | SRCC | PLCC | RMSE | Status |
|-----|-------------|----------|------|------|------|--------|
| **Best** | **Baseline (Alpha=0.3)** | `...alpha0.3_20251221_215123.log` | **0.9352** | **0.9460** | - | ✅ |
| A1 | Remove Attention | `...alpha0.3_20251222_123450.log` | - | - | - | ⏸️ |
| A2 | Remove Ranking | `...alpha0_20251222_104715.log` | - | - | - | ⏸️ |
| A3 | Remove Multi-scale | - | - | - | - | ⏳ |
| C1 | Alpha=0.1 | - | - | - | - | ⏳ |
| C2 | Alpha=0.5 | - | - | - | - | ⏳ |
| C3 | Alpha=0.7 | - | - | - | - | ⏳ |
| B1 | Swin-Tiny | - | - | - | - | ⏳ |
| B2 | Swin-Small | - | - | - | - | ⏳ |
| D1 | WD=5e-5 | - | - | - | - | ⏳ |
| D2 | WD=1e-4 | - | - | - | - | ⏳ |
| D4 | WD=4e-4 | - | - | - | - | ⏳ |
| E1 | LR=2.5e-6 | - | - | - | - | ⏳ |
| E3 | LR=7.5e-6 | - | - | - | - | ⏳ |
| E4 | LR=1e-5 | - | - | - | - | ⏳ |

**Legend**:
- ✅ Complete
- ⏸️ Incomplete (interrupted or in progress)
- ⏳ Not started
- ⭐ Best result

---

## 📝 Update Instructions

### When an experiment completes:

1. Find the log file path
2. Extract final results:
   ```bash
   grep "best model" <log_file> | tail -1
   ```
3. Update this document with:
   - Log file path
   - SRCC/PLCC/RMSE results
   - Checkpoint path
   - Status: ✅ COMPLETE
4. Update the summary table

### Example Update:
```markdown
**Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_experiment_date.log`

**Results**:
- **SRCC**: 0.9350
- **PLCC**: 0.9455
- **RMSE**: 0.1850
- **Checkpoint**: `checkpoints/.../best_model_...pkl`
- **Status**: ✅ COMPLETE
```

---

## 🎯 Next Steps

### Priority 1 (Core Ablations - Most Important):
1. **A1** - Complete or re-run (Remove Attention)
2. **A2** - Complete or re-run (Remove Ranking)
3. **A3** - Run (Remove Multi-scale)

### Priority 2 (Ranking Sensitivity):
4. **C1** - Run (Alpha=0.1)
5. **C2** - Run (Alpha=0.5)
6. **C3** - Run (Alpha=0.7)

### Priority 3 (Model Size):
7. **B1** - Run (Swin-Tiny)
8. **B2** - Run (Swin-Small)

### Priority 4 (Weight Decay):
9. **D1** - Run (WD=5e-5)
10. **D2** - Run (WD=1e-4)
11. **D4** - Run (WD=4e-4)

### Priority 5 (Learning Rate):
12. **E1** - Run (LR=2.5e-6)
13. **E3** - Run (LR=7.5e-6)
14. **E4** - Run (LR=1e-5)

---

## 📊 Progress

- **Completed**: 1/15 (6.7%)
- **In Progress/Incomplete**: 2/15 (13.3%)
- **Not Started**: 12/15 (80.0%)

**Estimated Time Remaining**: ~12 experiments × 5-10 minutes = 60-120 minutes

---

**Last Updated**: 2025-12-22 12:45 (after stopping all running experiments)

