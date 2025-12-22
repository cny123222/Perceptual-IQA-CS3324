# Experiments Log Tracker

**Purpose**: Track all ablation and sensitivity experiments with their log files and results.  
**Baseline**: Alpha=0.3 (SRCC 0.9352, PLCC 0.9460)  
**Configuration**: batch_size=32, epochs=5, train_test_num=1

---

## 📊 Baseline (Best Model)

### Best (OLD) - Full Model with ColorJitter (Alpha=0.3)

**Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0.3_20251221_215123.log`

**Configuration**:
- Model Size: base
- Multi-scale: ✅ True
- Attention Fusion: ✅ True
- **ColorJitter**: ✅ **Enabled**
- Ranking Loss Alpha: **0.3**
- Learning Rate: 5e-6
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4

**Results**:
- **SRCC**: 0.9352
- **PLCC**: 0.9460
- **Time**: ~3.2 hours
- **Checkpoint**: `checkpoints/koniq-10k-swin-ranking-alpha0.3_20251221_215124/best_model_srcc_0.9352_plcc_0.9460.pkl`
- **Status**: ✅ COMPLETE (superseded by new baseline)

**Notes**: Original baseline with ColorJitter. Superseded by new baseline without ColorJitter.

---

### Best (NEW) - Full Model WITHOUT ColorJitter (Alpha=0.3) ⭐

**Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0.3_20251222_135111.log`

**Configuration**:
- Model Size: base
- Multi-scale: ✅ True
- Attention Fusion: ✅ True
- **ColorJitter**: ❌ **Disabled** (A4 ablation result)
- Ranking Loss Alpha: **0.3**
- Learning Rate: 5e-6
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4

**Results**:
- **SRCC**: 0.9350 ⭐ (only -0.0002 vs with ColorJitter)
- **PLCC**: 0.9460 (same as with ColorJitter)
- **Time**: ~1.7 hours (47% faster!)
- **Checkpoint**: TBD
- **Status**: ✅ COMPLETE

**Notes**: 
- **This is the NEW baseline for all subsequent experiments!**
- ColorJitter removed after A4 ablation showed negligible performance impact (-0.0002 SRCC)
- All future experiments should use `--no_color_jitter` for fair comparison
- Significantly faster training (~1.9x speedup)

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
- **Status**: 

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
- **SRCC**: - 0.9345
- **PLCC**: - 0.9453
- **RMSE**: - 
- **Checkpoint**: `checkpoints/koniq-10k-swin_20251222_104715/best_model_srcc_0.9345_plcc_0.9453.pkl`
- **Status**: ✅ COMPLETE

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

### A4 - Remove ColorJitter (Data Augmentation) ⭐

**Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0.3_20251222_135111.log`

**Configuration**: Same as OLD baseline except:
- **ColorJitter**: ❌ **Disabled** (via --no_color_jitter flag)

**Results**:
- **SRCC**: 0.9350 (baseline: 0.9352, drop: **-0.0002 only!**)
- **PLCC**: 0.9460 (same as baseline)
- **RMSE**: -
- **Time**: ~1.7 hours (vs 3.2 hours with ColorJitter)
- **Status**: ✅ COMPLETE

**Purpose**: Quantify the contribution of ColorJitter data augmentation. ColorJitter causes ~2x training slowdown.

**Conclusion**: 
- ✅ **ColorJitter can be safely removed!**
- Performance loss is negligible (0.0002 SRCC ≈ 0.02%)
- Training speed improves by ~1.9x
- **Decision**: Remove ColorJitter from all subsequent experiments

**How to run**:
1. Comment out line 49 in `data_loader.py`:
   ```python
   # torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
   ```
2. Run the same command as baseline
3. Restore the line after experiment

**Command**:
```bash
# Step 1: Comment out ColorJitter in data_loader.py
# Step 2: Run experiment
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
  --ranking_loss_alpha 0.3 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
# Step 3: Restore ColorJitter line
```

**Expected Time**: ~40 minutes (vs 2 hours with ColorJitter)

**Decision Point**: 
- If SRCC drop < 0.002: Remove ColorJitter, save 67% time on all experiments
- If SRCC drop > 0.005: Keep ColorJitter, accept slower training
- If 0.002 < drop < 0.005: Discuss trade-off in paper

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

| Exp | Description | Log File | SRCC | PLCC | Time | Status |
|-----|-------------|----------|------|------|------|--------|
| **Best** | **Baseline (Alpha=0.3)** | `...alpha0.3_20251221_215123.log` | **0.9352** | **0.9460** | 2h | ✅ |
| **A4** | **Remove ColorJitter** | - | - | - | **40min** | **⏳ PRIORITY!** |
| A1 | Remove Attention | `...alpha0.3_20251222_123450.log` | - | - | 2h | ⏸️ |
| A2 | Remove Ranking | `...alpha0_20251222_104715.log` | - | - | 2h | ⏸️ |
| A3 | Remove Multi-scale | - | - | - | 2h | ⏳ |
| C1 | Alpha=0.1 | - | - | - | 2h | ⏳ |
| C2 | Alpha=0.5 | - | - | - | 2h | ⏳ |
| C3 | Alpha=0.7 | - | - | - | 2h | ⏳ |
| B1 | Swin-Tiny | - | - | - | 1.5h | ⏳ |
| B2 | Swin-Small | - | - | - | 2.5h | ⏳ |
| D1 | WD=5e-5 | - | - | - | 2h | ⏳ |
| D2 | WD=1e-4 | - | - | - | 2h | ⏳ |
| D4 | WD=4e-4 | - | - | - | 2h | ⏳ |
| E1 | LR=2.5e-6 | - | - | - | 2h | ⏳ |
| E3 | LR=7.5e-6 | - | - | - | 2h | ⏳ |
| E4 | LR=1e-5 | - | - | - | 2h | ⏳ |

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

### ⚡ PRIORITY 0 (CRITICAL - Do First!):
1. **A4 (Remove ColorJitter)** - 40 minutes
   - **WHY FIRST**: If ColorJitter is not important, all subsequent experiments will be 3x faster!
   - **Impact**: Could save 18.7 hours (67% time) on remaining 14 experiments
   - **Decision point**: Based on A4 results, decide whether to keep or remove ColorJitter
   - See `COLORJITTER_ANALYSIS.md` for details

### Priority 1 (Core Ablations - Most Important):
2. **A1** - Complete or re-run (Remove Attention)
3. **A2** - Complete or re-run (Remove Ranking)
4. **A3** - Run (Remove Multi-scale)

### Priority 2 (Ranking Sensitivity):
5. **C1** - Run (Alpha=0.1)
6. **C2** - Run (Alpha=0.5)
7. **C3** - Run (Alpha=0.7)

### Priority 3 (Model Size):
8. **B1** - Run (Swin-Tiny)
9. **B2** - Run (Swin-Small)

### Priority 4 (Weight Decay):
10. **D1** - Run (WD=5e-5)
11. **D2** - Run (WD=1e-4)
12. **D4** - Run (WD=4e-4)

### Priority 5 (Learning Rate):
13. **E1** - Run (LR=2.5e-6)
14. **E3** - Run (LR=7.5e-6)
15. **E4** - Run (LR=1e-5)

---

## 📊 Progress

- **Completed**: 1/15 (6.7%)
- **In Progress/Incomplete**: 2/15 (13.3%)
- **Not Started**: 13/15 (86.7%) - including A4 (ColorJitter)

**Estimated Time**:
- **With ColorJitter**: ~14 experiments × 2h = 28 hours
- **Without ColorJitter** (if A4 shows it's not important): ~14 experiments × 40min = 9.3 hours
- **Time Savings Potential**: 18.7 hours (67%)

**⚡ CRITICAL**: Run A4 (Remove ColorJitter) first to determine optimal strategy!

---

**Last Updated**: 2025-12-22 12:45 (after stopping all running experiments)

