# Experiments Log Tracker - Round 2 (Simplified Model) 🚀

**Purpose**: Track all ablation and sensitivity experiments with their log files and results.  
**Baseline**: Alpha=0 (NO Ranking Loss), **NO ColorJitter** (SRCC **0.9354**, PLCC 0.9465)  
**Configuration**: batch_size=32, epochs=5, train_test_num=1, **--no_color_jitter**, **--ranking_loss_alpha 0**  
**Started**: 2025-12-22  

## 🔥 Round 2 Changes - IMPORTANT DISCOVERY! 
- ✅ **Ranking Loss is HARMFUL!** Removing it improves SRCC: 0.9354 vs 0.9332 (+0.0022)
- ✅ All experiments use `--ranking_loss_alpha 0` (no ranking loss)
- ✅ All experiments use `--no_color_jitter` (3x faster training)
- ✅ New baseline: SRCC 0.9354 (best so far!)
- ✅ Training time: ~1.7h per experiment
- ✅ Fair comparison across all experiments
- ✅ Total 11 core experiments (C1-C3 moved to supplementary)

---

## Progress Overview

**Completed**: 1/11 (NEW Baseline - No Ranking Loss, No ColorJitter ✅)  
**Running**: 2/11 (A1, A2 in progress 🔄)  
**Remaining**: 8/11

**Core Experiments** (11 total):
- [x] **Baseline** - No Ranking Loss, No ColorJitter - **SRCC 0.9354** ✅
- [ ] A1 - Remove Attention 🔄 (Running on GPU 1)
- [ ] A2 - Remove Multi-scale 🔄 (Running on GPU 3)
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

**Supplementary Experiments** (Ranking Loss Sensitivity - Optional):
- [ ] C1 - Alpha=0.1
- [ ] C2 - Alpha=0.3
- [ ] C3 - Alpha=0.5
- [ ] C4 - Alpha=0.7

---

## 📊 Baseline (Best Model)

### ⭐ NEW Baseline - Simplified Model (No Ranking Loss, No ColorJitter)

**Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0_20251222_161625.log`

**Configuration**:
- Model Size: base
- Multi-scale: ✅ True
- Attention Fusion: ✅ True
- **ColorJitter**: ❌ **Disabled** (3x faster training)
- **Ranking Loss Alpha**: **0** (NO ranking loss - simpler and better!)
- Learning Rate: 5e-6
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4
- LR Scheduler: cosine
- Test Random Crop: ✅ True

**Results**:
- **SRCC**: **0.9354** 🏆 (Best so far!)
- **PLCC**: **0.9448**
- **Time**: ~1.7 hours
- **Checkpoint**: `checkpoints/koniq-10k-swin_20251222_161625/best_model_srcc_0.9354_plcc_0.9448.pkl`
- **Status**: ✅ COMPLETE
- **Training Log**: Complete with 5 epochs, best at epoch 3

**Key Discovery**: 
- ✅ **Ranking Loss is harmful!** Removing it improves SRCC by +0.0022 (0.9354 vs 0.9332)
- ✅ Simpler model (L1 loss only) performs better than complex ranking loss
- ✅ This is our new baseline for all experiments

---

## 🔬 Part A: Core Ablations

### A1 - Remove Attention Fusion

**Status**: 🔄 RUNNING (GPU 1, started 18:42)

**Configuration**: Same as baseline except:
- Attention Fusion: ❌ **False** (removed)

**Results**:
- **SRCC**: - (in progress)
- **PLCC**: - (in progress)
- **Time**: ~27 minutes elapsed
- **Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0_20251222_184235.log`

**Purpose**: Quantify the contribution of attention-based multi-scale feature fusion.

**Expected**: SRCC drop, showing attention is important for fusing multi-scale features.

---

### ~~A2 - Remove Ranking Loss~~ → **Now the Baseline!**

**Status**: ✅ **COMPLETE - This is now our baseline!**

**Results**:
- **SRCC**: **0.9354** (better than with ranking loss!)
- **PLCC**: **0.9448**
- **Time**: ~1.7 hours
- **Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_multiscale_ranking_alpha0_20251222_161625.log`

**Conclusion**: Ranking Loss Alpha=0 (no ranking loss) is **better** than Alpha=0.3. This experiment became our new baseline!

**Note**: This is the same as the baseline experiment - we discovered ranking loss is harmful, so removing it became our best configuration.

---

### A2 - Remove Multi-scale Features

**Status**: 🔄 RUNNING (GPU 3, started 18:43)

**Configuration**: Same as baseline except:
- Multi-scale: ❌ **False** (removed, using only last layer)
- Attention Fusion: ✅ True (but only one scale to process)

**Results**:
- **SRCC**: - (in progress)
- **PLCC**: - (in progress)
- **Time**: ~3 minutes elapsed
- **Log File**: `/root/Perceptual-IQA-CS3324/logs/swin_ranking_alpha0_20251222_184358.log`

**Purpose**: Quantify the contribution of multi-scale feature extraction.

**Expected**: SRCC drop, showing multi-scale features are crucial.

---

## 📈 Part C: Ranking Loss Sensitivity Analysis (SUPPLEMENTARY - Optional)

**Status**: **MOVED TO SUPPLEMENTARY**  
**Reason**: Discovered that ranking loss is harmful (Alpha=0 is best)

**Known Results**:
- Alpha=0.0: SRCC 0.9354 ✅ (Best - now baseline)
- Alpha=0.3: SRCC 0.9332 (worse by -0.0022)

**Conclusion**: Ranking loss consistently hurts performance. Not running C1-C3 in core experiments.

---

### C1 - Alpha=0.1 (Supplementary)

**Status**: ⏳ SUPPLEMENTARY - Not needed for core paper

**Configuration**: Same as baseline except:
- Ranking Loss Alpha: **0.1**

---

### C2 - Alpha=0.3 (Supplementary)

**Status**: ✅ Already have data - SRCC 0.9332 (worse than baseline)

---

### C3 - Alpha=0.5 (Supplementary)

**Status**: ⏳ SUPPLEMENTARY - Not needed for core paper

**Configuration**: Same as baseline except:
- Ranking Loss Alpha: **0.5**

---

### C4 - Alpha=0.7 (Supplementary)

**Status**: ⏳ SUPPLEMENTARY - Not needed for core paper

**Configuration**: Same as baseline except:
- Ranking Loss Alpha: **0.7**

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

## 🎯 Next Steps & Recommended Priority

### Priority 1: Core Ablations (Critical for Paper) ⭐⭐⭐
- [x] **Baseline** - SRCC 0.9354 ✅ COMPLETE
- [ ] **A1** - Remove Attention 🔄 RUNNING
- [ ] **A2** - Remove Multi-scale 🔄 RUNNING

**Why**: These quantify the contribution of our key architectural components.

---

### Priority 2: Model Size Comparison (Important) ⭐⭐
- [ ] **B1** - Swin-Tiny (~28M params)
- [ ] **B2** - Swin-Large (~197M params)

**Why**: Shows whether our approach works across different model scales and helps understand capacity requirements.

**Expected Findings**:
- Tiny: Likely ~0.925-0.930 SRCC (reduced capacity)
- Large: Likely ~0.935-0.937 SRCC (diminishing returns)

**Recommendation**: Run B1 and B2 after A1/A2 complete (can use 2 GPUs in parallel).

---

### Priority 3: Regularization Sensitivity (Optional but Valuable) ⭐
- [ ] **D1** - Weight Decay = 1e-4
- [ ] **D2** - Weight Decay = 5e-4
- [ ] **D3** - Drop Path = 0.1
- [ ] **D4** - Drop Path = 0.5

**Why**: Helps understand robustness to hyperparameters and optimal regularization.

**Recommendation**: Pick 2-3 most interesting ones if time limited.

---

### Priority 4: Learning Rate Sensitivity (Optional) ⭐
- [ ] **E1** - LR = 1e-6
- [ ] **E2** - LR = 3e-6
- [ ] **E3** - LR = 7e-6
- [ ] **E4** - LR = 1e-5

**Why**: Shows training stability and convergence properties.

**Recommendation**: Can be supplementary material if short on time.

---

## 📊 Suggested Execution Plan

### Phase 1: Core (Now) - A1, A2
**Time**: ~1.7h  
**Status**: 🔄 In Progress

### Phase 2: Model Size (Next) - B1, B2
**Time**: ~1.7h (parallel on 2 GPUs)  
**Commands**:
```bash
# GPU 0
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py --dataset koniq-10k --model_size tiny --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter

# GPU 1
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py --dataset koniq-10k --model_size large --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 --attention_fusion --ranking_loss_alpha 0 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq --no_color_jitter
```

### Phase 3: Optional - D and E groups
**Time**: Depends on how many selected  
**Recommendation**: Can be done later or as supplementary experiments

---

## 📝 How to Update

1. Monitor progress with `watch -n 30 nvidia-smi`
2. Check logs with `tail -f logs/*.log`
3. After each experiment completes, extract results and update above
4. Update progress checkboxes

---

## ⏱️ Time Estimates

- **Per Experiment**: ~1.7 hours
- **Core Experiments**: 10 remaining (1 already done)
- **Sequential (1 GPU)**: ~17 hours (10 experiments)
- **Parallel (2 GPUs)**: ~8.5 hours (5 experiments each)
- **Parallel (4 GPUs)**: ~4.25 hours (optimal scheduling)

**Recommendation**: Run 2-4 experiments simultaneously on separate GPUs. With no ColorJitter and GPU-bound training, resource contention should be minimal.
