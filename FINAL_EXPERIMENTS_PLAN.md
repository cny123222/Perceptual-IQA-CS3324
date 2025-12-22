# 🎯 Final Experiments Plan - 10 Rounds with Optimal Settings

**Date**: 2024-12-22  
**Purpose**: Complete ablation studies with optimal hyperparameters and statistical robustness  
**Key Changes**: 
- ✅ **10 rounds** (train_test_num=10) for statistical significance
- ✅ **Epochs=10, Patience=3** for better convergence
- ✅ **Best LR = 1e-6** (discovered from E1 experiment)
- ✅ All experiments use **NO ColorJitter**, **NO Ranking Loss**

---

## 📊 Optimal Baseline Configuration

```bash
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

**Expected Best Performance**: SRCC ~0.937 ± 0.002 (based on E1 single-round result)

---

## 🔬 Experiment Groups

### Group 1: Core Ablation Studies (Critical for Paper) ⭐⭐⭐

These experiments quantify the contribution of each architectural component.

#### Baseline - Full Model (Swin-Base + Multi-scale + Attention)
```bash
python train_swin.py \
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
**Purpose**: Establish the best model performance with statistical significance  
**Expected**: SRCC ~0.937 ± 0.002  
**Time**: ~3.4 hours (10 rounds × 10 epochs × ~2 min/epoch)

---

#### A1 - Remove Attention Fusion
```bash
python train_swin.py \
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
**Note**: Remove `--attention_fusion` flag  
**Purpose**: Quantify attention fusion contribution  
**Expected**: SRCC ~0.932 ± 0.002 (Δ -0.005)  
**Time**: ~3.4 hours

---

#### A2 - Remove Multi-scale Fusion (Single-scale only)
```bash
python train_swin.py \
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
**Note**: Add `--no_multi_scale` flag (uses only stage 4 features)  
**Purpose**: Quantify multi-scale fusion contribution  
**Expected**: SRCC ~0.930 ± 0.002 (Δ -0.007)  
**Time**: ~3.4 hours

---

### Group 2: Backbone Comparison (Most Important!) ⭐⭐⭐⭐⭐

This is THE most critical experiment - comparing original ResNet50 vs our Swin Transformer.

#### ResNet50 Baseline (Original HyperIQA)
```bash
python train_test_IQA.py \
  --dataset koniq-10k \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --lr_scheduler cosine \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**Purpose**: Establish original HyperIQA baseline with same training settings  
**Expected**: SRCC ~0.907 ± 0.003  
**Time**: ~2.5 hours (ResNet50 is faster)  
**Note**: This shows the **primary contribution** of Swin Transformer backbone

---

### Group 3: Model Size Comparison ⭐⭐

Compare different Swin Transformer sizes to understand capacity requirements.

#### B1 - Swin-Tiny
```bash
python train_swin.py \
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
**Purpose**: Test smallest model (28M params)  
**Expected**: SRCC ~0.921 ± 0.002  
**Time**: ~3.0 hours  
**Note**: Lower regularization (drop_path=0.2, dropout=0.3) for smaller model

---

#### B2 - Swin-Small
```bash
python train_swin.py \
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
**Purpose**: Test medium model (50M params)  
**Expected**: SRCC ~0.933 ± 0.002  
**Time**: ~3.2 hours  
**Note**: Moderate regularization (drop_path=0.25, dropout=0.35)

---

## 📋 Experiment Summary Table

| ID | Experiment | Model | Multi-scale | Attention | LR | Expected SRCC | Δ vs Best | Priority | Time |
|----|------------|-------|-------------|-----------|----|--------------|-----------| ---------|------|
| **Best** | **Full Model** | **Swin-Base** | ✅ | ✅ | **1e-6** | **0.937** | - | ⭐⭐⭐ | 3.4h |
| A1 | No Attention | Swin-Base | ✅ | ❌ | 1e-6 | 0.932 | -0.005 | ⭐⭐⭐ | 3.4h |
| A2 | No Multi-scale | Swin-Base | ❌ | ❌ | 1e-6 | 0.930 | -0.007 | ⭐⭐⭐ | 3.4h |
| ResNet | Original | ResNet50 | ❌ | ❌ | 1e-6 | 0.907 | -0.030 | ⭐⭐⭐⭐⭐ | 2.5h |
| B1 | Tiny Model | Swin-Tiny | ✅ | ✅ | 1e-6 | 0.921 | -0.016 | ⭐⭐ | 3.0h |
| B2 | Small Model | Swin-Small | ✅ | ✅ | 1e-6 | 0.933 | -0.004 | ⭐⭐ | 3.2h |

**Total Time**: ~19 hours (can run 4 in parallel on 4 GPUs → ~5 hours wall time)

---

## 🚀 Execution Strategy

### Option 1: Sequential (Single GPU)
Run experiments one by one. Total time: ~19 hours.

### Option 2: Parallel (4 GPUs) - RECOMMENDED ⭐
```bash
# GPU 0: Baseline (most important)
CUDA_VISIBLE_DEVICES=0 python train_swin.py [baseline_args] &

# GPU 1: ResNet50 (most important comparison)
CUDA_VISIBLE_DEVICES=1 python train_test_IQA.py [resnet_args] &

# GPU 2: A1 (ablation)
CUDA_VISIBLE_DEVICES=2 python train_swin.py [A1_args] &

# GPU 3: A2 (ablation)
CUDA_VISIBLE_DEVICES=3 python train_swin.py [A2_args] &

# Wait for first batch to complete, then run B1 and B2
```

**Wall Time**: ~5 hours (4 parallel + 2 sequential)

---

## 📊 Expected Results Summary

### Contribution Breakdown (Expected):

| Component | SRCC Improvement | Percentage | Importance |
|-----------|------------------|------------|------------|
| **Swin Transformer (vs ResNet50)** | **+3.0%** | **100%** | ⭐⭐⭐⭐⭐ |
| - Multi-scale Fusion | +0.7% | 23% | ⭐⭐⭐ |
| - Attention Fusion | +0.5% | 17% | ⭐⭐ |
| - Transformer Architecture | +1.8% | 60% | ⭐⭐⭐⭐⭐ |

**Key Insight**: The Swin Transformer backbone itself (hierarchical architecture + shifted window attention) is the primary contributor, with multi-scale and attention fusion providing additional gains.

---

## 📝 Results Recording Template

For each experiment, record:

```markdown
### [Experiment ID] - [Experiment Name]

**Configuration**:
- Model: [base/tiny/small/resnet50]
- Multi-scale: [Yes/No]
- Attention: [Yes/No]
- LR: 1e-6
- Epochs: 10
- Patience: 3
- Rounds: 10

**Results** (10 rounds):
- **Median SRCC**: X.XXXX ± 0.00XX
- **Median PLCC**: X.XXXX ± 0.00XX
- **Best SRCC**: X.XXXX
- **Worst SRCC**: X.XXXX
- **Std Dev**: 0.00XX

**Comparison to Best**:
- Δ SRCC: +/- X.XX%
- Statistical Significance: [Yes/No, p-value]

**Log File**: `logs/[filename].log`
**Checkpoint**: `checkpoints/[dirname]/best_model_*.pkl`
```

---

## 🎯 Paper Writing Implications

### Main Contributions (in order of importance):

1. **Swin Transformer Backbone** (+3.0% SRCC)
   - Hierarchical feature extraction
   - Shifted window multi-head self-attention
   - Pre-trained on ImageNet-22k
   - **This is the PRIMARY contribution**

2. **Multi-scale Feature Fusion** (+0.7% SRCC)
   - Extracts features from all 4 stages [56×56, 28×28, 14×14, 7×7]
   - Captures quality degradations at multiple scales
   - Adaptive pooling to unified size

3. **Attention-based Fusion** (+0.5% SRCC)
   - Channel attention mechanism (Squeeze-Excitation)
   - Adaptive weighting of multi-scale features
   - Content-aware feature selection

4. **Training Optimization** (+0.16% SRCC from LR tuning)
   - Lower learning rate (1e-6 vs 5e-6)
   - Cosine annealing scheduler
   - Simplified loss (no ranking loss)

### Ablation Study Structure for Paper:

**Table 1: Component Ablation**
| Model | Multi-scale | Attention | SRCC | PLCC | Δ SRCC |
|-------|-------------|-----------|------|------|--------|
| ResNet50 (Original) | ❌ | ❌ | 0.907 | - | - |
| Swin-Base (Single-scale) | ❌ | ❌ | 0.930 | - | +2.3% |
| Swin-Base + Multi-scale | ✅ | ❌ | 0.932 | - | +2.5% |
| **Swin-Base + Multi-scale + Attention** | ✅ | ✅ | **0.937** | - | **+3.0%** |

**Table 2: Model Size Comparison**
| Model | Params | SRCC | PLCC | Speed |
|-------|--------|------|------|-------|
| Swin-Tiny | 28M | 0.921 | - | Fast |
| Swin-Small | 50M | 0.933 | - | Medium |
| **Swin-Base** | **88M** | **0.937** | - | **Slow** |

---

## ⚠️ Important Notes

1. **Statistical Significance**: 10 rounds provide robust statistics
   - Can compute mean, median, std dev
   - Can perform statistical tests (t-test, Wilcoxon)
   - More reliable than single-round results

2. **Early Stopping**: Patience=3 (vs 5 before)
   - Faster convergence with 10 epochs
   - Prevents overfitting
   - Saves computation time

3. **Learning Rate**: Use 1e-6 for ALL experiments
   - Discovered as optimal in E1 experiment
   - Provides stable training
   - Better convergence than 5e-6

4. **Reproducibility**: 
   - Fixed random seed (42)
   - CuDNN deterministic mode
   - Same data splits across all experiments

5. **Checkpoints**: 
   - Save best model based on median SRCC
   - Keep only best checkpoint per experiment
   - ~2.7GB per checkpoint

---

## 🔄 Next Steps After Experiments Complete

1. **Analyze Results**:
   - Compute statistics (mean, median, std dev)
   - Create visualization plots (box plots, bar charts)
   - Perform statistical significance tests

2. **Update Documentation**:
   - Record all results in `EXPERIMENTS_LOG_TRACKER.md`
   - Update `MODEL_IMPROVEMENTS_SUMMARY.md` with final numbers
   - Create result tables for paper

3. **Paper Writing**:
   - Method section: describe architecture
   - Experiments section: present ablation results
   - Discussion: analyze contributions
   - Conclusion: summarize findings

4. **Optional Extensions** (if time permits):
   - Cross-dataset evaluation (SPAQ, LIVE-itW)
   - Visualization (attention maps, feature maps)
   - Error analysis (failure cases)

---

**Status**: 📋 Ready to execute  
**Estimated Completion**: ~5 hours (with 4 GPUs in parallel)  
**Expected Outcome**: Robust, statistically significant results for paper submission

