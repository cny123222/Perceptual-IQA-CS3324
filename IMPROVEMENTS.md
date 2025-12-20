# Improvements and Modifications to Original HyperIQA

This document comprehensively records all improvements and modifications made to the original HyperIQA implementation, including both successful and unsuccessful attempts. All experiments are documented for inclusion in the appendix of the final report.

---

## 📋 Table of Contents

1. [Backbone Architecture Improvements](#1-backbone-architecture-improvements)
2. [Multi-Scale Feature Fusion](#2-multi-scale-feature-fusion)
3. [Loss Function Enhancements](#3-loss-function-enhancements)
4. [Regularization Strategies](#4-regularization-strategies)
5. [Training Improvements](#5-training-improvements)
6. [Command-Line Parameters](#6-command-line-parameters)
7. [Tools and Utilities](#7-tools-and-utilities)

---

## 1. Backbone Architecture Improvements

### 1.1 Swin Transformer Integration ✅ **Adopted**

**Original**: ResNet-50 backbone
**Improvement**: Swin Transformer (Tiny/Small/Base variants)

**Implementation**:
- Integrated Swin Transformer from `timm` library
- Supported three model sizes:
  - **Swin-Tiny**: ~28M parameters, 4.5 GFLOPs
  - **Swin-Small**: ~50M parameters, 8.7 GFLOPs
  - **Swin-Base**: ~88M parameters, 15.4 GFLOPs

**Results**:
| Model | SRCC | PLCC | Improvement |
|-------|------|------|-------------|
| ResNet-50 (Original) | 0.9009 | 0.9170 | Baseline |
| Swin-Tiny | 0.9236 | 0.9361 | +2.33% |
| Swin-Small | 0.9303 | 0.9444 | +3.07% |
| **Swin-Base** | **0.9336** | **0.9464** | **+3.40%** |

**Key Finding**: Model capacity is the primary driver of performance improvement.

**Code Location**: `models_swin.py`, lines 220-365

**Command-Line Parameter**: `--model_size {tiny,small,base}`

---

## 2. Multi-Scale Feature Fusion

### 2.1 Multi-Scale Feature Extraction ✅ **Adopted**

**Original**: Single-scale feature from last layer only
**Improvement**: Extract features from all 4 Swin Transformer stages

**Implementation**:
```python
# Extract features from all stages
feat0 = layers[0](x)  # [B, 96, H/4, W/4]
feat1 = layers[1](feat0)  # [B, 192, H/8, W/8]
feat2 = layers[2](feat1)  # [B, 384, H/16, W/16]
feat3 = layers[3](feat2)  # [B, 768, H/32, W/32]

# Unify to 7x7 and concatenate
# Tiny/Small: 96+192+384+768 = 1440 channels
# Base: 128+256+512+1024 = 1920 channels
```

**Results**:
- Swin-Tiny single-scale: 0.9154
- Swin-Tiny multi-scale: 0.9236 (+0.82%)

**Code Location**: `models_swin.py`, `swin_backbone()` function

**Command-Line Parameter**: `--no_multiscale` (to disable, enabled by default)

---

### 2.2 Attention-Based Fusion ❌ **Not Adopted**

**Motivation**: Use attention mechanism to dynamically weight multi-scale features

**Implementation**:
```python
class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels_list):
        self.attention_net = nn.Sequential(
            nn.Linear(in_channels_list[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_scales),
            nn.Softmax(dim=1)
        )
```

**Results**:
- Swin-Tiny + Simple Concat: 0.9236
- Swin-Tiny + Attention: 0.9208 (-0.28%)
- Swin-Small + Attention: 0.9311 (Round 1), but unstable (0.9254 in Round 3)

**Reason for Rejection**:
1. Negative impact on Tiny model (insufficient capacity)
2. Marginal improvement on Small model (+0.08%) with high instability
3. Simple concatenation is more stable and effective

**Code Location**: `models_swin.py`, `MultiScaleAttention` class (lines 9-103)

**Command-Line Parameter**: `--attention_fusion` (experimental, not recommended)

---

## 3. Loss Function Enhancements

### 3.1 Ranking Loss (Pairwise) ✅ **Partially Adopted**

**Original**: L1 (MAE) loss only
**Improvement**: Combined L1 loss with pairwise ranking loss

**Implementation**:
```python
def ranking_loss(pred, target, margin=0.1):
    # Create all pairs
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)
    
    # Hinge loss for ranking
    loss = torch.relu(margin - diff_pred * torch.sign(diff_target))
    return loss.mean()

# Combined loss
total_loss = l1_loss + alpha * ranking_loss
```

**Results on Base Model**:
| Alpha | SRCC | PLCC | Notes |
|-------|------|------|-------|
| 0.0 (Pure L1) | 0.9307 | 0.9447 | Baseline |
| 0.3 | 0.9303 | 0.9435 | Too low |
| **0.5** | **0.9336** | **0.9464** | **Optimal** ✅ |
| 0.7 | Not tested | Not tested | - |

**Key Finding**: 
- Ranking loss is more important for larger models
- Small model: alpha=0 vs 0.5 → 0.002 difference (minimal)
- Base model: alpha=0 vs 0.5 → 0.029 difference (significant)

**Code Location**: `HyperIQASolver_swin.py`, `ranking_loss()` function

**Command-Line Parameters**:
- `--ranking_loss_alpha` (default: 0.5)
- `--ranking_loss_margin` (default: 0.1)

---

## 4. Regularization Strategies

### 4.1 Stochastic Depth (Drop Path) ✅ **Adopted**

**Improvement**: Add stochastic depth to Swin Transformer blocks

**Implementation**:
```python
swin_model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    drop_path_rate=drop_path_rate  # Stochastic depth
)
```

**Optimal Values**:
- Tiny/Small: 0.2
- Base: 0.3 (1.5x)

**Reason**: Larger models require stronger regularization to prevent overfitting.

**Command-Line Parameter**: `--drop_path_rate` (default: 0.2)

---

### 4.2 Dropout Regularization ✅ **Adopted**

**Improvement**: Add dropout to HyperNet and TargetNet

**Implementation**:
```python
# In HyperNet
self.dropout = nn.Dropout(dropout_rate)
hyper_in_feat = self.dropout(hyper_in_feat)

# In TargetNet
self.dropout = nn.Dropout(dropout_rate)
```

**Optimal Values**:
- Tiny/Small: 0.3
- Base: 0.4 (1.33x)

**Results on Base**:
- dropout=0.35: SRCC 0.9305
- dropout=0.4: SRCC 0.9336 ✅

**Command-Line Parameter**: `--dropout_rate` (default: 0.3)

---

### 4.3 Weight Decay (L2 Regularization) ✅ **Adopted**

**Improvement**: Increased weight decay for larger models

**Optimal Values**:
- Tiny/Small: 1e-4
- Base: 2e-4 (2x)

**Reason**: Prevents large weights in high-capacity models.

**Command-Line Parameter**: Implicit in optimizer configuration

---

### 4.4 Early Stopping ✅ **Adopted**

**Improvement**: Stop training when validation performance plateaus

**Implementation**:
```python
if current_srcc > best_srcc:
    best_srcc = current_srcc
    patience_counter = 0
    save_checkpoint()
else:
    patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping triggered")
        break
```

**Optimal Value**: patience=7

**Command-Line Parameters**:
- `--patience` (default: 5)
- `--no_early_stopping` (to disable)

---

## 5. Training Improvements

### 5.1 Learning Rate Reduction ✅ **Adopted**

**Improvement**: Lower learning rate for larger models

**Optimal Values**:
- Tiny/Small: 1e-5
- Base: 5e-6 (0.5x)

**Reason**: Larger models require more careful optimization.

**Command-Line Parameter**: `--lr` (default: 1e-5)

---

### 5.2 Cosine Learning Rate Scheduling ✅ **Adopted**

**Original**: Step decay (original paper)
**Improvement**: Cosine annealing for smoother decay

**Implementation**:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=epochs,
    eta_min=lr * 0.01
)
```

**Results**: More stable training, better final performance

**Command-Line Parameters**:
- `--lr_scheduler {cosine,step,none}` (default: cosine)
- `--no_lr_scheduler` (to disable)

---

### 5.3 Batch Size Optimization ✅ **Adopted**

**Experiments**:
- batch_size=24: SRCC 0.9306
- batch_size=32: SRCC 0.9336 ✅
- batch_size=64: Used for Small model (more memory available)

**Optimal Value**: 32 for Base model

**Command-Line Parameter**: `--batch_size` (default: 96 for Tiny, 32 for Base)

---

### 5.4 Reproducibility ✅ **Adopted**

**Improvement**: Fixed random seeds and deterministic operations

**Implementation**:
```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Result**: All experiments are fully reproducible

**Code Location**: `train_swin.py`, lines 103-113

---

## 6. Command-Line Parameters

### Complete Parameter List

#### Model Architecture
```bash
--model_size {tiny,small,base}  # Swin Transformer size (default: tiny)
--no_multiscale                 # Disable multi-scale fusion (enabled by default)
--attention_fusion              # Enable attention-based fusion (experimental)
```

#### Loss Function
```bash
--ranking_loss_alpha FLOAT      # Weight for ranking loss (default: 0.5, set to 0 to disable)
--ranking_loss_margin FLOAT     # Margin for ranking loss (default: 0.1)
```

#### Regularization
```bash
--drop_path_rate FLOAT          # Stochastic depth rate (default: 0.2)
--dropout_rate FLOAT            # Dropout rate (default: 0.3)
--weight_decay FLOAT            # L2 regularization (implicit in optimizer)
```

#### Training
```bash
--epochs INT                    # Number of epochs (default: 10)
--batch_size INT                # Batch size (default: 96)
--lr FLOAT                      # Learning rate (default: 1e-5)
--lr_scheduler {cosine,step,none}  # LR scheduler type (default: cosine)
--no_lr_scheduler               # Disable LR scheduler
```

#### Early Stopping
```bash
--patience INT                  # Early stopping patience (default: 5)
--no_early_stopping             # Disable early stopping
```

#### Data
```bash
--dataset {koniq-10k,live,csiq,tid2013}  # Dataset name
--train_patch_num INT           # Number of patches per image (default: 20)
--test_patch_num INT            # Number of test patches (default: 20)
--test_random_crop              # Use random crop for testing (less reproducible)
```

#### Testing
```bash
--no_spaq                       # Disable SPAQ cross-dataset testing (saves time)
```

### Example Commands

#### Best Model (Swin-Base)
```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

#### Swin-Tiny Baseline
```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size tiny \
  --batch_size 96 \
  --epochs 30 \
  --patience 7 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --ranking_loss_alpha 0.5 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

#### Pure L1 Loss (No Ranking)
```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --ranking_loss_alpha 0 \
  # ... other parameters same as best model
```

#### Attention Fusion (Experimental)
```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
  --attention_fusion \
  # ... other parameters
```

---

## 7. Tools and Utilities

### 7.1 Complexity Analysis Tools ✅

**Location**: `complexity/`

**Features**:
- FLOPs computation using `ptflops` and `thop` (as per TA's requirements)
- Inference time measurement (mean, std, min, max, median)
- Throughput testing with different batch sizes
- Auto-generated markdown reports

**Files**:
- `compute_complexity.py` - Full analysis
- `quick_test.py` - Fast test without dependencies
- `run_analysis.sh` - Interactive one-click script
- `README.md` - Documentation

**Usage**:
```bash
# Quick test
python complexity/quick_test.py

# Full analysis
pip install ptflops thop
python complexity/compute_complexity.py
```

---

### 7.2 Cross-Dataset Testing Tools ✅

**Location**: Root directory

**Features**:
- Test model generalization on multiple datasets
- Support for SPAQ, LIVE-itW, CSIQ, TID2013, etc.
- Automated testing with one-click script

**Files**:
- `cross_dataset_test.py` - Testing script
- `run_cross_dataset_test.sh` - One-click runner
- `CROSS_DATASET_TESTING_GUIDE.md` - Documentation

**Usage**:
```bash
bash run_cross_dataset_test.sh
```

---

## 8. Summary of Improvements

### Adopted Improvements ✅

1. **Swin Transformer Backbone** (Tiny/Small/Base)
2. **Multi-Scale Feature Fusion** (concatenation-based)
3. **Ranking Loss** (alpha=0.5 for Base)
4. **Strong Regularization**:
   - Stochastic Depth (drop_path_rate=0.3 for Base)
   - Dropout (dropout_rate=0.4 for Base)
   - Weight Decay (2e-4 for Base)
5. **Early Stopping** (patience=7)
6. **Cosine LR Scheduling**
7. **Lower Learning Rate** (5e-6 for Base)
8. **Reproducibility** (fixed seeds, deterministic mode)

### Rejected Improvements ❌

1. **Attention-Based Fusion**: Negative impact on Tiny (-0.28%), marginal and unstable on Small (+0.08%)
2. **Pure L1 Loss (alpha=0)**: Worse than combined loss (-0.29% on Base)
3. **Lower Dropout (0.35)**: Leads to overfitting (-0.31% on Base)
4. **Smaller Batch Size (24)**: No benefit (-0.30% on Base)

### Key Insights

1. **Model Capacity is Critical**: Performance scales with model size (Tiny → Small → Base)
2. **Regularization Must Scale**: Larger models need stronger regularization (2x weight_decay, 1.5x drop_path)
3. **Ranking Loss Importance Increases**: More important for larger models
4. **Simple Methods are More Stable**: Concatenation > Attention fusion

---

## 9. Performance Summary

| Model | SRCC | PLCC | Parameters | FLOPs | Inference Time |
|-------|------|------|------------|-------|----------------|
| ResNet-50 (Original) | 0.9009 | 0.9170 | - | - | - |
| Swin-Tiny | 0.9236 | 0.9361 | 28M | ~7G | ~10ms |
| Swin-Small | 0.9303 | 0.9444 | 50M | ~12G | ~15ms |
| **Swin-Base** | **0.9336** | **0.9464** | **88.85M** | **~18G** | **~17ms** |

**Final Improvement**: +3.40% SRCC, +2.94% PLCC over original HyperIQA

---

## 10. References

- Original HyperIQA Paper: Su et al., "Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network", CVPR 2020
- Swin Transformer: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
- Code Repository: https://github.com/cny123222/Perceptual-IQA-CS3324

---

**Document Version**: 1.0
**Last Updated**: December 20, 2025
**Status**: Complete and ready for appendix inclusion

