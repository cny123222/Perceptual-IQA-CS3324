# Anti-Overfitting Strategy Guide

## 🎯 Problem Statement

The baseline model achieved strong performance (SRCC=0.9195, PLCC=0.9342) after just **1 epoch**, but suffered from **severe overfitting** in subsequent epochs:

```
Epoch 1: Test SRCC 0.9195 ⭐ BEST
Epoch 2: Test SRCC 0.9185 ↓
Epoch 3: Test SRCC 0.9145 ↓↓
Epoch 4: Test SRCC 0.9174 ↓
```

**Root Cause**: High model capacity (Swin-Tiny 28.8M + HyperNet 0.5M) with insufficient regularization on limited data (7,046 images).

---

## ✅ Implemented Solutions

Based on expert AI recommendations, we implemented a **comprehensive 3-phase anti-overfitting strategy**:

### **Phase 1: Regularization** 🛡️

#### 1. **AdamW Optimizer with Weight Decay**

**What**: Switched from `Adam` to `AdamW` optimizer
- **Weight Decay**: `1e-4` (default in config: `5e-4`)
- **Why AdamW**: Decouples weight decay from gradient updates, more effective regularization

**Code Change** (`HyperIQASolver_swin.py`):
```python
# Before: torch.optim.Adam(paras, weight_decay=self.weight_decay)
# After:
self.solver = torch.optim.AdamW(paras, weight_decay=self.weight_decay)
```

**Expected Impact**: Prevents weights from growing too large, reducing overfitting by ~10-15%

---

#### 2. **Dropout in HyperNet and TargetNet**

**What**: Added `nn.Dropout(0.3)` to both networks
- **HyperNet**: After `hyper_in_feat` computation
- **TargetNet**: After each layer (l1, l2, l3)

**Code Changes** (`models_swin.py`):
```python
class HyperNet:
    def __init__(self, ..., dropout_rate=0.3):
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, img):
        hyper_in_feat = self.conv1(...).view(...)
        hyper_in_feat = self.dropout(hyper_in_feat)  # Apply dropout

class TargetNet:
    def __init__(self, paras, dropout_rate=0.3):
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        q = self.l1(x)
        q = self.dropout(q)  # After each layer
        q = self.l2(q)
        q = self.dropout(q)
        q = self.l3(q)
        q = self.dropout(q)
        q = self.l4(q).squeeze()
        return q
```

**Expected Impact**: Randomly drops 30% of activations during training, forcing the network to learn redundant representations. Should reduce overfitting by ~20-30%.

---

#### 3. **Stochastic Depth (Drop Path) in Swin Transformer**

**What**: Enabled `drop_path_rate=0.2` in Swin Transformer backbone
- **How it works**: Randomly drops entire residual blocks during training
- **Swin-specific**: This is a standard regularization technique for Vision Transformers

**Code Changes** (`models_swin.py`):
```python
class SwinBackbone:
    def __init__(self, lda_out_channels, in_chn, drop_path_rate=0.2):
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            features_only=True,
            drop_path_rate=drop_path_rate,  # Enable stochastic depth
        )
```

**Expected Impact**: Reduces overfitting in the backbone by ~15-20%. Particularly effective for deep networks like Swin Transformer.

---

### **Phase 2: Data Augmentation** 🖼️

#### 1. **RandomHorizontalFlip** ✅

**Status**: Already present in baseline code!
- **Effect**: Doubles effective dataset size (mirror symmetry preserves quality)

**Code** (`data_loader.py`):
```python
transforms.RandomHorizontalFlip()  # Already implemented!
```

---

#### 2. **Light ColorJitter** (New)

**What**: Added conservative color perturbations
- **brightness**: ±10% (0.1)
- **contrast**: ±10% (0.1)
- **saturation**: ±10% (0.1)
- **hue**: ±5% (0.05)

**Why conservative**: Must not alter perceived image quality significantly

**Code Changes** (`data_loader.py`):
```python
transforms.ColorJitter(
    brightness=0.1, 
    contrast=0.1, 
    saturation=0.1, 
    hue=0.05
)
```

**Expected Impact**: Forces model to learn quality features invariant to slight color variations. Should improve generalization by ~5-10%.

---

### **Phase 3: Training Optimization** ⚙️

#### 1. **Gradient Clipping**

**What**: Clip gradients to `max_norm=1.0` before optimizer step
- **Purpose**: Prevent gradient explosion, stabilize training

**Code Changes** (`HyperIQASolver_swin.py`):
```python
total_loss.backward()
torch.nn.utils.clip_grad_norm_(self.model_hyper.parameters(), max_norm=1.0)
self.solver.step()
```

**Expected Impact**: More stable training, especially in early epochs. May allow using slightly higher learning rates safely.

---

#### 2. **Learning Rate Strategy** (Already Implemented)

**Current Best Practice**:
- **Scheduler**: CosineAnnealingLR (default in `train_swin.py`)
- **HyperNet LR**: `1e-4` (suggested, currently `2e-4`)
- **Backbone LR**: `1e-5` (suggested, currently `2e-5`)

**Note**: To use lower LR, run with:
```bash
--lr 1e-5  # This will set backbone LR, HyperNet will be 1e-4 (10× ratio)
```

---

## 🚀 Usage

### **Command-Line Arguments**

New arguments added to control regularization:

```bash
--drop_path_rate FLOAT    # Stochastic depth rate (default: 0.2)
--dropout_rate FLOAT       # Dropout rate (default: 0.3)
--weight_decay FLOAT       # Weight decay (default: 5e-4)
```

### **Recommended Training Command**

```bash
python train_swin.py \
  --dataset koniq-10k \
  --epochs 30 \
  --patience 7 \
  --ranking_loss_alpha 0 \
  --batch_size 96 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

### **Ablation Study Commands**

To test the impact of each component:

#### **Baseline (No Regularization)**
```bash
python train_swin.py \
  --dataset koniq-10k \
  --dropout_rate 0.0 \
  --drop_path_rate 0.0 \
  --weight_decay 0.0 \
  --no_spaq
```

#### **Phase 1 Only (Regularization)**
```bash
python train_swin.py \
  --dataset koniq-10k \
  --dropout_rate 0.3 \
  --drop_path_rate 0.2 \
  --weight_decay 1e-4 \
  --no_spaq
```

#### **Phase 1 + 2 (Regularization + Augmentation)**
```bash
# ColorJitter is now automatic, just use normal training
python train_swin.py \
  --dataset koniq-10k \
  --dropout_rate 0.3 \
  --drop_path_rate 0.2 \
  --weight_decay 1e-4 \
  --no_spaq
```

---

## 📊 Expected Results

### **Baseline (Before)**
```
Epoch 1: SRCC 0.9195, PLCC 0.9342 ⭐
Epoch 2: SRCC 0.9185, PLCC 0.9320 ↓
Epoch 3: SRCC 0.9145, PLCC 0.9275 ↓↓
```

### **With Anti-Overfitting (Expected)**
```
Epoch 1: SRCC 0.9100, PLCC 0.9300 (may be slightly lower due to regularization)
Epoch 2: SRCC 0.9150, PLCC 0.9330 ↑
Epoch 3: SRCC 0.9180, PLCC 0.9350 ↑
Epoch 4: SRCC 0.9200, PLCC 0.9360 ↑ ⭐ NEW BEST
```

**Key Expectations**:
1. **First epoch may be slightly worse** (regularization prevents overfitting, but slows initial learning)
2. **Performance continues improving** after epoch 1 (instead of degrading)
3. **Peak performance reached later** (epoch 4-6 instead of epoch 1)
4. **Peak performance may be higher** (better generalization)

---

## 🧪 Experimental Design

As suggested by the AI expert, run experiments in phases:

### **Experiment Group 1: Baseline**
- Current best model (master branch)
- **Purpose**: Reference point

### **Experiment Group 2: Regularization Only**
- AdamW + Dropout + Stochastic Depth
- **Purpose**: Measure regularization impact

### **Experiment Group 3: Regularization + Augmentation**
- Group 2 + ColorJitter
- **Purpose**: Measure combined effect

### **Experiment Group 4: Full Strategy** (Recommended)
- Group 3 + Lower LR + Gradient Clipping
- **Purpose**: Optimal configuration

---

## 🔍 Monitoring and Analysis

### **What to Watch During Training**

1. **Train vs. Test Gap**
   - **Before**: Train SRCC (0.9747) >> Test SRCC (0.9174) at epoch 4
   - **Goal**: Reduce this gap to < 0.03

2. **Test Performance Trend**
   - **Before**: Peaks at epoch 1, then declines
   - **Goal**: Peaks at epoch 4-6, continuous improvement

3. **Loss Convergence**
   - **Before**: Train loss drops very fast, test loss increases
   - **Goal**: Both losses decrease smoothly

### **Success Criteria**

✅ Test SRCC keeps improving for at least 3-4 epochs  
✅ Best performance occurs after epoch 3  
✅ Final test SRCC ≥ 0.920 (ideally > baseline's 0.9195)  
✅ Train-test gap < 0.04 at convergence  

---

## 🎓 Theoretical Background

### **Why Does This Work?**

1. **Regularization prevents memorization**
   - Without regularization: Model memorizes training set noise
   - With regularization: Model forced to learn general patterns

2. **Dropout creates ensemble effect**
   - Each training step uses a different "sub-network"
   - Final model is an implicit ensemble of many networks

3. **Data augmentation expands data manifold**
   - Original: 7,046 unique images
   - With flip + ColorJitter: Effectively millions of variations

4. **Gradient clipping stabilizes optimization**
   - Prevents large parameter updates that cause overfitting
   - Allows higher learning rates without instability

---

## 📝 Implementation Details

### **Files Modified**

1. **`models_swin.py`**
   - Added `drop_path_rate` parameter to `SwinBackbone`
   - Added `dropout_rate` parameter to `HyperNet` and `TargetNet`
   - Applied dropout in forward passes

2. **`HyperIQASolver_swin.py`**
   - Changed optimizer from `Adam` to `AdamW`
   - Added gradient clipping before `optimizer.step()`
   - Passed regularization parameters to model

3. **`data_loader.py`**
   - Added `ColorJitter` to training transforms

4. **`train_swin.py`**
   - Added `--drop_path_rate` argument
   - Added `--dropout_rate` argument

---

## 💡 Tips for Further Improvement

If overfitting persists:

### **Stronger Regularization**
```bash
--dropout_rate 0.5          # Increase from 0.3
--drop_path_rate 0.3        # Increase from 0.2
--weight_decay 5e-4         # Increase from 1e-4
```

### **More Data Augmentation**
Consider adding (carefully):
- `transforms.RandomAffine(degrees=5, translate=(0.05, 0.05))`
- `transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))`

### **Architecture Changes**
- Try `--no_multiscale` (reduce model capacity)
- Consider using ResNet-50 instead of Swin-Tiny (fewer parameters)

### **Training Strategy**
- Reduce `--train_patch_num` from 20 to 10 (less data redundancy)
- Increase `--batch_size` to 128 (more stable gradients)

---

## 🎯 Summary

**Problem**: Severe overfitting after epoch 1  
**Solution**: 3-phase anti-overfitting strategy  
**Expected**: Performance peak at epoch 4-6, improved generalization  
**Status**: ✅ Fully implemented, ready for testing  

**Next Step**: Run the recommended training command and compare with baseline!

---

## 📚 References

- **Weight Decay**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, ICLR 2019)
- **Dropout**: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (Srivastava et al., JMLR 2014)
- **Stochastic Depth**: "Deep Networks with Stochastic Depth" (Huang et al., ECCV 2016)
- **Data Augmentation for IQA**: Generally safe for flip; be cautious with color/blur

---

**Created**: 2024-12-18  
**Branch**: `anti-overfitting`  
**Based on**: AI expert suggestions in `suggestions5.md`

