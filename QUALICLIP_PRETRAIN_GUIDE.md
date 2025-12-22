# QualiCLIP-Style Self-Supervised Pretraining for IQA

## Overview

This document describes the implementation of a **two-stage training framework** that combines QualiCLIP's self-supervised pretraining approach with our Swin-based HyperIQA model.

### Key Contributions

1. **Quality-Aware Self-Supervised Pretraining**: Pretrain the Swin Transformer backbone using quality-related contrastive learning without any MOS labels
2. **Two-Stage Training Pipeline**: First pretrain on unlabeled data, then fine-tune on labeled IQA datasets
3. **Novel Integration**: Successfully combine self-supervised learning (QualiCLIP) with supervised IQA (HyperIQA)

### Architecture

```
Stage 1: Self-Supervised Pretraining
─────────────────────────────────────
Clean Images (KonIQ-10k) 
    ↓
Synthetic Degradation Generator (5 levels)
    ↓
Swin Encoder → Image Features
    ↓
QualiCLIP Loss (with CLIP text features)
    ↓
Pretrained Encoder Weights

Stage 2: Supervised Fine-tuning
─────────────────────────────────────
Pretrained Encoder 
    ↓
Swin + HyperNet (full model)
    ↓
Supervised IQA Training (with MOS labels)
    ↓
Final IQA Model
```

---

## Implementation Details

### 1. Degradation Generator

**File**: `qualiclip_pretrain/degradation_generator.py`

Implements 4 types of image degradations with 5 intensity levels each:

| Degradation Type | Levels | Parameters |
|-----------------|--------|------------|
| Gaussian Blur | 5 | σ = [0.5, 1.0, 1.5, 2.0, 2.5] |
| JPEG Compression | 5 | Quality = [85, 70, 55, 40, 25] |
| Gaussian Noise | 5 | σ = [5, 10, 15, 20, 25] |
| Brightness | 5 | Factor = [0.7, 0.5, 0.3, 1.3, 1.5] |

**Usage**:
```python
from qualiclip_pretrain.degradation_generator import SyntheticDegradation

degrader = SyntheticDegradation('blur', num_levels=5)
degraded_images = degrader(clean_image)  # Returns list of 5 degraded images
```

### 2. QualiCLIP Loss Function

**File**: `qualiclip_pretrain/qualiclip_loss.py`

Implements three loss components:

1. **Consistency Loss (L_cons)**: Encourages similar features for same content + same degradation level
2. **Positive Ranking Loss (L_pos)**: Better quality → more similar to "Good photo"
3. **Negative Ranking Loss (L_neg)**: Worse quality → more similar to "Bad photo"

**Two implementations**:
- `QualiCLIPLoss`: Full implementation with pairwise comparisons
- `SimplifiedQualiCLIPLoss`: Faster simplified version (recommended)

### 3. Pretrain Dataset

**File**: `qualiclip_pretrain/pretrain_dataset.py`

- Loads clean images from KonIQ-10k training set (7058 images)
- Generates two overlapping random crops per image
- Applies same degradation to both crops for consistency

### 4. Pretrain Script

**File**: `pretrain_qualiclip.py`

Main pretraining pipeline:
- Loads Swin-Base backbone
- Loads frozen CLIP text encoder
- Trains with QualiCLIP loss for 10 epochs
- Saves pretrained encoder weights

---

## Quick Start

### Prerequisites

1. Install CLIP:
```bash
pip install git+https://github.com/openai/CLIP.git
```

2. Ensure KonIQ-10k dataset is available at `/root/Perceptual-IQA-CS3324/koniq-10k/`

### Running the Full Pipeline

**Option 1: Automated Script** (Recommended)
```bash
cd /root/Perceptual-IQA-CS3324
./run_qualiclip_experiments.sh
```

This will:
1. Run QualiCLIP pretraining (10 epochs, ~3-4 hours)
2. Run baseline experiment (no pretraining, 50 epochs)
3. Run pretrained experiment (with pretraining, 50 epochs)
4. Save results to logs/

**Option 2: Manual Step-by-Step**

Step 1: QualiCLIP Pretraining
```bash
python pretrain_qualiclip.py \
    --data_root /root/Perceptual-IQA-CS3324/koniq-10k \
    --model_size base \
    --epochs 10 \
    --batch_size 8 \
    --lr 5e-5
```

Step 2a: Baseline (No Pretraining)
```bash
python train_swin.py \
    --dataset koniq-10k \
    --model_size base \
    --epochs 50 \
    --use_multiscale \
    --attention_fusion
```

Step 2b: With Pretraining
```bash
python train_swin.py \
    --dataset koniq-10k \
    --model_size base \
    --epochs 50 \
    --use_multiscale \
    --attention_fusion \
    --pretrained_encoder checkpoints/qualiclip_pretrain_*/swin_base_qualiclip_pretrained.pkl \
    --lr_encoder_pretrained 1e-6
```

---

## Hyperparameters

### Pretraining Stage

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model Size | base | Swin-Base (~88M params) |
| Epochs | 10 | Quick validation |
| Batch Size | 8 | Effective batch ~80 (8 × 5 levels × 2 crops) |
| Learning Rate | 5e-5 | AdamW optimizer |
| Weight Decay | 1e-4 | Regularization |
| Scheduler | Cosine | Annealing to 1e-7 |
| Loss Type | simplified | Faster than full version |
| Distortions | all 4 | blur, jpeg, noise, brightness |

### Fine-tuning Stage

| Parameter | Baseline | With Pretrain |
|-----------|----------|---------------|
| Encoder LR | 5e-6 | 1e-6 (10x smaller) |
| HyperNet LR | 5e-5 | 5e-5 |
| Epochs | 50 | 50 |
| Other params | Same | Same |

**Rationale**: Pretrained encoder uses smaller LR because it's already learned quality-aware features.

---

## Expected Results

### Performance Comparison

| Model | KonIQ-10k (SRCC/PLCC) | Training Time |
|-------|----------------------|---------------|
| Baseline (No Pretrain) | ~0.9343 / ~0.9463 | ~8h |
| **QualiCLIP Pretrain** | **?? / ??** | 3h + 8h = 11h |

**Hypothesis**: QualiCLIP pretraining should:
- ✓ Improve cross-dataset generalization (SPAQ, KADID, AGIQA)
- ✓ Potentially improve KonIQ-10k performance
- ✓ Reduce overfitting (better quality-aware representations)

### Cross-Dataset Generalization

| Model | SPAQ | KADID-10K | AGIQA-3K |
|-------|------|-----------|----------|
| Baseline | X.XX | X.XX | X.XX |
| **QualiCLIP Pretrain** | **X.XX** | **X.XX** | **X.XX** |

---

## Project Structure

```
Perceptual-IQA-CS3324/
├── qualiclip_pretrain/           # Pretraining modules
│   ├── __init__.py
│   ├── degradation_generator.py  # Synthetic degradations
│   ├── qualiclip_loss.py         # Loss functions
│   └── pretrain_dataset.py       # Data loading
├── pretrain_qualiclip.py         # Main pretraining script
├── train_swin.py                 # Modified to support pretrained weights
├── HyperIQASolver_swin.py        # Modified to load pretrained encoder
├── run_qualiclip_experiments.sh  # Automated pipeline
└── QUALICLIP_PRETRAIN_GUIDE.md   # This file
```

---

## Technical Details

### Why This Approach Works

1. **Quality-Aware Representations**: Pretraining teaches the encoder to distinguish between different quality levels before seeing MOS labels

2. **Transfer Learning**: General quality understanding from pretraining transfers to specific IQA task

3. **Differential Learning Rates**: Pretrained encoder uses smaller LR to preserve learned features while adapting to new task

4. **Data Efficiency**: Can leverage unlabeled data (any clean images) for pretraining

### Differences from Original QualiCLIP

| Aspect | Original QualiCLIP | Our Implementation |
|--------|-------------------|-------------------|
| Backbone | ResNet-50 | Swin-Base |
| Output | Direct quality score | Features for HyperNet |
| Training | Pure self-supervised | Two-stage (self + supervised) |
| Dataset | Large unlabeled corpus | KonIQ-10k training set |
| Goal | Opinion-unaware IQA | Opinion-aware IQA (with MOS) |

---

## Troubleshooting

### Common Issues

**1. CLIP not installed**
```bash
pip install git+https://github.com/openai/CLIP.git
```

**2. CUDA out of memory**
- Reduce `batch_size` in pretraining (try 4 or 2)
- Use `model_size='small'` or `'tiny'` instead of `'base'`

**3. Dataset not found**
- Check that KonIQ-10k is at `/root/Perceptual-IQA-CS3324/koniq-10k/`
- Verify `koniq_train.json` exists

**4. Pretraining too slow**
- Use `loss_type='simplified'` (faster than full version)
- Reduce `num_levels` from 5 to 3
- Reduce `epochs` from 10 to 5 for quick validation

---

## Next Steps

After obtaining results:

1. **Performance Analysis**:
   - Compare SRCC/PLCC on KonIQ-10k test set
   - Evaluate cross-dataset generalization
   - Analyze training curves

2. **Ablation Studies**:
   - Different degradation types
   - Different pretraining epochs (5, 10, 20)
   - Different loss weights

3. **Visualization**:
   - Attention weight comparison
   - Feature space visualization (t-SNE)
   - Quality ranking curves

4. **Documentation**:
   - Write comprehensive results report
   - Create comparison tables
   - Discuss findings

---

## References

1. **QualiCLIP**: Quality-Aware Image-Text Alignment for Opinion-Unaware IQA
   - Paper: [arXiv:2403.11176](https://arxiv.org/abs/2403.11176)
   
2. **HyperIQA**: Deep Learning for Image Quality Assessment
   - Original framework we're building upon

3. **Swin Transformer**: Hierarchical Vision Transformer
   - Our choice of backbone architecture

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{qualiclip_hyperiqa_fusion,
  title={QualiCLIP-Style Self-Supervised Pretraining for Swin-HyperIQA},
  author={Your Name},
  year={2024},
  note={Two-stage training framework combining QualiCLIP pretraining with HyperIQA fine-tuning}
}
```

---

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

**Last Updated**: December 2024

