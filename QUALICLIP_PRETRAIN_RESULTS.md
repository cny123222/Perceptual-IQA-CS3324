# QualiCLIP Pretraining Experimental Results

## Experiment Overview

**Date**: December 2024  
**Branch**: `feature/qualiclip-pretrain`  
**Goal**: Evaluate the effect of QualiCLIP-style self-supervised pretraining on Swin-HyperIQA model performance

---

## Methodology

### Two-Stage Training Framework

**Stage 1: Quality-Aware Self-Supervised Pretraining**
- Dataset: KonIQ-10k training set (7058 images, unlabeled)
- Model: Swin-Base backbone only
- Loss: QualiCLIP loss (consistency + ranking)
- Degradations: Gaussian Blur, JPEG Compression, Gaussian Noise, Brightness
- Epochs: 10
- Time: ~3-4 hours

**Stage 2: Supervised Fine-tuning**
- Dataset: KonIQ-10k (with MOS labels)
- Model: Swin-Base + HyperNet (full model)
- Loss: L1 loss (MAE)
- Epochs: 50
- Time: ~8 hours

---

## Experimental Setup

### Hardware
- GPU: [TO BE FILLED]
- CPU: [TO BE FILLED]
- RAM: [TO BE FILLED]

### Software
- PyTorch: [VERSION]
- CUDA: [VERSION]
- Python: 3.x

### Hyperparameters

**Pretraining Stage:**
```
Model Size:      base (88M parameters)
Batch Size:      8 (effective ~80 after degradation)
Learning Rate:   5e-5
Optimizer:       AdamW
Scheduler:       CosineAnnealingLR
Weight Decay:    1e-4
Loss Type:       Simplified QualiCLIP
Distortions:     blur, jpeg, noise, brightness
Levels:          5
```

**Fine-tuning Stage:**
```
Epochs:          50
Batch Size:      96
Optimizer:       AdamW
Weight Decay:    2e-4
Multi-Scale:     Enabled
Attention:       Enabled
Drop Path:       0.2
Dropout:         0.3
LR Scheduler:    Cosine
Early Stopping:  Enabled (patience=10)

Baseline:
  Encoder LR:    5e-6
  HyperNet LR:   5e-5

With Pretrain:
  Encoder LR:    1e-6 (smaller)
  HyperNet LR:   5e-5
```

---

## Results

### Main Results: KonIQ-10k Test Set

| Model | SRCC | PLCC | Source | Training Time |
|-------|------|------|--------|---------------|
| Baseline (No Pretrain) | [TO BE FILLED] | [TO BE FILLED] | Test | ~8h |
| **QualiCLIP Pretrain** | **[TO BE FILLED]** | **[TO BE FILLED]** | Test | 3h + 8h |
| Previous Best | 0.9343 | 0.9463 | Historical | - |
| HyperIQA Paper | 0.906 | 0.917 | Paper | - |

**Improvement**:
- ΔSRCC = [TO BE CALCULATED]
- ΔPLCC = [TO BE CALCULATED]

### Cross-Dataset Generalization

Testing on unseen datasets (models trained on KonIQ-10k only):

| Model | SPAQ (SRCC/PLCC) | KADID-10K (SRCC/PLCC) | AGIQA-3K (SRCC/PLCC) |
|-------|------------------|----------------------|---------------------|
| Baseline | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| **QualiCLIP Pretrain** | **[TO BE FILLED]** | **[TO BE FILLED]** | **[TO BE FILLED]** |

**Key Observations**:
- [TO BE FILLED after experiments]

---

## Training Curves

### Pretraining Stage

**Loss Components** (10 epochs):
```
Epoch | Total Loss | Consistency | Ranking
------|------------|-------------|--------
1     | [PENDING]  | [PENDING]   | [PENDING]
2     | [PENDING]  | [PENDING]   | [PENDING]
...   | ...        | ...         | ...
10    | [PENDING]  | [PENDING]   | [PENDING]
```

### Fine-tuning Stage

**Baseline (No Pretrain)**:
```
Epoch | Train Loss | Train SRCC | Test SRCC | Test PLCC
------|------------|------------|-----------|----------
1     | [PENDING]  | [PENDING]  | [PENDING] | [PENDING]
...   | ...        | ...        | ...       | ...
50    | [PENDING]  | [PENDING]  | [PENDING] | [PENDING]
```

**With QualiCLIP Pretrain**:
```
Epoch | Train Loss | Train SRCC | Test SRCC | Test PLCC
------|------------|------------|-----------|----------
1     | [PENDING]  | [PENDING]  | [PENDING] | [PENDING]
...   | ...        | ...        | ...       | ...
50    | [PENDING]  | [PENDING]  | [PENDING] | [PENDING]
```

---

## Analysis

### Does Pretraining Help?

**Expected Outcomes**:
1. ✓/✗ Better cross-dataset generalization
2. ✓/✗ Improved KonIQ-10k performance
3. ✓/✗ Faster convergence
4. ✓/✗ Better quality-aware features

**Actual Results**:
- [TO BE FILLED after experiments]

### Why Does It Work (or Not)?

**If Successful**:
- Quality-aware representations learned during pretraining
- Transfer of general quality understanding to specific IQA task
- Better initialization compared to ImageNet pretraining

**If Not Successful**:
- Domain mismatch between synthetic degradations and real distortions
- Insufficient pretraining epochs
- Loss function may need tuning
- Differential learning rate may need adjustment

---

## Ablation Studies

### Effect of Pretraining Epochs

| Epochs | SRCC | PLCC | Notes |
|--------|------|------|-------|
| 0 (Baseline) | X.XX | X.XX | No pretraining |
| 5 | X.XX | X.XX | Quick pretrain |
| 10 | X.XX | X.XX | Default |
| 20 | X.XX | X.XX | Longer pretrain |

### Effect of Degradation Types

| Distortions | SRCC | PLCC | Notes |
|-------------|------|------|-------|
| Blur only | X.XX | X.XX | Single type |
| JPEG only | X.XX | X.XX | Single type |
| All 4 types | X.XX | X.XX | Default |

### Effect of Loss Type

| Loss Type | SRCC | PLCC | Training Time |
|-----------|------|------|---------------|
| Full QualiCLIP | X.XX | X.XX | Slower |
| Simplified | X.XX | X.XX | Faster (default) |

---

## Visualizations

### 1. Training Curves
[TO BE ADDED: Line plots showing loss and metrics over epochs]

### 2. Feature Space Analysis
[TO BE ADDED: t-SNE visualization of learned features]

### 3. Attention Maps
[TO BE ADDED: Comparison of attention patterns]

### 4. Quality Ranking
[TO BE ADDED: Model predictions vs ground truth quality ordering]

---

## Computational Cost

### Training Time Breakdown

| Stage | Duration | GPU Util | Notes |
|-------|----------|----------|-------|
| Pretraining | ~3-4h | ~90% | 10 epochs, batch=8 |
| Baseline Train | ~8h | ~90% | 50 epochs, batch=96 |
| Pretrain Train | ~8h | ~90% | 50 epochs, batch=96 |
| **Total** | **~19-20h** | - | For complete comparison |

### Model Size

| Component | Parameters | Size on Disk |
|-----------|------------|--------------|
| Swin-Base Encoder | ~88M | ~350 MB |
| HyperNet | ~24M | ~95 MB |
| **Total Model** | ~112M | ~445 MB |

---

## Conclusions

### Key Findings
1. [TO BE FILLED after experiments]
2. [TO BE FILLED after experiments]
3. [TO BE FILLED after experiments]

### Recommendations
- [TO BE FILLED after experiments]

### Future Work
1. **Longer Pretraining**: Try 20-50 epochs on larger dataset
2. **More Degradations**: Add more realistic distortion types
3. **Different Architectures**: Try with different backbones
4. **Multi-Dataset Pretraining**: Use multiple IQA datasets for pretraining
5. **Fine-grained Loss Tuning**: Optimize loss component weights

---

## Code and Reproducibility

### Repository
- Branch: `feature/qualiclip-pretrain`
- Commit: [TO BE FILLED]

### Running the Experiments
```bash
# Clone and setup
git checkout feature/qualiclip-pretrain
pip install git+https://github.com/openai/CLIP.git

# Run full pipeline
./run_qualiclip_experiments.sh

# Or run manually
# 1. Pretraining
python pretrain_qualiclip.py --model_size base --epochs 10

# 2. Baseline
python train_swin.py --dataset koniq-10k --model_size base --epochs 50

# 3. With pretrain
python train_swin.py --dataset koniq-10k --model_size base --epochs 50 \
    --pretrained_encoder checkpoints/*/swin_base_qualiclip_pretrained.pkl \
    --lr_encoder_pretrained 1e-6
```

### Checkpoints
- Pretrained encoder: `checkpoints/qualiclip_pretrain_*/swin_base_qualiclip_pretrained.pkl`
- Baseline model: `checkpoints/koniq-10k-swin_*/best_model_*.pkl`
- Pretrained model: `checkpoints/koniq-10k-swin_*/best_model_*.pkl`

---

## References

1. **QualiCLIP**: Agnolucci et al., "Quality-Aware Image-Text Alignment for Opinion-Unaware Image Quality Assessment", arXiv:2403.11176, 2024

2. **HyperIQA**: Su et al., "Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network", CVPR 2020

3. **Swin Transformer**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021

---

## Appendix

### A. Full Configuration Files

**pretrain_config.json**:
```json
{
  "model_size": "base",
  "epochs": 10,
  "batch_size": 8,
  "lr": 5e-5,
  "weight_decay": 1e-4,
  "num_levels": 5,
  "distortion_types": ["blur", "jpeg", "noise", "brightness"],
  "loss_type": "simplified"
}
```

**finetune_baseline_config.json**:
```json
{
  "dataset": "koniq-10k",
  "model_size": "base",
  "epochs": 50,
  "batch_size": 96,
  "lr": 5e-6,
  "lr_ratio": 10,
  "use_multiscale": true,
  "use_attention": true,
  "pretrained_encoder": null
}
```

**finetune_pretrained_config.json**:
```json
{
  "dataset": "koniq-10k",
  "model_size": "base",
  "epochs": 50,
  "batch_size": 96,
  "lr": 5e-6,
  "lr_ratio": 10,
  "use_multiscale": true,
  "use_attention": true,
  "pretrained_encoder": "checkpoints/.../swin_base_qualiclip_pretrained.pkl",
  "lr_encoder_pretrained": 1e-6
}
```

### B. Environment Setup

**requirements_qualiclip.txt**:
```
torch>=1.10.0
torchvision>=0.11.0
timm>=0.4.12
scipy>=1.7.0
numpy>=1.21.0
Pillow>=8.3.0
tqdm>=4.62.0
git+https://github.com/openai/CLIP.git
```

---

**Status**: EXPERIMENTS PENDING - Results to be filled after running experiments

**Last Updated**: [DATE TO BE FILLED]

