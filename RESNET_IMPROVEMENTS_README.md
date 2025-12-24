# ResNet50 + Improvements 实验

## 🎯 **实验目的**

验证我们的改进（Multi-scale Feature Fusion + Channel Attention）是否对CNN backbone（ResNet50）也有效，从而证明方法的普适性。

---

## 📁 **文件说明**

### **核心代码**：
- **`models_resnet_improved.py`** - ResNet50改进版模型
  - 包含3个配置：Baseline, +Multi-scale, +Multi-scale+Attention
  - 完全兼容原始HyperIQA的TargetNet设计
  
- **`train_resnet_improved.py`** - 训练脚本
  - 支持所有3个配置
  - 使用与SMART-IQA相同的训练策略
  
- **`run_resnet_ablation.sh`** - 一键运行3个消融实验
  - 自动化运行所有实验
  - 自动提取结果
  - 约4.5小时完成

---

## 🚀 **快速开始**

### **方法1：运行完整消融实验（推荐）**

```bash
# 一键运行3个实验
bash run_resnet_ablation.sh
```

这将依次运行：
1. ResNet50 Baseline (Single-scale, No attention)
2. ResNet50 + Multi-scale
3. ResNet50 + Multi-scale + Attention

### **方法2：单独运行某个实验**

```bash
# Baseline
python3 train_resnet_improved.py \
    --dataset koniq-10k \
    --data_path ./koniq-10k \
    --epochs 10 \
    --lr 1e-4 \
    --batch_size 32 \
    --no_color_jitter \
    --test_random_crop \
    --save_model

# + Multi-scale
python3 train_resnet_improved.py \
    --dataset koniq-10k \
    --use_multiscale \
    --epochs 10 \
    --lr 1e-4 \
    --save_model

# + Multi-scale + Attention
python3 train_resnet_improved.py \
    --dataset koniq-10k \
    --use_multiscale \
    --use_attention \
    --epochs 10 \
    --lr 1e-4 \
    --save_model
```

---

## 📊 **预期结果**

### **假设1：改进有效（乐观）**
```
ResNet50 Baseline            0.8998  (已测得)
ResNet50 + Multi-scale       0.9050  (+0.52%)
ResNet50 + Multi + Attention 0.9120  (+1.35%)
```
**结论**：改进具有普适性 ✅

### **假设2：改进有限（中性）**
```
ResNet50 Baseline            0.8998
ResNet50 + Multi-scale       0.9010  (+0.13%)
ResNet50 + Multi + Attention 0.9040  (+0.47%)
```
**结论**：Swin的层次化特征更适合我们的方法 ✅

### **假设3：改进无效（悲观）**
```
ResNet50 Baseline            0.8998
ResNet50 + Multi-scale       0.8995  (-0.03%)
ResNet50 + Multi + Attention 0.9000  (+0.02%)
```
**结论**：改进专为Transformer设计，需要hierarchical features ✅

**三种结果都有论文价值！**

---

## 🔧 **模型架构**

### **配置1：Baseline**
```
ResNet50 (pretrained) → Stage 4 only → HyperNet → TargetNet → Score
```
- 参数量：25.62M
- 与原始HyperIQA相同

### **配置2：+ Multi-scale**
```
ResNet50 → [Stage 1,2,3,4] 
         → Adaptive Pool (7×7)
         → Conv 1×1 + BN + ReLU
         → Concatenate
         → HyperNet → TargetNet → Score
```
- 参数量：28.12M
- 融合4个stage的features

### **配置3：+ Multi-scale + Attention**
```
ResNet50 → [Stage 1,2,3,4] 
         → Adaptive Pool (7×7)
         → Conv 1×1 + BN + ReLU
         → Channel Attention (weights: w1,w2,w3,w4)
         → Weighted Concatenate
         → HyperNet → TargetNet → Score
```
- 参数量：28.65M
- 动态加权fusion

---

## 📝 **实验设置**

### **超参数**（与原始ResNet50 baseline一致）：
- Learning Rate: 1e-4
- Batch Size: 32
- Epochs: 10
- Train Patches: 25
- Test Patches: 25
- ColorJitter: ❌ Disabled
- Test Crop: RandomCrop ✅
- Weight Decay: 1e-4
- Dropout: 0.3

### **数据集**：
- KonIQ-10k
- Train: 7,046 images
- Test: 2,010 images

---

## 📂 **输出文件**

### **日志文件**：
```
logs/resnet_ablation_YYYYMMDD_HHMMSS/
├── exp1_baseline.log               # Baseline实验日志
├── exp2_multiscale.log             # Multi-scale实验日志
└── exp3_multiscale_attention.log   # 完整改进实验日志
```

### **模型权重**：
```
checkpoints/
├── resnet_improved_ss_noatt_best.pth   # Baseline
├── resnet_improved_ms_noatt_best.pth   # + Multi-scale
└── resnet_improved_ms_att_best.pth     # + Multi-scale + Attention
```

---

## 🧪 **测试模型**

```bash
# 测试模型forward pass
python3 models_resnet_improved.py
```

这将测试所有3个配置，输出：
- 参数量
- Forward pass成功与否
- Attention weights（如果有）

---

## 📊 **与SMART-IQA的对比**

| 模型 | Backbone | Multi-scale | Attention | Params | SRCC (预期) |
|-----|---------|------------|-----------|--------|------------|
| **HyperIQA** | ResNet50 | ❌ | ❌ | 25M | 0.8998 |
| **ResNet+Ours** | ResNet50 | ✅ | ✅ | 28.7M | **0.90-0.91?** |
| **SMART-IQA** | Swin-Base | ✅ | ✅ | 88M | **0.9378** ✅ |

**关键问题**：ResNet+改进 vs SMART-IQA的差距是多少？
- 如果差距大（0.91 vs 0.94）→ Swin的层次化特征是关键
- 如果差距小（0.92 vs 0.94）→ 改进本身贡献更大

---

## 💡 **论文中的呈现**

### **如果结果好（+1-2%）**：
```latex
We verify the generality of our improvements by applying them to ResNet50.
Results show +1.35% improvement, demonstrating that our method benefits
CNN backbones. However, Swin Transformer achieves +3.80% improvement,
suggesting that hierarchical vision features are more suitable for
quality-aware multi-scale fusion.
```

### **如果结果一般（+0.3-0.5%）**：
```latex
To understand the contribution of backbone architecture, we apply our
improvements to ResNet50. The gain (+0.47%) is much smaller than with
Swin Transformer (+3.80%), indicating that hierarchical, self-attention
based features are crucial for our method's success.
```

### **如果结果不好（<0.3%）**：
```latex
Interestingly, applying the same improvements to ResNet50 shows minimal
gains (<0.3%), while Swin Transformer benefits significantly (+3.80%).
This suggests that our multi-scale attention mechanism specifically
leverages the hierarchical, window-based features of vision transformers.
```

---

## ⏱️ **预计时间**

- **单个实验**: ~1.5小时
- **完整消融**: ~4.5小时
- **测试模型**: <1分钟

---

## 🎯 **建议**

1. **先运行完整消融**：`bash run_resnet_ablation.sh`
2. **等待结果**：约4.5小时
3. **分析结果**：对比3个配置的SRCC
4. **更新论文**：根据结果选择合适的呈现方式

---

## 📧 **问题排查**

### **CUDA Out of Memory**：
```bash
# 减小batch size
python3 train_resnet_improved.py --batch_size 16 ...
```

### **DataLoader错误**：
```bash
# 确保data_loader.py已更新
# 确保koniq-10k数据集路径正确
```

### **模型测试失败**：
```bash
# 运行测试脚本
python3 models_resnet_improved.py
```

---

## ✅ **检查清单**

- [x] 模型代码完成（`models_resnet_improved.py`）
- [x] 训练脚本完成（`train_resnet_improved.py`）
- [x] 批处理脚本完成（`run_resnet_ablation.sh`）
- [x] 模型测试通过
- [ ] 运行实验
- [ ] 提取结果
- [ ] 更新论文

---

**创建日期**: 2024-12-24  
**状态**: ✅ 代码完成，等待运行实验

