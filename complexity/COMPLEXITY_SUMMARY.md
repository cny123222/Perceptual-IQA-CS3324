# 计算复杂度分析汇总

## 📊 所有模型对比

| 模型 | 参数量 (M) | FLOPs (G) | 推理时间 (ms) | 吞吐量 (images/sec) |
|------|-----------|-----------|--------------|---------------------|
| HyperIQA (ResNet50) | 27.38 | 4.33 | 3.12 | 329.73 |
| SMART-Tiny (Swin-T) | 29.52 | 4.47 | - | - |
| SMART-Small (Swin-S) | 50.84 | 8.65 | - | - |
| SMART-Base (Swin-B) | 89.11 | 15.28 | 10.06 | 97.37 |

## 🔍 关键观察

### 1. 参数量分析
- **HyperIQA (ResNet50)**: 27.38M - 最轻量
- **SMART-Tiny**: 29.52M - 与ResNet50相近 (+7.8%)
- **SMART-Small**: 50.84M - 中等规模 (+85.6% vs ResNet50)
- **SMART-Base**: 89.11M - 最大模型 (+225.5% vs ResNet50)

### 2. 计算复杂度分析
- **HyperIQA (ResNet50)**: 4.33G FLOPs - 最低计算量
- **SMART-Tiny**: 4.47G FLOPs - 与ResNet50相近 (+3.2%)
- **SMART-Small**: 8.65G FLOPs - 约2倍于ResNet50
- **SMART-Base**: 15.28G FLOPs - 约3.5倍于ResNet50

### 3. 推理速度分析（实测）
- **HyperIQA (ResNet50)**: 
  - 推理时间: 3.12ms
  - 吞吐量: 329.73 images/sec
  - **最快的模型**
  
- **SMART-Base**: 
  - 推理时间: 10.06ms (约3.2倍于ResNet50)
  - 吞吐量: 97.37 images/sec
  - 虽然较慢，但准确度显著提升（SRCC: 0.9378 vs ~0.89）

### 4. 准确度-效率权衡

从已有的实验结果来看：

| 模型 | SRCC | PLCC | 参数量 (M) | FLOPs (G) | 推理时间 (ms) |
|------|------|------|-----------|-----------|--------------|
| HyperIQA (ResNet50) | ~0.890 | ~0.910 | 27.38 | 4.33 | 3.12 |
| SMART-Base | 0.9378 | 0.9485 | 89.11 | 15.28 | 10.06 |

**准确度提升**: +5.4% SRCC, +4.2% PLCC  
**计算成本**: +3.5× FLOPs, +3.2× 推理时间

## 💡 结论

1. **SMART-Tiny** 与 HyperIQA (ResNet50) 复杂度相近，可作为直接替代方案
2. **SMART-Base** 虽然计算量较大，但仍在实时推理范围内（10ms < 100fps）
3. Swin Transformer backbone 相比ResNet50：
   - 以适度的计算开销（3-4倍）
   - 换取显著的准确度提升（5%+ SRCC）
   - 推理速度仍然实用（97 images/sec on RTX 5090）

## 📝 测试环境

- **GPU**: NVIDIA GeForce RTX 5090
- **输入尺寸**: 224×224×3
- **Batch Size**: 1 (用于推理时间测量)
- **精度**: FP32
- **Warmup**: 10次迭代
- **测量次数**: 100次迭代（取平均）

## 📚 详细报告

- [HyperIQA (ResNet50) 详细报告](complexity_results_resnet50.md)
- [SMART-Tiny 详细报告](complexity_results_swin_tiny.md)
- [SMART-Small 详细报告](complexity_results_swin_small.md)
- [SMART-Base 详细报告](complexity_results_swin_base.md)
