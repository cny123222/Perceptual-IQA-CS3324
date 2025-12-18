# 快速总结 - 给 AI 看

## 🎯 核心问题

**第 1 个 epoch 达到最佳性能，之后持续过拟合**

```
Epoch 1: Test SRCC 0.9195 ⭐ 最佳
Epoch 2: Test SRCC 0.9185 ↓
Epoch 3: Test SRCC 0.9145 ↓↓
Epoch 4: Test SRCC 0.9174 ↓
```

训练集性能持续上升：0.8758 → 0.9747

---

## 🏗️ 架构

**Swin Transformer Tiny (28.8M) + HyperNet (0.5M)**
- Multi-scale feature fusion (4 stages: 96, 192, 384, 768 channels)
- 简单 concat + conv 降维
- 数据：7,046 训练图像，每个采样 20 patches (224×224)

---

## ❌ 失败的尝试

1. **Ranking Loss**: SRCC 0.9195 → 0.9092 ❌
2. **Attention Fusion**: PLCC 0.9342 → 0.9317，过拟合更严重 ❌
3. **Attention + Strong Dropout (0.5)**: 无改善 ❌

---

## 💭 可能的改进方向（未尝试）

### 优先级 1：正则化
- [ ] Weight decay (1e-4)
- [ ] Dropout in HyperNet (0.3)
- [ ] Gradient clipping
- [ ] Smaller learning rate (1e-4 / 1e-5)

### 优先级 2：数据增强
- [ ] ColorJitter (小心不要破坏图像质量)
- [ ] RandomHorizontalFlip
- [ ] Reduce patch_num (20 → 10, 降低冗余)

### 优先级 3：模型容量
- [ ] 更小的 backbone (Swin-Nano / ResNet50)
- [ ] Freeze early layers
- [ ] Remove HyperNet（太复杂？）

---

## ❓ 关键问题

1. **为什么 1 个 epoch 就学到位了？**
   - 学习率太大？模型太强？数据太简单？

2. **HyperNet + Swin Transformer 合理吗？**
   - 两个都是复杂架构，是否冲突？

3. **如何系统性地测试假设？**
   - 应该先尝试哪个方向？

4. **IQA 任务的数据增强最佳实践？**
   - 哪些增强不会改变质量分数？

---

## 📊 当前最佳性能

- **SRCC**: 0.9195
- **PLCC**: 0.9342
- **时机**: 第 1 个 epoch
- **模型**: Multi-scale concat (无 attention)

接近 SOTA，但过拟合是大问题。

---

完整报告见：`EXPERIMENT_SUMMARY.md`

