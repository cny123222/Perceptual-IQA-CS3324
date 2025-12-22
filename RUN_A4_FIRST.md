# ⚡ 先运行 A4 实验！

## 🚨 为什么A4最重要？

**ColorJitter导致3倍速度下降！**

- ❌ **有ColorJitter**: 每个实验 ~2小时
- ✅ **无ColorJitter**: 每个实验 ~40分钟

**潜在节省**:
- 14个实验 × 1小时20分钟 = **节省18.7小时** (67%时间)

**必须先验证**: ColorJitter的性能提升是否值得3倍的时间成本？

---

## 🎯 A4实验：移除ColorJitter

### 步骤1: 修改代码

编辑 `data_loader.py`，注释掉第49行：

```bash
cd /root/Perceptual-IQA-CS3324
nano data_loader.py
# 或者
vim data_loader.py
```

找到第49行，注释掉：

```python
# Line 47-52
if istrain:
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize((512, 384)),
        torchvision.transforms.RandomCrop(size=patch_size),
        # ABLATION A4: Comment out ColorJitter
        # torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])
```

**或者使用sed快速注释**:

```bash
cd /root/Perceptual-IQA-CS3324
sed -i '49s/^/# /' data_loader.py
```

### 步骤2: 运行实验

```bash
cd /root/Perceptual-IQA-CS3324

# 在tmux中运行
tmux new -s a4_colorjitter

# 运行A4实验
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
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
```

**预计时间**: ~40分钟

### 步骤3: 提取结果

```bash
# 找到最新的日志文件
ls -lth logs/swin_*.log | head -1

# 提取最佳SRCC
grep "best model" logs/swin_multiscale_ranking_alpha0.3_*.log | tail -1
```

### 步骤4: 恢复代码

```bash
cd /root/Perceptual-IQA-CS3324

# 恢复ColorJitter行
sed -i '49s/^# //' data_loader.py

# 或者手动编辑去掉注释符号
```

---

## 📊 结果分析

### Baseline (有ColorJitter)
- **SRCC**: 0.9352
- **PLCC**: 0.9460
- **Time**: ~2小时

### A4 (无ColorJitter)
- **SRCC**: ??? (待测试)
- **PLCC**: ??? (待测试)
- **Time**: ~40分钟

### 决策标准

**场景1: SRCC下降 < 0.002 (例如: 0.9352 → 0.9332)**
```
✅ 移除ColorJitter！
- 性能损失可忽略 (<0.2%)
- 节省67%训练时间
- 所有后续实验都快3倍
```

**场景2: SRCC下降 > 0.005 (例如: 0.9352 → 0.9302)**
```
❌ 保留ColorJitter
- 性能损失显著 (>0.5%)
- 接受2小时/实验的时间成本
- 在论文中强调数据增强的重要性
```

**场景3: 0.002 < SRCC下降 < 0.005 (例如: 0.9352 → 0.9320)**
```
🤔 权衡取舍
- 性能损失适中 (0.2-0.5%)
- 可以在论文中讨论这个trade-off
- 根据deadline和计算资源决定
```

---

## 🔄 如果决定移除ColorJitter

### 永久移除

```bash
cd /root/Perceptual-IQA-CS3324

# 注释掉ColorJitter
sed -i '49s/^/# /' data_loader.py

# 提交更改
git add data_loader.py
git commit -m "perf: Remove ColorJitter for 3x training speedup

A4 ablation results show ColorJitter contribution is minimal
(SRCC drop < 0.002) while causing 3x training slowdown.

Trade-off analysis:
- Speed gain: 2h → 40min per experiment (67% faster)
- Performance loss: SRCC -0.00XX (negligible)
- Total time saved: 18.7 hours on 14 experiments

Decision: Remove ColorJitter to accelerate research iteration."

git push origin master
```

### 更新所有实验时间估算

所有后续实验时间从2小时降到40分钟！

---

## 📋 快速命令汇总

```bash
# 1. 注释ColorJitter
cd /root/Perceptual-IQA-CS3324
sed -i '49s/^/# /' data_loader.py

# 2. 运行A4实验 (tmux)
tmux new -s a4_colorjitter
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 \
  --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0.3 --lr 5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq

# 3. 查看结果 (40分钟后)
grep "best model" logs/swin_multiscale_ranking_alpha0.3_*.log | tail -1

# 4a. 如果决定保留ColorJitter
sed -i '49s/^# //' data_loader.py

# 4b. 如果决定移除ColorJitter
git add data_loader.py
git commit -m "perf: Remove ColorJitter (A4 ablation shows minimal impact)"
git push origin master
```

---

## 🎯 后续行动

### 如果移除ColorJitter

**机器A** (8个实验 × 40min = 5.3小时):
- A1, A2, A3, C1, C2, C3, B1, B2

**机器B** (6个实验 × 40min = 4小时):
- D1, D2, D4, E1, E3, E4

**总时间**: ~5.3小时 (vs 28小时) 🚀

### 如果保留ColorJitter

**机器A** (8个实验 × 2h = 16小时)
**机器B** (6个实验 × 2h = 12小时)

**总时间**: ~16小时 (并行) ⏰

---

## 💡 额外优化建议

如果ColorJitter很重要但又想加速，可以考虑：

### 方案1: 轻量级ColorJitter
```python
# 只使用brightness，去掉hue (hue最慢)
torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1),
```

### 方案2: GPU加速增强
```python
# 使用Kornia (GPU加速)
import kornia.augmentation as K
K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
```

### 方案3: 减少训练patch数量
```bash
# 从20降到10 (但可能影响性能)
--train_patch_num 10
```

---

## ✅ 检查清单

- [ ] 注释掉 `data_loader.py` 第49行
- [ ] 在tmux中运行A4实验
- [ ] 等待40分钟
- [ ] 提取SRCC/PLCC结果
- [ ] 对比baseline (0.9352)
- [ ] 根据结果决定保留或移除
- [ ] 更新 `EXPERIMENTS_LOG_TRACKER.md`
- [ ] 如果移除，提交代码更改

---

**关键点**: 这40分钟的投资可能为你节省18.7小时！绝对值得先做！

**最后更新**: 2025-12-22

