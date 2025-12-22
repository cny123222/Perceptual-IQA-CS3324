# A4 实验快速启动指南

## 🚀 一键运行 A4 (移除ColorJitter)

现在已经添加了 `--no_color_jitter` 参数，不需要手动修改代码！

---

## ⚡ 快速命令

```bash
# 在tmux中运行
tmux new -s a4_experiment

# 运行A4实验（移除ColorJitter）
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
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
  --no_spaq \
  --no_color_jitter
```

**关键变化**: 添加了 `--no_color_jitter` 参数！

---

## ⏱️ 预期时间

- **有ColorJitter**: ~2小时
- **无ColorJitter (A4)**: **~40分钟** ⚡

---

## 📊 查看结果

### 40分钟后提取结果

```bash
# 找到最新的日志文件
ls -lth logs/swin_*.log | head -1

# 提取最佳SRCC
grep "best model" logs/swin_multiscale_ranking_alpha0.3_*.log | tail -1
```

### 预期输出

```
⭐ New best model saved! SRCC: 0.XXXX, PLCC: 0.XXXX
```

---

## 🎯 决策标准

对比 Baseline (有ColorJitter):
- **Baseline SRCC**: 0.9352
- **A4 SRCC**: ??? (待测试)

### 场景1: SRCC下降 < 0.002
```
✅ 移除ColorJitter！
例如: 0.9352 → 0.9332+ (下降 < 0.002)

行动:
1. 性能损失可忽略
2. 节省67%训练时间
3. 所有后续实验都快3倍
4. 更新 data_loader.py 永久移除ColorJitter
```

### 场景2: SRCC下降 > 0.005
```
❌ 保留ColorJitter
例如: 0.9352 → 0.9302- (下降 > 0.005)

行动:
1. 性能损失显著
2. 接受2小时/实验的时间成本
3. 在论文中强调数据增强的重要性
```

### 场景3: 0.002 < SRCC下降 < 0.005
```
🤔 权衡取舍
例如: 0.9352 → 0.9320 (下降 0.002-0.005)

行动:
1. 性能损失适中
2. 根据deadline和计算资源决定
3. 可以在论文中讨论这个trade-off
```

---

## 📝 更新实验跟踪

实验完成后，更新 `EXPERIMENTS_LOG_TRACKER.md`:

```bash
cd /root/Perceptual-IQA-CS3324
nano EXPERIMENTS_LOG_TRACKER.md

# 填入结果
# A4 - Remove ColorJitter
# Log File: logs/swin_multiscale_ranking_alpha0.3_YYYYMMDD_HHMMSS.log
# SRCC: 0.XXXX
# PLCC: 0.XXXX
# Status: ✅ COMPLETE

git add EXPERIMENTS_LOG_TRACKER.md
git commit -m "docs: Add A4 (ColorJitter ablation) results"
git push origin master
```

---

## 🔄 如果决定永久移除ColorJitter

### 方法1: 默认关闭ColorJitter（推荐）

修改 `HyperIQASolver_swin.py` 第51行:

```python
# 改为默认关闭
self.use_color_jitter = getattr(config, 'use_color_jitter', False)  # Changed default to False
```

### 方法2: 完全删除ColorJitter代码

如果确定不再需要，可以从 `data_loader.py` 中完全删除相关代码。

---

## 🎉 优势

使用 `--no_color_jitter` 参数的优势：

1. ✅ **无需手动编辑代码**
2. ✅ **干净且可逆**
3. ✅ **易于对比实验**
4. ✅ **在配置日志中清晰显示**

查看实验配置输出:
```
Training Strategy:
  LR Scheduler:             cosine
  Multi-Scale Fusion:       True
  Attention Fusion:         True
  ColorJitter Augmentation: False  <-- 清楚显示已关闭
  Test Random Crop:         True
  SPAQ Cross-Dataset Test:  False
```

---

## 📚 相关文档

- `COLORJITTER_ANALYSIS.md` - 详细分析
- `RUN_A4_FIRST.md` - 完整指南
- `EXPERIMENTS_LOG_TRACKER.md` - 结果跟踪

---

## ✅ 快速检查清单

- [ ] 启动tmux会话
- [ ] 运行A4命令（包含 `--no_color_jitter`）
- [ ] 等待40分钟
- [ ] 提取SRCC/PLCC结果
- [ ] 对比baseline (0.9352)
- [ ] 根据结果决定策略
- [ ] 更新 EXPERIMENTS_LOG_TRACKER.md
- [ ] 提交结果到Git

---

**准备好了吗？复制上面的命令，开始运行A4实验！** 🚀

这40分钟的投资可能为你节省18.7小时！

**最后更新**: 2025-12-22

