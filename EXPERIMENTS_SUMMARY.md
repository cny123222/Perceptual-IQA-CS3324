# 实验总结 - 所有14个实验

**更新时间**: 2025-12-22  
**新Baseline**: SRCC 0.9350 (无ColorJitter)  
**单个实验时间**: ~1.7小时 (~20min/epoch × 5 epochs)

---

## 🎯 总览

| 实验组 | 数量 | 总时间 | 说明 |
|--------|------|--------|------|
| **A组** (核心消融) | 3 | ~5h | 移除Attention, Ranking, Multi-scale |
| **C组** (Ranking灵敏度) | 3 | ~5h | Alpha=0.1, 0.5, 0.7 |
| **B组** (模型大小) | 2 | ~3.3h | Tiny, Small |
| **D组** (正则化) | 3 | ~5h | WD=5e-5, 1e-4, 4e-4 |
| **E组** (学习率) | 3 | ~5h | LR=2.5e-6, 7.5e-6, 1e-5 |
| **合计** | **14** | **~23.3h** | 单机顺序运行 |

**双机并行**: ~12小时  
**四机并行**: ~6小时

---

## 📋 14个实验清单

### ✅ 已完成
- [x] **Best (Baseline)** - SRCC: 0.9350, PLCC: 0.9460 (~1.7h)
- [x] **A4 (Remove ColorJitter)** - SRCC: 0.9350 (~1.7h, 证明ColorJitter不重要)

### ⏳ 待运行 (所有都加 `--no_color_jitter`)

#### A组 - 核心消融 (优先级最高)
- [ ] **A1** - 移除Attention Fusion (~1.7h)
- [ ] **A2** - 移除Ranking Loss (~1.7h)
- [ ] **A3** - 移除Multi-scale (~1.7h)

#### C组 - Ranking Loss灵敏度
- [ ] **C1** - Alpha=0.1 (~1.7h)
- [ ] **C2** - Alpha=0.5 (~1.7h)
- [ ] **C3** - Alpha=0.7 (~1.7h)

#### B组 - 模型大小对比
- [ ] **B1** - Swin-Tiny (~1.5h, 更快)
- [ ] **B2** - Swin-Small (~1.8h, 稍慢)

#### D组 - Weight Decay灵敏度
- [ ] **D1** - WD=5e-5 (~1.7h)
- [ ] **D2** - WD=1e-4 (~1.7h)
- [ ] **D4** - WD=4e-4 (~1.7h)

#### E组 - Learning Rate灵敏度
- [ ] **E1** - LR=2.5e-6 (~1.7h)
- [ ] **E3** - LR=7.5e-6 (~1.7h)
- [ ] **E4** - LR=1e-5 (~1.7h)

---

## 🚀 快速开始

### 方式1: 顺序运行（最简单）

```bash
# 进入项目目录
cd /root/Perceptual-IQA-CS3324

# 在tmux中运行
tmux new -s experiments

# 依次复制粘贴 ALL_EXPERIMENTS_COMMANDS.md 中的命令
# 每个实验运行完后再启动下一个
```

### 方式2: 双GPU并行（推荐）

```bash
# Terminal 1 (GPU 0)
tmux new -s exp_gpu0
# 运行 A1, A3, C2, B2, D2, E1, E3

# Terminal 2 (GPU 1)
tmux new -s exp_gpu1
# 运行 A2, C1, C3, B1, D1, D4, E4
```

### 方式3: 多机器并行（最快）

参考 `MULTI_MACHINE_SETUP.md` 配置第二台机器。

---

## 📊 关键参数

**所有实验统一使用**:
```bash
--batch_size 32
--epochs 5
--train_test_num 1
--patience 5
--train_patch_num 20
--test_patch_num 20
--lr_scheduler cosine
--test_random_crop
--no_spaq
--no_color_jitter  # 🔥 所有实验都加这个！
```

**每个实验只改变一个目标参数！**

---

## 🔍 监控命令

```bash
# 查看GPU使用
watch -n 1 nvidia-smi

# 查看最新日志
tail -f logs/swin_*.log

# 查看正在运行的实验
ps aux | grep train_swin.py

# 提取最新结果
grep "best model" logs/swin_*.log | tail -5
```

---

## 📝 完成后更新

每完成一个实验，更新 `EXPERIMENTS_LOG_TRACKER.md`:

```markdown
### [实验名称]
**Log File**: logs/swin_...log
**Results**:
- SRCC: 0.XXXX
- PLCC: 0.XXXX
- Status: ✅ COMPLETE
```

---

## ⚡ 为什么移除ColorJitter？

**A4消融实验结果**:
- 有ColorJitter: SRCC 0.9352, Time ~3.2h
- 无ColorJitter: SRCC 0.9350, Time ~1.7h
- **性能损失**: 仅-0.0002 (0.02%)
- **速度提升**: 1.9倍

**结论**: ColorJitter的微小性能提升不值得2倍的时间成本。

---

## 📚 相关文档

- **`ALL_EXPERIMENTS_COMMANDS.md`** - 所有14个实验的详细命令
- **`EXPERIMENTS_LOG_TRACKER.md`** - 实验结果跟踪表
- **`COLORJITTER_ANALYSIS.md`** - ColorJitter消融分析
- **`MULTI_MACHINE_SETUP.md`** - 多机器并行设置
- **`FINAL_ABLATION_PLAN.md`** - 消融实验设计

---

## 🎯 预期结果

根据这14个实验，我们将能够回答：

1. **核心组件贡献** (A组):
   - Attention Fusion对性能的提升
   - Ranking Loss的重要性
   - Multi-scale的作用

2. **超参数敏感度** (C/D/E组):
   - Ranking Loss Alpha的最优值
   - Weight Decay的最优范围
   - Learning Rate的最优设置

3. **模型大小影响** (B组):
   - Tiny vs Base vs Small的性能-效率权衡

---

**准备好了就开始吧！** 🚀

所有命令都在 `ALL_EXPERIMENTS_COMMANDS.md` 中，复制粘贴即可运行！

