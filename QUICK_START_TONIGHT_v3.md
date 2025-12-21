# 🚀 Quick Start Guide - 4-GPU Overnight Experiments (v3)

**Date**: 2025-12-22  
**Total Experiments**: 14  
**Estimated Time**: ~6 hours  
**GPUs**: 4 (parallel execution)

---

## ⚡ Ultra Quick Start (Copy-Paste)

### 🔒 SSH-Safe方式（推荐！即使SSH断开也继续运行）

```bash
cd /root/Perceptual-IQA-CS3324
./start_overnight_experiments.sh
```

**就这么简单！** 脚本会在tmux中运行，**SSH断开后实验继续进行**！

按 `Ctrl+B` 然后按 `D` 可以退出tmux但保持实验运行。

---

### ⚠️ 备选方式（不推荐，SSH断开会停止）

如果你打算一直保持SSH连接：

```bash
cd /root/Perceptual-IQA-CS3324
./run_experiments_4gpus.sh
```

**警告**: SSH断开后脚本会停止！建议使用上面的SSH-safe方式。

---

## 📊 Experiment Overview

### What's Being Tested:

| Category | Experiments | Purpose |
|----------|-------------|---------|
| **A. Core Ablations** | 3 | 测试每个新组件的贡献 (Attention, Ranking Loss, Multi-scale) |
| **C. Ranking Loss** | 3 | 测试不同的ranking loss权重 (0.1, 0.5, 0.7 vs baseline 0.3) |
| **B. Model Size** | 2 | 比较不同大小的Swin模型 (Tiny, Small vs Base) |
| **D. Regularization** | 3 | Weight Decay灵敏度 (5e-5, 1e-4, 4e-4 vs baseline 2e-4) |
| **E. Learning Rate** | 3 | 学习率灵敏度 (2.5e-6, 7.5e-6, 1e-5 vs baseline 5e-6) |

### Execution Plan (4 GPUs):

```
Batch 1 (1.5h): A1, A2, A3, C1    ← Core Ablations + Ranking start
Batch 2 (1.5h): C2, C3, B1, B2    ← Ranking + Model Size
Batch 3 (1.5h): D1, D2, D4, E1    ← Regularization + LR start
Batch 4 (1.5h): E3, E4            ← Learning Rate finish
```

**Total**: ~6 hours (晚上11点 → 早上5点) ⏰

---

## 🔐 SSH断开问题（重要！）

### ⚠️ 问题说明：

如果直接运行 `./run_experiments_4gpus.sh`：
- SSH断开 → 脚本主进程停止
- 已启动的实验会继续运行（在tmux中）
- 但后续batch不会启动！❌

### ✅ 解决方案：

**方法1: 使用 `start_overnight_experiments.sh`（最简单）**

```bash
./start_overnight_experiments.sh
```

这个脚本会在tmux中运行主脚本，SSH断开后完全没问题！

**方法2: 手动使用tmux**

```bash
tmux new-session -s exp-runner "./run_experiments_4gpus.sh"
# 按 Ctrl+B 然后 D 退出
```

**方法3: 使用nohup**

```bash
nohup ./run_experiments_4gpus.sh > experiment_runner.log 2>&1 &
tail -f experiment_runner.log
```

---

## 🔍 Monitoring Progress

### Check Running Experiments:

```bash
# List all tmux sessions
tmux ls

# Attach to a specific experiment (e.g., A1)
tmux attach -t exp-a1

# Detach from tmux session (保持实验继续运行)
# Press: Ctrl+B, then D
```

### Monitor GPU Usage:

```bash
watch -n 1 nvidia-smi
```

### Check Log Files:

```bash
# List recent logs
ls -lth logs/ | head -20

# Tail a specific log
tail -f logs/swin_multiscale_ranking_alpha0.3_TIMESTAMP.log
```

---

## 📋 Experiment Details

### Baseline (Alpha=0.3) ✅ Already Completed:

```
Checkpoint: koniq-10k-swin_20251221_203438/best_model_srcc_0.9352_plcc_0.9471.pkl
SRCC: 0.9352
PLCC: 0.9471
RMSE: 0.1846

Configuration:
- Model: Swin-Base
- Attention: Yes
- Multi-scale: Yes (stages 1,2,3)
- Ranking Alpha: 0.3
- Weight Decay: 2e-4
- Drop Path: 0.3
- Dropout: 0.4
- LR: 5e-6
- Patience: 5
```

### Stage 1: Core Ablations (A1, A2, A3)

**目的**: 量化每个新组件的贡献

- **A1**: 去掉Attention → 预期SRCC < 0.9352
- **A2**: 去掉Ranking Loss → 预期SRCC < 0.9352
- **A3**: 去掉Multi-scale → 预期SRCC < 0.9352

### Stage 2: Ranking Loss Sensitivity (C1, C2, C3)

**目的**: 找到最优的ranking loss权重

- **C1**: Alpha=0.1 (更弱)
- **C2**: Alpha=0.5 (更强)
- **C3**: Alpha=0.7 (很强)

### Stage 3: Model Size (B1, B2)

**目的**: 比较不同大小模型的性能

- **B1**: Swin-Tiny (~28M params)
- **B2**: Swin-Small (~50M params)

### Stage 4: Regularization Sensitivity (D1, D2, D4)

**目的**: Weight Decay参数灵敏度分析

- **D1**: 5e-5 (0.25× baseline, 很弱)
- **D2**: 1e-4 (0.5× baseline, 弱)
- **D4**: 4e-4 (2× baseline, 强)

范围: 0.25× to 2× baseline (跨度8倍)

### Stage 5: Learning Rate Sensitivity (E1, E3, E4)

**目的**: 学习率灵敏度分析

- **E1**: 2.5e-6 (0.5× baseline, 保守)
- **E3**: 7.5e-6 (1.5× baseline)
- **E4**: 1e-5 (2× baseline, 激进)

范围: 0.5× to 2× baseline (跨度4倍)

---

## 🎯 Expected Results

### Core Ablations (A):
- 预期所有ablation实验 **SRCC < 0.9352** (每个组件都应该有贡献)
- 可以量化每个组件的具体贡献值

### Ranking Loss (C):
- Alpha=0.1: 可能太弱
- Alpha=0.3: **当前最优** ✅
- Alpha=0.5: 可能过强
- Alpha=0.7: 可能严重过强

### Model Size (B):
- Tiny < Small < Base (预期)
- 但Tiny和Small可能更快，适合部署

### Regularization (D):
- 5e-5: 可能过拟合
- 1e-4: 可能还不错
- 2e-4: **当前最优** ✅
- 4e-4: 可能欠拟合

### Learning Rate (E):
- 2.5e-6: 可能收敛慢
- 5e-6: **当前最优** ✅
- 7.5e-6: 可能不错
- 1e-5: 可能不稳定

---

## 🛑 Emergency Controls

### Stop All Experiments:

```bash
# Kill all tmux sessions
tmux kill-server
```

### Stop Specific Experiment:

```bash
# Example: stop experiment A1
tmux kill-session -t exp-a1
```

### Resume After Interruption:

如果脚本中断，可以手动启动剩余实验。每个实验的完整命令都在 `EXPERIMENTS_TO_RUN_v3.md` 中。

---

## 📝 After Completion

### 1. Check Results:

```bash
# Find best checkpoints
find checkpoints/ -name "best_model_*" | sort

# Check recent logs
ls -lth logs/ | head -20
```

### 2. Extract Best Metrics:

每个实验的日志文件包含:
- Round 1最佳SRCC/PLCC
- 训练曲线
- 早停信息

### 3. Update Documentation:

将结果填入:
- `VALIDATION_AND_ABLATION_LOG.md`
- `EXPERIMENTS_TO_RUN_v3.md` (Results Summary部分)

### 4. Generate Plots:

使用结果生成以下图表:
- Ranking Loss Sensitivity Curve (C1-C3)
- Weight Decay Sensitivity Curve (D1-D4)
- Learning Rate Sensitivity Curve (E1, E2, E3, E4)
- Component Contribution Bar Chart (A1-A3)
- Model Size Comparison (B1-B2)

---

## 🎨 Results Template

复制到 `VALIDATION_AND_ABLATION_LOG.md`:

```markdown
## Ablation & Sensitivity Analysis Results (2025-12-22)

### Core Ablations:
| Exp | Config | SRCC | PLCC | RMSE | Δ SRCC | Component Impact |
|-----|--------|------|------|------|--------|------------------|
| Baseline | Full Model | 0.9352 | 0.9471 | 0.1846 | - | - |
| A1 | No Attention | ? | ? | ? | ? | ? |
| A2 | No Ranking | ? | ? | ? | ? | ? |
| A3 | No Multi-scale | ? | ? | ? | ? | ? |

### Ranking Loss Sensitivity:
| Exp | Alpha | SRCC | PLCC | RMSE | Δ SRCC |
|-----|-------|------|------|------|--------|
| C1 | 0.1 | ? | ? | ? | ? |
| Baseline | 0.3 | 0.9352 | 0.9471 | 0.1846 | - |
| C2 | 0.5 | ? | ? | ? | ? |
| C3 | 0.7 | ? | ? | ? | ? |

### Model Size:
| Exp | Size | Params | SRCC | PLCC | RMSE |
|-----|------|--------|------|------|------|
| B1 | Tiny | ~28M | ? | ? | ? |
| B2 | Small | ~50M | ? | ? | ? |
| Baseline | Base | ~88M | 0.9352 | 0.9471 | 0.1846 |

### Weight Decay Sensitivity:
| Exp | WD | SRCC | PLCC | RMSE | Δ SRCC |
|-----|-----|------|------|------|--------|
| D1 | 5e-5 | ? | ? | ? | ? |
| D2 | 1e-4 | ? | ? | ? | ? |
| Baseline | 2e-4 | 0.9352 | 0.9471 | 0.1846 | - |
| D4 | 4e-4 | ? | ? | ? | ? |

### Learning Rate Sensitivity:
| Exp | LR | SRCC | PLCC | RMSE | Δ SRCC |
|-----|-----|------|------|------|--------|
| E1 | 2.5e-6 | ? | ? | ? | ? |
| Baseline | 5e-6 | 0.9352 | 0.9471 | 0.1846 | - |
| E3 | 7.5e-6 | ? | ? | ? | ? |
| E4 | 1e-5 | ? | ? | ? | ? |
```

---

## 💡 Pro Tips

1. **监控第一个batch**: 启动后等待几分钟，确认4个实验都正常运行
2. **日志检查**: 如果某个实验失败，日志文件会包含错误信息
3. **GPU内存**: 每个实验约占用10-12GB，4个GPU应该足够
4. **早停机制**: patience=5，如果5个epoch没提升就会自动停止
5. **断点恢复**: 如果需要中断，可以手动从`EXPERIMENTS_TO_RUN_v3.md`复制命令继续

---

## 🎉 Ready to Go!

一切准备就绪！只需运行：

```bash
./run_experiments_4gpus.sh
```

**晚安！明早见结果！** 🌙✨

---

## 📞 Troubleshooting

### Problem: "tmux: command not found"
```bash
apt-get update && apt-get install -y tmux
```

### Problem: "CUDA out of memory"
- 减少batch_size (从4改为2)
- 或者一次只运行2个实验

### Problem: 脚本权限错误
```bash
chmod +x run_experiments_4gpus.sh
```

### Problem: 某个实验卡住不动
```bash
# 检查GPU使用情况
nvidia-smi

# 如果某个GPU空闲，可能实验已经完成或失败
# 检查对应的tmux session
tmux attach -t exp-a1  # 替换为对应的session名
```

