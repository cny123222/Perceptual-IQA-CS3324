# 🔬 消融实验计划 - 使用最佳LR=5e-7

## 📋 实验配置

**基础设置** (所有实验统一):
- **Learning Rate**: **5e-7** 🏆 (已验证的最佳学习率)
- **Epochs**: 10
- **Patience**: 3
- **train_test_num**: 1 (单轮)
- **Batch Size**: 32
- **Weight Decay**: 2e-4
- **LR Scheduler**: cosine
- **Test Random Crop**: True
- **ColorJitter**: False (已禁用)
- **Ranking Loss Alpha**: 0 (无ranking loss)

---

## 🎯 实验列表

### **Baseline: 完整模型 (LR=5e-7)**

**目的**: 重新跑baseline，确保所有实验使用相同的最佳LR

**配置**:
- Model Size: base
- Multi-scale: ✅ True
- Attention Fusion: ✅ True
- Learning Rate: 5e-7

**预期**: SRCC ~0.938 (基于E6结果)

---

### **A1: Remove Attention Fusion**

**目的**: 测试Attention Fusion的贡献

**配置**:
- Model Size: base
- Multi-scale: ✅ True
- Attention Fusion: ❌ **False** (移除)
- Learning Rate: 5e-7

**预期**: SRCC ~0.932-0.935 (预计下降~0.003-0.006)

---

### **A2: Remove Multi-scale Fusion**

**目的**: 测试Multi-scale Fusion的贡献

**配置**:
- Model Size: base
- Multi-scale: ❌ **False** (移除)
- Attention Fusion: ✅ True
- Learning Rate: 5e-7

**预期**: SRCC ~0.930-0.933 (预计下降~0.005-0.008)

---

### **B1: Tiny Model**

**目的**: 测试模型容量的影响

**配置**:
- Model Size: **tiny** (vs base)
- Multi-scale: ✅ True
- Attention Fusion: ✅ True
- Learning Rate: 5e-7
- Drop Path: 0.2 (vs 0.3)
- Dropout: 0.3 (vs 0.4)

**预期**: SRCC ~0.922-0.925 (预计下降~0.013-0.016)

---

### **B2: Small Model**

**目的**: 测试中等模型的性能平衡

**配置**:
- Model Size: **small** (vs base)
- Multi-scale: ✅ True
- Attention Fusion: ✅ True
- Learning Rate: 5e-7
- Drop Path: 0.25 (vs 0.3)
- Dropout: 0.35 (vs 0.4)

**预期**: SRCC ~0.933-0.936 (预计下降~0.002-0.005)

---

## 📊 实验批次安排

### **Batch 1**: Baseline + A1
- **GPU 0**: Baseline (LR=5e-7)
- **GPU 1**: A1 (No Attention)
- **时间**: ~20分钟

### **Batch 2**: A2 + B1
- **GPU 0**: A2 (No Multi-scale)
- **GPU 1**: B1 (Tiny)
- **时间**: ~20分钟

### **Batch 3**: B2
- **GPU 0**: B2 (Small)
- **时间**: ~20分钟

**总时间**: ~1小时

---

## 🚀 启动方法

### **方法1: 使用自动化脚本 (推荐)**

```bash
cd /root/Perceptual-IQA-CS3324
./run_ablations_lr5e7.sh
```

### **方法2: 使用nohup后台运行**

```bash
nohup ./run_ablations_lr5e7.sh > ablations_5e7.out 2>&1 &
```

### **方法3: 手动启动单个实验**

```bash
# Baseline
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 \
  --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 \
  --train_test_num 1 --lr 5e-7 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --attention_fusion --ranking_loss_alpha 0 --test_random_crop \
  --no_spaq --no_color_jitter 2>&1 | tee logs/baseline_lr5e7.log

# A1 (No Attention)
CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 \
  --epochs 10 --patience 3 --train_patch_num 20 --test_patch_num 20 \
  --train_test_num 1 --lr 5e-7 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --ranking_loss_alpha 0 --test_random_crop \
  --no_spaq --no_color_jitter 2>&1 | tee logs/A1_no_attention_lr5e7.log
```

---

## 🔍 监控方法

### **1. 查看tmux窗口**
```bash
tmux attach -t iqa_ablations
# Ctrl+B 然后 1/2 切换窗口
# Ctrl+B D 分离
```

### **2. 查看日志**
```bash
# 实时监控
tail -f logs/baseline_lr5e7_*.log

# 查看所有日志
ls -lht logs/*_lr5e7_*.log
```

### **3. 查看GPU状态**
```bash
watch -n 10 nvidia-smi
```

### **4. 查看进程**
```bash
ps aux | grep train_swin.py | grep -v grep
```

---

## 📈 预期结果

基于之前的实验结果，使用LR=5e-7后预期：

| 实验 | 配置 | 预期SRCC | vs Baseline | 说明 |
|------|------|----------|-------------|------|
| **Baseline** | Full model, LR=5e-7 | **0.9380** | - | 新基准 |
| **A1** | No Attention | 0.9320-0.9350 | -0.003 ~ -0.006 | Attention贡献 |
| **A2** | No Multi-scale | 0.9300-0.9330 | -0.005 ~ -0.008 | Multi-scale贡献 |
| **B1** | Tiny | 0.9220-0.9250 | -0.013 ~ -0.016 | 容量影响 |
| **B2** | Small | 0.9330-0.9360 | -0.002 ~ -0.005 | 平衡点 |

**注**: 这些预期基于之前用5e-6的结果，使用5e-7后所有结果可能整体提升0.002-0.004

---

## 📝 结果提取

实验完成后：

```bash
# 快速查看所有结果
grep "Best test SRCC" logs/*_lr5e7_*.log

# 详细提取
for log in logs/*_lr5e7_*.log; do
    echo "=== $(basename $log) ==="
    grep "Best test SRCC" "$log"
    echo ""
done
```

---

## ✅ 完成后

1. ✅ 记录所有结果到 `EXPERIMENTS_LOG_TRACKER.md`
2. ✅ 更新贡献分析
3. ✅ 对比新旧LR的结果差异
4. ✅ 准备论文图表数据

---

## 🎯 关键问题

### **为什么重新跑Baseline？**
- 使用统一的LR=5e-7确保公平比较
- 之前的E6虽然达到0.9378，但训练中断了
- 确保baseline的稳定性和可重复性

### **为什么用10 epochs + patience 3？**
- E6显示5e-7需要8个epoch才达到最佳
- 10个epoch给足够的收敛空间
- patience=3在保证充分训练的同时避免过拟合

### **预计改进幅度？**
- 所有实验从5e-6换到5e-7，预期整体提升0.002-0.004 SRCC
- Baseline: 0.9354 → 0.9380 (+0.0026)
- 其他实验也会相应提升

---

**准备好开始了吗？** 🚀

执行: `./run_ablations_lr5e7.sh` 或 `nohup ./run_ablations_lr5e7.sh > ablations_5e7.out 2>&1 &`

