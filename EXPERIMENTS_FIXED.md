# 🔧 实验问题修复报告

## ❌ **昨晚的问题**

### 1. **磁盘空间满了**
- **原因**: checkpoint目录积累了73个子目录，占用约31GB空间
- **解决**: 清理了67个旧目录，只保留6个最重要的checkpoint
  - 保留: batch1的2个 + batch3的2个 + baseline + ResNet baseline
  - **释放空间: 31GB ✅**

### 2. **train_test_num 设置错误**
- **问题**: 所有脚本都设置了 `--train_test_num 10`，导致每个实验跑10轮
- **应该**: `--train_test_num 1`，只跑1轮
- **影响**: 
  - 每个实验时间从~20分钟变成~3.4小时 (10倍)
  - 总时间从~2小时变成~10小时+
  - 每个实验生成约8.7MB的日志（正常应该~1MB）

### 3. **参数名错误**
- **问题**: A2实验使用了 `--no_multi_scale`（下划线）
- **正确**: `--no_multiscale`（无下划线）
- **影响**: A2实验启动失败

---

## ✅ **修复内容**

### 清理的Checkpoint
```bash
删除: 67 个目录
保留: 6 个目录
  - koniq-10k-swin_20251223_002219     (Batch1 LR=1e-6, 2.4GB)
  - koniq-10k-swin_20251223_002226     (Batch1 LR=5e-7, 2.7GB)
  - koniq-10k-swin_20251223_035309     (Batch3 Small, 67MB)
  - koniq-10k-swin_20251223_035433     (Batch3 Tiny, 388KB)
  - koniq-10k-swin_20251222_161625     (Baseline SRCC 0.9354, 1GB)
  - koniq-10k-resnet_20251221_004809   (ResNet baseline, 629MB)
释放: 31243MB 空间
```

### 新脚本: `run_experiments_fixed.sh`
所有问题已修复：
1. ✅ `train_test_num` 改为 `1`
2. ✅ `--no_multi_scale` 改为 `--no_multiscale`
3. ✅ 改进的进程等待逻辑
4. ✅ 使用tmux避免SSH断开

---

## 📊 **实验配置（修复后）**

| 实验 | GPU | 参数变化 | 预计时间 | 日志文件 |
|------|-----|----------|----------|----------|
| **Batch 1** |
| LR=1e-6 | GPU 0 | train_test_num=1 | ~20分钟 | batch1_gpu0_lr1e6_*.log |
| LR=5e-7 | GPU 1 | train_test_num=1 | ~20分钟 | batch1_gpu1_lr5e7_*.log |
| **Batch 2** |
| A1 (无Attention) | GPU 0 | train_test_num=1 | ~20分钟 | batch2_gpu0_A1_*.log |
| A2 (无Multi-scale) | GPU 1 | train_test_num=1, --no_multiscale | ~20分钟 | batch2_gpu1_A2_*.log |
| **Batch 3** |
| B1 (Tiny) | GPU 0 | train_test_num=1 | ~15分钟 | batch3_gpu0_B1_*.log |
| B2 (Small) | GPU 1 | train_test_num=1 | ~18分钟 | batch3_gpu1_B2_*.log |

**总时间**: ~2小时（修复前是~10小时+）

---

## 🚀 **重新开始实验**

### 1. 检查磁盘空间
```bash
df -h /root
# 应该有充足空间（27G可用）
```

### 2. 清理旧的运行进程（如果有）
```bash
ps aux | grep train_swin.py | grep -v grep
# 如果有进程，使用 kill -9 <PID> 终止
```

### 3. 运行新脚本
```bash
cd /root/Perceptual-IQA-CS3324
nohup ./run_experiments_fixed.sh > experiments_${TIMESTAMP}.out 2>&1 &
```

### 4. 监控进度

#### 方法1: 查看tmux窗口
```bash
tmux attach -t iqa_experiments
# Ctrl+B 然后 1 → GPU 0窗口
# Ctrl+B 然后 2 → GPU 1窗口
# Ctrl+B 然后 D → 分离（实验继续运行）
```

#### 方法2: 查看日志
```bash
# 查看最新日志
ls -lht logs/batch*.log | head -5

# 实时监控
tail -f logs/batch1_gpu0_lr1e6_*.log

# 查看进度
grep -E "Round|Epoch|Best test SRCC" logs/batch1_gpu0_lr1e6_*.log | tail -20
```

#### 方法3: 查看GPU使用
```bash
nvidia-smi
watch -n 10 nvidia-smi
```

#### 方法4: 查看进程
```bash
ps aux | grep train_swin.py | grep -v grep
```

---

## 📝 **提取结果**

所有实验完成后：

```bash
# 快速查看所有结果
grep "Best test SRCC" logs/batch*_20251223*.log

# 详细结果
for log in logs/batch*_20251223*.log; do
    echo "=== $log ==="
    grep "Best test SRCC" "$log" | tail -1
done
```

---

## ⏱️ **预期时间线（修复后）**

| 时间点 | 事件 | 累计时间 |
|--------|------|----------|
| 00:00 | 开始 Batch 1 | 0h |
| 00:20 | Batch 1 完成 → 开始 Batch 2 | 20min |
| 00:40 | Batch 2 完成 → 开始 Batch 3 | 40min |
| 00:58 | Batch 3 完成 → 全部完成 | ~1h |
| **01:00** | **所有6个实验完成** ✅ | **~1小时** |

*(原来错误设置需要~10小时)*

---

## 🎯 **实验目标**

1. **Batch 1**: 确认最优学习率（1e-6 vs 5e-7）
2. **Batch 2**: 验证架构贡献
   - A1: Attention Fusion的贡献
   - A2: Multi-scale Fusion的贡献
3. **Batch 3**: 模型大小对比
   - B1: Tiny vs Base（速度 vs 精度）
   - B2: Small vs Base（平衡点）

---

## ✅ **检查清单**

开始前确认：
- [ ] 磁盘空间充足（`df -h /root` 显示 >20G可用）
- [ ] 没有旧进程运行（`ps aux | grep train_swin.py`）
- [ ] tmux session不存在或已清理（`tmux ls`）
- [ ] 新脚本有执行权限（`ls -l run_experiments_fixed.sh`）

开始后检查：
- [ ] 两个GPU都在运行（`nvidia-smi`）
- [ ] 日志文件在生成（`ls -lht logs/batch*.log`）
- [ ] 进程正常（`ps aux | grep train_swin.py`）

完成后验证：
- [ ] 6个日志文件都生成了
- [ ] 每个日志都有"Best test SRCC"结果
- [ ] checkpoint目录大小合理（不会爆满）

---

## 🔍 **故障排查**

### 如果某个实验卡住
```bash
# 1. 附加到tmux查看实时输出
tmux attach -t iqa_experiments

# 2. 检查该GPU的实际进程
ps aux | grep "CUDA.*train_swin.py" | grep -v grep

# 3. 查看日志最后几行
tail -30 logs/batch*_<问题实验>_*.log
```

### 如果磁盘再次满了
```bash
# 1. 检查checkpoint目录
du -sh checkpoints/*/

# 2. 删除正在运行实验的临时checkpoint（它们会重新生成）
rm -rf checkpoints/koniq-10k-swin_$(date +%Y%m%d)*

# 3. 清理日志
rm logs/swin_multiscale_ranking_alpha0_202512*.log
```

### 如果需要中断重启
```bash
# 1. 停止所有训练进程
pkill -9 -f train_swin.py

# 2. 清理tmux
tmux kill-session -t iqa_experiments

# 3. 重新运行脚本
./run_experiments_fixed.sh
```

---

**准备好开始了吗？** 🚀

