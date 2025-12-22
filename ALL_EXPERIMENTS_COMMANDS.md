# All 14 Experiment Commands (Individual)

**Date**: 2025-12-22  
**Configuration**: batch_size=32, epochs=5, train_test_num=1  
**Baseline**: Alpha=0.3 (SRCC 0.9352)

**建议运行方式**：
- **一次跑1个实验**：最快，无资源竞争
- **一次跑2个实验**：可接受，注意分配不同GPU
- ❌ 不建议4个同时跑：会很慢！

---

## A. Core Ablations (核心消融)

### A1: 移除Attention Fusion

**目的**: 验证Attention Fusion的贡献

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --ranking_loss_alpha 0.3 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

### A2: 移除Ranking Loss (Alpha=0)

**目的**: 验证Ranking Loss的贡献

```bash
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
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

### A3: 移除Multi-scale Feature Fusion

**目的**: 验证Multi-scale的贡献

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 5 \
  --patience 5 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 1 \
  --no_multiscale \
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

---

## C. Ranking Loss Sensitivity (Ranking Loss灵敏度)

### C1: Alpha=0.1 (Lower)

**目的**: 测试较低的ranking loss权重

```bash
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
  --ranking_loss_alpha 0.1 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

### C2: Alpha=0.5 (Higher)

**目的**: 测试较高的ranking loss权重（原best配置）

```bash
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
  --ranking_loss_alpha 0.5 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

### C3: Alpha=0.7 (Much Higher)

**目的**: 测试更高的ranking loss权重

```bash
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
  --ranking_loss_alpha 0.7 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

## B. Model Size Comparison (模型大小对比)

### B1: Swin-Tiny (~28M params)

**目的**: 测试更小的模型

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size tiny \
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

---

### B2: Swin-Small (~50M params)

**目的**: 测试中等大小的模型

```bash
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
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

---

## D. Weight Decay Sensitivity (正则化强度)

### D1: Weight Decay=5e-5 (Very Weak, 0.25×)

**目的**: 测试非常弱的正则化

```bash
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
  --weight_decay 5e-5 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

### D2: Weight Decay=1e-4 (Weak, 0.5×)

**目的**: 测试较弱的正则化

```bash
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
  --weight_decay 1e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

### D4: Weight Decay=4e-4 (Strong, 2×)

**目的**: 测试较强的正则化

```bash
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
  --weight_decay 4e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

## E. Learning Rate Sensitivity (学习率灵敏度)

### E1: LR=2.5e-6 (Conservative, 0.5×)

**目的**: 测试更保守的学习率

```bash
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
  --lr 2.5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

### E3: LR=7.5e-6 (Faster, 1.5×)

**目的**: 测试更快的学习率

```bash
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
  --lr 7.5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

### E4: LR=1e-5 (Aggressive, 2×)

**目的**: 测试激进的学习率

```bash
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
  --lr 1e-5 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```

---

## 📋 实验运行建议

### 方式1：顺序运行（最稳定）

**一次跑1个，完成后再跑下一个**

```bash
# 先跑A1，等它完成
# 然后跑A2，等它完成
# ...
```

**优点**: 速度最快，无GPU竞争  
**缺点**: 需要手动监控  
**预计时间**: 每个5-10分钟，总共1.5-2小时

---

### 方式2：双GPU并行（推荐）

**一次跑2个，用不同的GPU**

```bash
# Terminal 1 - GPU 0
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py ...

# Terminal 2 - GPU 1
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=1 python train_swin.py ...
```

**优点**: 速度较快，可接受的资源竞争  
**缺点**: 需要开两个terminal  
**预计时间**: 每对8-12分钟，总共1-1.5小时

---

### 方式3：tmux后台运行

**用tmux运行，可以关闭SSH**

```bash
# 创建tmux会话
tmux new-session -s exp1

# 运行实验
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py ...

# 按 Ctrl+B 然后 D 退出tmux（实验继续运行）

# 重新连接
tmux attach-session -s exp1
```

---

## 📊 实验顺序建议

### 优先级1：核心消融（必须）
1. **A1** - 移除Attention （最重要）
2. **A2** - 移除Ranking （最重要）
3. **A3** - 移除Multi-scale （最重要）

### 优先级2：Ranking灵敏度
4. **C1** - Alpha=0.1
5. **C2** - Alpha=0.5
6. **C3** - Alpha=0.7

### 优先级3：模型大小
7. **B1** - Swin-Tiny
8. **B2** - Swin-Small

### 优先级4：正则化灵敏度
9. **D1** - WD=5e-5
10. **D2** - WD=1e-4
11. **D4** - WD=4e-4

### 优先级5：学习率灵敏度
12. **E1** - LR=2.5e-6
13. **E3** - LR=7.5e-6
14. **E4** - LR=1e-5

---

## ✅ 参数验证

所有14个命令都包含：
- ✅ `--batch_size 32`
- ✅ `--epochs 5`
- ✅ `--train_test_num 1`
- ✅ `--patience 5`
- ✅ `--train_patch_num 20`
- ✅ `--test_patch_num 20`
- ✅ `--lr_scheduler cosine`
- ✅ `--test_random_crop`
- ✅ `--no_spaq`

每个实验只改变一个目标参数！

---

## 🔍 监控命令

### 查看GPU使用情况
```bash
watch -n 1 nvidia-smi
```

### 查看最新日志
```bash
tail -f logs/swin_*.log
```

### 查看所有正在运行的实验
```bash
ps aux | grep train_swin.py
```

---

## 📝 完成后

记得将所有结果记录到 `VALIDATION_AND_ABLATION_LOG.md`！

每个实验记录：
- 实验名称
- SRCC
- PLCC
- RMSE
- 变化的参数

