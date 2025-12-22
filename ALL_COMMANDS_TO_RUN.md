# 🔍 所有实验命令清单 - 请仔细检查

**总实验数**: 6个实验，分3个批次  
**预计总时间**: ~10小时  
**使用GPU**: 2块（每批次并行2个实验）

---

## 📋 Batch 1: Learning Rate Comparison (Phase 1)

### GPU 0: LR = 1e-6
```bash
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**输出**: `logs/batch1_gpu0_lr1e6.log`

---

### GPU 1: LR = 5e-7
```bash
CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 5e-7 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**输出**: `logs/batch1_gpu1_lr5e7.log`

**预计时间**: ~3.4小时

---

## 📋 Batch 2: Ablation Studies (Phase 2)

### GPU 0: A1 - Remove Attention Fusion
```bash
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**注意**: 没有 `--attention_fusion` 参数  
**输出**: `logs/batch2_gpu0_A1_no_attention.log`

---

### GPU 1: A2 - Remove Multi-scale Fusion
```bash
CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --no_multi_scale \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**注意**: 有 `--no_multi_scale` 参数  
**输出**: `logs/batch2_gpu1_A2_no_multiscale.log`

**预计时间**: ~3.4小时

---

## 📋 Batch 3: Model Size Comparison (Phase 3)

### GPU 0: B1 - Swin-Tiny
```bash
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size tiny \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**注意**: `model_size=tiny`, `drop_path_rate=0.2`, `dropout_rate=0.3` (更低的正则化)  
**输出**: `logs/batch3_gpu0_B1_tiny.log`

---

### GPU 1: B2 - Swin-Small
```bash
CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.25 \
  --dropout_rate 0.35 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**注意**: `model_size=small`, `drop_path_rate=0.25`, `dropout_rate=0.35` (中等正则化)  
**输出**: `logs/batch3_gpu1_B2_small.log`

**预计时间**: ~3.2小时

---

## ✅ 关键参数确认

### 所有实验共同参数:
- ✅ `--dataset koniq-10k`
- ✅ `--batch_size 32`
- ✅ `--epochs 10`
- ✅ `--patience 3`
- ✅ `--train_patch_num 20`
- ✅ `--test_patch_num 20`
- ✅ `--train_test_num 10` (10轮)
- ✅ `--lr_scheduler cosine`
- ✅ `--ranking_loss_alpha 0` (不用ranking loss)
- ✅ `--test_random_crop`
- ✅ `--no_spaq`
- ✅ `--no_color_jitter`

### 变化的参数:

| Batch | GPU 0 | GPU 1 |
|-------|-------|-------|
| **Batch 1** | LR=1e-6, base, attention+multiscale | LR=5e-7, base, attention+multiscale |
| **Batch 2** | LR=1e-6, base, NO attention | LR=1e-6, base, NO multiscale |
| **Batch 3** | LR=1e-6, tiny, drop_path=0.2, dropout=0.3 | LR=1e-6, small, drop_path=0.25, dropout=0.35 |

---

## 📊 预期结果

| 实验 | 预期SRCC | 说明 |
|------|---------|------|
| **Batch 1 - LR 1e-6** | **0.937** | 最佳模型 |
| Batch 1 - LR 5e-7 | 0.935 | 对比实验 |
| Batch 2 - A1 (No Attention) | 0.932 | 量化Attention贡献 |
| Batch 2 - A2 (No Multi-scale) | 0.930 | 量化Multi-scale贡献 |
| Batch 3 - B1 (Tiny) | 0.921 | 小模型 |
| Batch 3 - B2 (Small) | 0.933 | 中等模型 |

---

## ⚠️ 请检查以下内容

1. **参数正确性**:
   - [ ] Batch 2 的 A1 确实没有 `--attention_fusion`
   - [ ] Batch 2 的 A2 确实有 `--no_multi_scale`
   - [ ] Batch 3 的 Tiny 用的是 drop_path=0.2, dropout=0.3
   - [ ] Batch 3 的 Small 用的是 drop_path=0.25, dropout=0.35
   - [ ] 所有实验都用 `--lr 1e-6` (除了Batch 1的GPU 1用5e-7对比)

2. **数据集路径**:
   - [ ] `/root/Perceptual-IQA-CS3324/koniq-10k/` 存在
   - [ ] 数据集完整

3. **磁盘空间**:
   - [ ] 至少有 30GB 空闲空间（6个实验 × ~2.7GB checkpoint）

4. **训练时间**:
   - [ ] Batch 1: ~3.4小时 (10 rounds × 10 epochs × 2 min/epoch)
   - [ ] Batch 2: ~3.4小时
   - [ ] Batch 3: ~3.2小时
   - [ ] **总计**: ~10小时

---

## 📝 日志文件位置

所有日志将保存在 `logs/` 目录下:
```
logs/
├── batch1_gpu0_lr1e6.log           # Phase 1, GPU 0
├── batch1_gpu1_lr5e7.log           # Phase 1, GPU 1
├── batch2_gpu0_A1_no_attention.log # Phase 2, GPU 0
├── batch2_gpu1_A2_no_multiscale.log# Phase 2, GPU 1
├── batch3_gpu0_B1_tiny.log         # Phase 3, GPU 0
└── batch3_gpu1_B2_small.log        # Phase 3, GPU 1
```

---

## ✅ 确认无误后

请检查完所有命令，如果确认无误，我将创建完整的tmux自动化脚本。

**脚本将包含**:
1. 自动创建tmux会话
2. 顺序执行3个batch
3. 每个batch在2个tmux窗口中并行
4. 自动等待batch完成再启动下一个
5. 失败重试机制
6. 完成后自动提取结果
7. 发送完成通知

**是否继续创建自动化脚本？**

