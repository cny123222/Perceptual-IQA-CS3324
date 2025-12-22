# QualiCLIP Training Scripts 使用指南

本文档说明如何使用自动化脚本进行QualiCLIP预训练和微调。

---

## 🚀 快速开始

### 方法1: 全自动流程（推荐）

等待预训练完成后**自动**开始微调：

```bash
cd /root/Perceptual-IQA-CS3324
./auto_run_qualiclip_finetune.sh
```

**功能：**
- ✅ 自动等待预训练进程完成
- ✅ 验证预训练checkpoint
- ✅ 自动启动微调训练
- ✅ 使用优化的学习率（基于实验经验）

**适用场景：**
- 预训练正在后台运行，想要完成后自动开始微调
- 不想手动监控，让脚本自动处理一切

---

### 方法2: 简化版（手动启动）

预训练完成后，**手动**运行微调：

```bash
cd /root/Perceptual-IQA-CS3324
./run_qualiclip_finetune_simple.sh
```

**功能：**
- 自动找到最新的预训练权重
- 使用优化的默认参数
- 更灵活，可以自定义参数

**自定义参数：**
```bash
./run_qualiclip_finetune_simple.sh [数据集] [epochs] [主学习率] [encoder学习率] [batch_size]

# 示例：
./run_qualiclip_finetune_simple.sh koniq10k 50 1e-6 5e-7 8
```

---

## 📊 学习率配置说明

根据您的实验经验，我们采用以下学习率：

| 参数 | 学习率 | 说明 |
|------|--------|------|
| **HyperNet** | `1e-6` | 基于您的baseline实验结果 |
| **Encoder (预训练)** | `5e-7` | 更小，保护预训练特征 |

**为什么Encoder用更小的学习率？**
- 预训练encoder已经学到了有用的通用特征
- 使用更小学习率进行"精细调整"，避免破坏预训练的知识
- 这种**differential learning rate**策略是迁移学习的常用技巧

---

## 📁 输出文件

### 预训练阶段
```
checkpoints/qualiclip_pretrain_YYYYMMDD_HHMMSS/
├── swin_base_epoch5.pkl   # 中间checkpoint
└── swin_base_epoch10.pkl  # 最终预训练权重
```

### 微调阶段
```
checkpoints/swin_base_qualiclip_pretrained/
├── best_model.pkl          # 验证集上最佳模型
├── checkpoint_epoch*.pkl   # 每5个epoch的checkpoint
└── training_history.json   # 训练曲线数据
```

### 日志文件
```
logs/
├── qualiclip_pretrain_run.log      # 预训练日志
└── qualiclip_finetune_run.log      # 微调日志
```

---

## ⚙️ 完整参数说明

### 预训练参数 (pretrain_qualiclip.py)

```bash
python pretrain_qualiclip.py \
    --data_root /path/to/koniq-10k \
    --model_size base \              # Swin模型大小: tiny/small/base
    --epochs 10 \                    # 预训练epochs
    --batch_size 8 \                 # Batch size
    --lr 5e-5 \                      # 学习率
    --crop_size 224 \                # 裁剪大小
    --base_size 512 \                # 基础图像大小
    --overlap_ratio 0.5 \            # 裁剪重叠比例
    --num_workers 4                  # DataLoader workers
```

### 微调参数 (train_swin.py with pre-training)

```bash
python train_swin.py \
    --database koniq10k \                           # 数据集
    --model_name swin_base_qualiclip \             # 模型名称
    --batch_size 8 \                               # Batch size
    --epochs 50 \                                  # 微调epochs
    --lr 1e-6 \                                    # HyperNet学习率
    --pretrained_encoder /path/to/weights.pkl \    # 预训练权重路径
    --lr_encoder_pretrained 5e-7                   # Encoder学习率
```

---

## 🔧 常见问题

### Q1: 脚本一直等待，如何检查预训练是否还在运行？

```bash
# 查看进程
ps aux | grep pretrain_qualiclip

# 查看最新日志
tail -f logs/qualiclip_pretrain_run.log

# 查看GPU使用
nvidia-smi
```

### Q2: 预训练失败了，如何重新开始？

```bash
# 删除失败的checkpoint
rm -rf checkpoints/qualiclip_pretrain_*

# 重新运行预训练
python pretrain_qualiclip.py --data_root /root/Perceptual-IQA-CS3324/koniq-10k --model_size base --epochs 10 --batch_size 8
```

### Q3: 想要调整学习率，如何修改？

**方法1: 修改脚本中的默认值**
```bash
nano auto_run_qualiclip_finetune.sh
# 修改 LR_MAIN 和 LR_ENCODER 的值
```

**方法2: 使用简化版脚本，传入自定义参数**
```bash
./run_qualiclip_finetune_simple.sh koniq10k 50 2e-6 1e-6 8
```

**方法3: 直接运行python命令**
```bash
python train_swin.py \
    --database koniq10k \
    --model_name my_custom_name \
    --lr 2e-6 \
    --pretrained_encoder checkpoints/qualiclip_pretrain_*/swin_base_epoch10.pkl \
    --lr_encoder_pretrained 1e-6
```

### Q4: 内存不足 (OOM) 怎么办？

减小batch size：
```bash
./run_qualiclip_finetune_simple.sh koniq10k 50 1e-6 5e-7 4  # batch_size=4
```

### Q5: 如何在后台运行？

```bash
# 使用nohup
nohup ./auto_run_qualiclip_finetune.sh > pipeline.log 2>&1 &

# 或使用screen
screen -S qualiclip
./auto_run_qualiclip_finetune.sh
# Ctrl+A, D 分离会话
```

---

## 📈 监控训练进度

### 实时查看训练日志

```bash
# 微调训练日志
tail -f logs/qualiclip_finetune_run.log

# 查看最近100行
tail -100 logs/qualiclip_finetune_run.log

# 搜索最佳结果
grep "best" logs/qualiclip_finetune_run.log
```

### 查看GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或
nvidia-smi -l 1
```

---

## 🎯 训练完成后

### 1. 查看训练结果

```bash
# 查看最终模型
ls -lh checkpoints/swin_base_qualiclip_pretrained/

# 查看训练历史
cat checkpoints/swin_base_qualiclip_pretrained/training_history.json
```

### 2. 测试模型

```bash
# 在测试集上评估
python test_swin.py \
    --model_path checkpoints/swin_base_qualiclip_pretrained/best_model.pkl \
    --test_datasets koniq10k spaq kadid agiqa
```

### 3. 与baseline对比

创建对比表格，记录：
- KonIQ-10k测试集 SRCC/PLCC
- 跨数据集泛化性能
- 训练收敛速度
- 最终性能提升

---

## 📝 实验记录模板

```markdown
## QualiCLIP Pre-training Experiment

### 预训练
- **数据集**: KonIQ-10k train (7046 images)
- **Epochs**: 10
- **学习率**: 5e-5
- **最终loss**: X.XXXX

### 微调
- **数据集**: KonIQ-10k train
- **Epochs**: 50
- **HyperNet LR**: 1e-6
- **Encoder LR**: 5e-7
- **最佳验证SRCC**: X.XXXX (epoch XX)

### 测试结果
| Dataset | SRCC | PLCC |
|---------|------|------|
| KonIQ-10k | X.XXX | X.XXX |
| SPAQ | X.XXX | X.XXX |
| KADID-10K | X.XXX | X.XXX |
| AGIQA-3K | X.XXX | X.XXX |

### 观察
- [ ] 训练收敛速度
- [ ] 过拟合情况
- [ ] 跨数据集泛化
- [ ] 与baseline对比
```

---

## 🔗 相关文档

- `QUALICLIP_EXPERIMENT_PLAN.md` - 完整实验计划
- `QUALICLIP_IMPLEMENTATION_SUMMARY.md` - 实现细节
- `QUALICLIP_PRETRAIN_GUIDE.md` - 预训练指南
- `benchmarks/QualiCLIP/suggestions.md` - 原始设计方案

---

## ⚡ 快速命令参考

```bash
# 1. 运行预训练（如果还没运行）
python pretrain_qualiclip.py --data_root koniq-10k --model_size base --epochs 10 --batch_size 8

# 2. 自动等待并微调
./auto_run_qualiclip_finetune.sh

# 3. 或手动启动微调
./run_qualiclip_finetune_simple.sh

# 4. 查看训练进度
tail -f logs/qualiclip_finetune_run.log

# 5. 测试模型
python test_swin.py --model_path checkpoints/swin_base_qualiclip_pretrained/best_model.pkl
```

---

**祝训练顺利！🚀**

