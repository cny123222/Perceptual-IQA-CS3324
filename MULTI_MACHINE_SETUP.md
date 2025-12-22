# 多机器并行训练方案

**目标**: 在两台机器上同时运行实验，加速完成14个实验

---

## 📋 方案概述

### 策略
- **机器A (当前机器)**: 运行 7-8 个实验
- **机器B (新机器)**: 运行 6-7 个实验
- **数据同步**: 使用 Git + 手动传输数据集
- **结果同步**: 通过 Git 提交日志和checkpoint

---

## 🔧 第一步：准备新机器 (机器B)

### 1.1 克隆代码仓库

```bash
cd /root
git clone https://github.com/cny123222/Perceptual-IQA-CS3324.git
cd Perceptual-IQA-CS3324
```

### 1.2 安装Python环境

**方法1: 使用requirements.txt (推荐)**

```bash
# 创建虚拟环境 (可选但推荐)
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**方法2: 手动安装核心依赖**

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.3.2 scipy==1.16.3 pillow==11.3.0
pip install tqdm==4.66.2 timm==1.0.22 kornia==0.8.2
pip install matplotlib==3.10.5 tensorboard==2.20.0
```

### 1.3 传输数据集

**数据集大小估算**: KonIQ-10k 约 5-10GB

**方法1: 从机器A直接传输 (推荐，最快)**

在**机器A**上执行：
```bash
# 压缩数据集
cd /root/Perceptual-IQA-CS3324
tar -czf koniq-10k.tar.gz koniq-10k/

# 使用scp传输到机器B
# 替换 <machine_b_ip> 和 <machine_b_user>
scp koniq-10k.tar.gz <machine_b_user>@<machine_b_ip>:/root/Perceptual-IQA-CS3324/
```

在**机器B**上执行：
```bash
cd /root/Perceptual-IQA-CS3324
tar -xzf koniq-10k.tar.gz
rm koniq-10k.tar.gz  # 解压后删除压缩包
```

**方法2: 使用rsync (更快，支持断点续传)**

在**机器A**上执行：
```bash
rsync -avz --progress /root/Perceptual-IQA-CS3324/koniq-10k/ \
  <machine_b_user>@<machine_b_ip>:/root/Perceptual-IQA-CS3324/koniq-10k/
```

**方法3: 从云存储下载 (如果有备份)**

如果你的数据集在百度云/阿里云OSS/AWS S3等：
```bash
# 示例：使用百度云盘命令行工具
bypy download koniq-10k /root/Perceptual-IQA-CS3324/koniq-10k
```

### 1.4 验证环境

在**机器B**上执行：
```bash
cd /root/Perceptual-IQA-CS3324

# 检查GPU
nvidia-smi

# 检查数据集
ls -lh koniq-10k/ | head -10
wc -l koniq-10k/koniq10k_scores_and_distributions.csv

# 测试代码 (快速验证，不实际训练)
python -c "
import torch
import models_swin
import data_loader
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ GPU count:', torch.cuda.device_count())
print('✅ All imports successful!')
"
```

**预期输出**:
```
✅ PyTorch: 2.8.0+cu128
✅ CUDA available: True
✅ GPU count: 4
✅ All imports successful!
```

---

## 🎯 第二步：分配实验任务

### 机器A (当前机器) - 运行8个实验

**优先级1-2: 核心消融 + Ranking敏感度**

| 实验 | 描述 | 预计时间 |
|------|------|---------|
| A1 | Remove Attention | 5-10 min |
| A2 | Remove Ranking | 5-10 min |
| A3 | Remove Multi-scale | 5-10 min |
| C1 | Alpha=0.1 | 5-10 min |
| C2 | Alpha=0.5 | 5-10 min |
| C3 | Alpha=0.7 | 5-10 min |
| B1 | Swin-Tiny | 5-10 min |
| B2 | Swin-Small | 5-10 min |

**总计**: 40-80分钟

### 机器B (新机器) - 运行6个实验

**优先级3-4: 正则化 + 学习率敏感度**

| 实验 | 描述 | 预计时间 |
|------|------|---------|
| D1 | WD=5e-5 | 5-10 min |
| D2 | WD=1e-4 | 5-10 min |
| D4 | WD=4e-4 | 5-10 min |
| E1 | LR=2.5e-6 | 5-10 min |
| E3 | LR=7.5e-6 | 5-10 min |
| E4 | LR=1e-5 | 5-10 min |

**总计**: 30-60分钟

---

## 🚀 第三步：运行实验

### 在机器A上运行

参考 `ALL_EXPERIMENTS_COMMANDS.md` 中的命令，依次运行 A1-A3, C1-C3, B1-B2

**示例** (在tmux中运行):
```bash
# 创建tmux会话
tmux new -s experiments_a

# 运行A1
cd /root/Perceptual-IQA-CS3324 && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 \
  --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --ranking_loss_alpha 0.3 --lr 5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq

# A1完成后运行A2...
# (依此类推)
```

### 在机器B上运行

创建一个脚本文件 `run_machine_b.sh`:

```bash
#!/bin/bash

cd /root/Perceptual-IQA-CS3324

# D1: WD=5e-5
echo "========== Starting D1: WD=5e-5 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 \
  --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0.3 --lr 5e-6 --weight_decay 5e-5 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq

# D2: WD=1e-4
echo "========== Starting D2: WD=1e-4 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 \
  --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0.3 --lr 5e-6 --weight_decay 1e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq

# D4: WD=4e-4
echo "========== Starting D4: WD=4e-4 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 \
  --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0.3 --lr 5e-6 --weight_decay 4e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq

# E1: LR=2.5e-6
echo "========== Starting E1: LR=2.5e-6 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 \
  --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0.3 --lr 2.5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq

# E3: LR=7.5e-6
echo "========== Starting E3: LR=7.5e-6 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 \
  --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0.3 --lr 7.5e-6 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq

# E4: LR=1e-5
echo "========== Starting E4: LR=1e-5 =========="
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 \
  --patience 5 --train_patch_num 20 --test_patch_num 20 --train_test_num 1 \
  --attention_fusion --ranking_loss_alpha 0.3 --lr 1e-5 --weight_decay 2e-4 \
  --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine \
  --test_random_crop --no_spaq

echo "========== All experiments completed! =========="
```

运行脚本：
```bash
chmod +x run_machine_b.sh

# 在tmux中运行，防止SSH断开
tmux new -s experiments_b
./run_machine_b.sh
```

---

## 📊 第四步：结果同步

### 4.1 机器B完成后，提交结果到Git

在**机器B**上执行：

```bash
cd /root/Perceptual-IQA-CS3324

# 添加日志文件
git add logs/*.log

# 提交 (不包含checkpoint，太大)
git commit -m "feat: Add experiment results from Machine B

Completed experiments:
- D1: WD=5e-5
- D2: WD=1e-4  
- D4: WD=4e-4
- E1: LR=2.5e-6
- E3: LR=7.5e-6
- E4: LR=1e-5

All logs saved to logs/ directory."

# 推送到GitHub
git push origin master
```

### 4.2 机器A拉取结果

在**机器A**上执行：

```bash
cd /root/Perceptual-IQA-CS3324

# 拉取机器B的结果
git pull origin master

# 查看新增的日志
ls -lth logs/ | head -20
```

### 4.3 更新实验跟踪文档

在**任意一台机器**上执行：

```bash
# 提取所有实验的最佳结果
cd /root/Perceptual-IQA-CS3324

# 示例：提取D1结果
grep "best model" logs/swin_*_wd5e-5_*.log | tail -1

# 手动更新 EXPERIMENTS_LOG_TRACKER.md
# 然后提交
git add EXPERIMENTS_LOG_TRACKER.md
git commit -m "docs: Update experiment results from both machines"
git push origin master
```

---

## 🔄 第五步：Checkpoint同步 (可选)

**注意**: Checkpoint文件很大 (~500MB-1GB 每个)，**不建议**提交到Git。

### 方法1: 只同步最佳模型 (推荐)

在**机器B**上执行：
```bash
cd /root/Perceptual-IQA-CS3324

# 找到最佳checkpoint
find checkpoints/ -name "best_model_*.pkl" -type f

# 使用scp传输到机器A
scp checkpoints/*/best_model_*.pkl \
  <machine_a_user>@<machine_a_ip>:/root/Perceptual-IQA-CS3324/checkpoints_from_b/
```

### 方法2: 使用Git LFS (如果需要版本控制)

```bash
# 在两台机器上都安装Git LFS
git lfs install

# 配置LFS追踪.pkl文件
git lfs track "*.pkl"
git add .gitattributes

# 提交checkpoint
git add checkpoints/
git commit -m "chore: Add best model checkpoints"
git push origin master
```

### 方法3: 使用云存储

上传到百度云/阿里云OSS/AWS S3，然后分享链接。

---

## ⚠️ 注意事项

### 1. 避免Git冲突

- **机器A**: 负责提交 A1-A3, C1-C3, B1-B2 的日志
- **机器B**: 负责提交 D1-D4, E1-E4 的日志
- 每次提交前先 `git pull`，确保同步

### 2. 数据集一致性

确保两台机器的数据集**完全一致**：
```bash
# 在两台机器上都运行
md5sum koniq-10k/koniq10k_scores_and_distributions.csv
# 输出应该相同
```

### 3. 环境一致性

确保两台机器的PyTorch版本一致，避免结果差异：
```bash
python -c "import torch; print(torch.__version__)"
# 应该都是 2.8.0+cu128 或类似版本
```

### 4. 随机种子

代码已经设置了 `random_seed=42`，确保可复现性。

---

## 📝 快速启动清单

### 机器B设置清单

- [ ] 克隆代码仓库
- [ ] 安装Python依赖 (`pip install -r requirements.txt`)
- [ ] 传输数据集 (scp/rsync)
- [ ] 验证环境 (GPU, 数据集, imports)
- [ ] 创建 `run_machine_b.sh` 脚本
- [ ] 在tmux中运行脚本
- [ ] 实验完成后提交日志到Git

### 机器A操作清单

- [ ] 运行 A1-A3, C1-C3, B1-B2 实验
- [ ] 提交日志到Git
- [ ] 拉取机器B的结果
- [ ] 更新 `EXPERIMENTS_LOG_TRACKER.md`
- [ ] (可选) 同步checkpoint

---

## 🎯 预期时间线

| 时间 | 机器A | 机器B |
|------|-------|-------|
| T+0 | 开始设置 | 开始设置 |
| T+30min | 设置完成，开始A1 | 设置完成，开始D1 |
| T+40min | A1完成，开始A2 | D1完成，开始D2 |
| T+50min | A2完成，开始A3 | D2完成，开始D4 |
| T+60min | A3完成，开始C1 | D4完成，开始E1 |
| T+70min | C1完成，开始C2 | E1完成，开始E3 |
| T+80min | C2完成，开始C3 | E3完成，开始E4 |
| T+90min | C3完成，开始B1 | E4完成，提交结果 |
| T+100min | B1完成，开始B2 | - |
| T+110min | B2完成，提交结果 | - |
| T+120min | 拉取机器B结果，更新文档 | - |

**总时间**: 约2小时 (vs 单机4小时+)

---

## 🆘 故障排除

### 问题1: 数据集传输太慢

**解决**: 
- 使用 `rsync` 而不是 `scp`
- 压缩后传输: `tar -czf | ssh user@host "tar -xzf -C /path"`
- 如果两台机器在同一内网，速度应该很快

### 问题2: 依赖安装失败

**解决**:
```bash
# 使用清华镜像源加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题3: Git push冲突

**解决**:
```bash
git pull --rebase origin master
# 解决冲突后
git push origin master
```

### 问题4: CUDA版本不匹配

**解决**:
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应版本的PyTorch
# 例如CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📚 相关文档

- `ALL_EXPERIMENTS_COMMANDS.md` - 所有14个实验的详细命令
- `EXPERIMENTS_LOG_TRACKER.md` - 实验结果跟踪表
- `FINAL_ABLATION_PLAN.md` - 消融实验设计
- `requirements.txt` - Python依赖列表

---

**最后更新**: 2025-12-22
**作者**: AI Assistant
**状态**: Ready to use ✅

