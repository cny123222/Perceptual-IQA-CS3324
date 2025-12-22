# 🚀 机器B快速设置指南

**目标**: 5分钟内在新机器上启动实验

---

## ⚡ 快速命令 (复制粘贴即可)

### 步骤1: 克隆代码 (1分钟)

```bash
cd /root
git clone https://github.com/cny123222/Perceptual-IQA-CS3324.git
cd Perceptual-IQA-CS3324
```

### 步骤2: 安装依赖 (2-3分钟)

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy pillow tqdm timm kornia matplotlib tensorboard
```

**或者使用requirements.txt**:
```bash
pip install -r requirements.txt
```

### 步骤3: 传输数据集 (取决于网速)

**在机器A上执行**:
```bash
cd /root/Perceptual-IQA-CS3324
tar -czf koniq-10k.tar.gz koniq-10k/

# 替换 <machine_b_ip> 为机器B的IP地址
scp koniq-10k.tar.gz root@<machine_b_ip>:/root/Perceptual-IQA-CS3324/
```

**在机器B上执行**:
```bash
cd /root/Perceptual-IQA-CS3324
tar -xzf koniq-10k.tar.gz
rm koniq-10k.tar.gz
```

### 步骤4: 验证环境 (30秒)

```bash
cd /root/Perceptual-IQA-CS3324

# 检查GPU
nvidia-smi

# 检查数据集
ls koniq-10k/ | head

# 验证代码
python -c "import torch; print('✅ PyTorch:', torch.__version__); print('✅ CUDA:', torch.cuda.is_available())"
```

### 步骤5: 启动实验 (1分钟)

```bash
# 在tmux中运行，防止SSH断开
tmux new -s experiments_b

# 运行所有6个实验
cd /root/Perceptual-IQA-CS3324
./run_machine_b.sh
```

**完成！** 现在可以断开SSH，实验会在后台继续运行。

---

## 📊 查看进度

### 重新连接到tmux会话

```bash
tmux attach -t experiments_b
```

### 查看日志

```bash
# 查看最新日志
tail -f logs/swin_*.log

# 查看所有日志
ls -lth logs/ | head -20
```

### 检查GPU使用

```bash
watch -n 1 nvidia-smi
```

---

## ✅ 实验完成后

### 提交结果到Git

```bash
cd /root/Perceptual-IQA-CS3324

# 添加日志
git add logs/*.log

# 提交
git commit -m "feat: Machine B experiment results (D1-D4, E1-E4)"

# 推送
git push origin master
```

---

## 🆘 常见问题

### Q1: pip安装太慢？

**A**: 使用清华镜像
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 数据集传输失败？

**A**: 使用rsync (支持断点续传)
```bash
# 在机器A上执行
rsync -avz --progress /root/Perceptual-IQA-CS3324/koniq-10k/ \
  root@<machine_b_ip>:/root/Perceptual-IQA-CS3324/koniq-10k/
```

### Q3: CUDA版本不匹配？

**A**: 检查CUDA版本并安装对应PyTorch
```bash
nvidia-smi  # 查看CUDA版本

# 如果是CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q4: 如何停止所有实验？

**A**: 
```bash
pkill -f train_swin.py
```

---

## 📋 实验清单

机器B负责运行以下6个实验：

- [x] D1: Weight Decay = 5e-5
- [x] D2: Weight Decay = 1e-4
- [x] D4: Weight Decay = 4e-4
- [x] E1: Learning Rate = 2.5e-6
- [x] E3: Learning Rate = 7.5e-6
- [x] E4: Learning Rate = 1e-5

**预计总时间**: 30-60分钟

---

## 🔗 相关文档

- **完整指南**: `MULTI_MACHINE_SETUP.md`
- **所有实验命令**: `ALL_EXPERIMENTS_COMMANDS.md`
- **实验跟踪**: `EXPERIMENTS_LOG_TRACKER.md`

---

**最后更新**: 2025-12-22

