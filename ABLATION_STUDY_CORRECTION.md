# 消融实验设计纠正

## 问题发现

当前的消融实验设计是**反向的**（从完整版本逐步移除组件），这不符合标准的消融实验原则。正确的消融实验应该是**从基础版本逐步添加组件**，以显示每个组件的独立贡献。

## 原始HyperIQA架构（baseline）

从`models.py`可以看到，原始HyperIQA包含：
- **Backbone**: ResNet50（2048维输出）
- **HyperNet**: 生成target network的参数
- **LDA Modules**: 4个Local Distortion Aware模块
- **Target Network**: 用于最终质量预测
- **特征使用**: 仅使用ResNet50的layer4输出（单尺度）
- **融合方式**: 无多尺度融合，无注意力机制

## 我们的改进版本（models_swin.py）

相比原始HyperIQA，我们添加了以下组件：

### 1. **Swin Transformer Backbone**（替换ResNet50）
- 使用timm的预训练Swin Transformer
- 输出4个阶段的特征：[feat0, feat1, feat2, feat3]
- Tiny: [96, 192, 384, 768]维
- Small: [96, 192, 384, 768]维
- Base: [128, 256, 512, 1024]维

### 2. **Multi-scale Feature Fusion**（新增）
- 将所有4个阶段的特征统一到7x7空间尺寸
- 在通道维度拼接：Tiny/Small=1440维, Base=1920维
- 由`use_multiscale`参数控制

### 3. **Attention-based Fusion**（新增）
- `MultiScaleAttention`模块
- 使用feat3生成4个尺度的注意力权重
- 动态加权融合多尺度特征
- 由`use_attention`参数控制（需要`use_multiscale=True`）

## 正确的消融实验设计

### 实验序列（从简单到复杂）

| 实验ID | 架构配置 | 说明 | 预期SRCC |
|--------|---------|------|----------|
| **C0** | ResNet50 + HyperNet | 原始HyperIQA | 0.907 ✓ |
| **C1** | Swin-Tiny + HyperNet | 仅换backbone | ? |
| **C2** | Swin-Tiny + HyperNet + Multi-scale | 添加多尺度融合 | ? |
| **C3** | Swin-Tiny + HyperNet + Multi-scale + Attention | 添加注意力机制（完整版本） | 0.9378 ✓ |

### 每个组件的预期贡献

- **C1 - C0**: Swin Transformer相比ResNet50的提升
- **C2 - C1**: Multi-scale融合的贡献
- **C3 - C2**: Attention机制的贡献
- **C3 - C0**: 总体提升 = +3.08% SRCC

## 当前已有的实验结果

### ✓ 已完成
- **C0** (ResNet50): SRCC = 0.907
- **C3** (完整版本): SRCC = 0.9378

### ✗ 错误的实验（逆向消融）
- **A1** (移除Attention): 这相当于C2，但实验设计是反的
- **A2** (移除Multi-scale): 这相当于C1，但实验设计是反的

## 需要重新运行的实验

### 实验C1: 仅换backbone（Swin + HyperNet）
```bash
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
    --dataset koniq-10k \
    --train_patch_num 1 \
    --batch_size 32 \
    --epochs 10 \
    --lr 5e-7 \
    --weight_decay 1e-4 \
    --drop_path_rate 0.2 \
    --dropout_rate 0.3 \
    --lr_decay_rate 0.95 \
    --lr_decay_freq 2 \
    --loss_type l1 \
    --train_test_num 1 \
    --patience 3 \
    --model_size tiny \
    --no_multiscale \
    --exp_name "C1_swin_only_lr5e7" \
    2>&1 | tee logs/C1_swin_only_lr5e7_$(date +%Y%m%d_%H%M%S).log
```

### 实验C2: 添加多尺度（Swin + Multi-scale）
```bash
CUDA_VISIBLE_DEVICES=1 python train_swin.py \
    --dataset koniq-10k \
    --train_patch_num 1 \
    --batch_size 32 \
    --epochs 10 \
    --lr 5e-7 \
    --weight_decay 1e-4 \
    --drop_path_rate 0.2 \
    --dropout_rate 0.3 \
    --lr_decay_rate 0.95 \
    --lr_decay_freq 2 \
    --loss_type l1 \
    --train_test_num 1 \
    --patience 3 \
    --model_size tiny \
    --no_attention_fusion \
    --exp_name "C2_swin_multiscale_lr5e7" \
    2>&1 | tee logs/C2_swin_multiscale_lr5e7_$(date +%Y%m%d_%H%M%S).log
```

注意：
- C1需要`--no_multiscale`（单尺度）
- C2需要`--no_attention_fusion`（多尺度但无注意力）
- C3就是当前的baseline（多尺度+注意力）

## 代码检查

需要确认`train_swin.py`中是否有`--no_attention_fusion`参数：

```python
# 应该有类似的参数定义
parser.add_argument('--no_multiscale', dest='use_multiscale', action='store_false',
                    help='Disable multi-scale feature fusion (use single scale)')
parser.add_argument('--no_attention_fusion', dest='use_attention', action='store_false',
                    help='Disable attention-based fusion (use simple concatenation)')
```

## 实验优先级

1. **高优先级**：C1（仅换backbone）- 这是最关键的，显示Swin相比ResNet50的提升
2. **中优先级**：C2（添加多尺度）- 显示多尺度融合的贡献
3. **低优先级**：重新验证C3（当前baseline）- 已有结果，可以复用

## 预期论文叙述

正确的消融实验后，论文可以这样写：

> "我们从原始HyperIQA（ResNet50）出发，逐步添加以下改进：
> 1. 将backbone从ResNet50替换为Swin Transformer，SRCC从0.907提升到X.XXX（+Y.Y%）
> 2. 引入多尺度特征融合，SRCC进一步提升到X.XXX（+Y.Y%）
> 3. 添加注意力机制进行动态加权融合，最终SRCC达到0.9378（+3.08%）
>
> 这表明：（1）Swin Transformer是性能提升的主要来源，（2）多尺度融合和注意力机制提供了额外的增益。"

## 模型尺寸实验（保持不变）

B1和B2实验设计是正确的，因为它们是在最优架构（C3）下比较不同模型大小：
- B1: Swin-Small
- B2: Swin-Base

这些实验不需要修改。

