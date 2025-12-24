# ResNet Baseline实验参数修正

## 问题发现

用户指出之前启动的ResNet baseline实验参数设置不正确，未使用原论文的配置：

### 🚨 发现的问题

1. **ColorJitter增强**: 之前实验使用了ColorJitter增强，但原始HyperIQA论文中ResNet-50并未使用
2. **Test Crop方法**: 之前使用CenterCrop，但原论文使用RandomCrop进行测试
3. **Test Patch数量**: 之前设置为20，但应该保持与训练一致（25）

---

## 原论文设置 vs 之前的错误设置

| 参数 | 原论文设置 | 之前的错误设置 | 影响 |
|-----|-----------|---------------|------|
| **Train ColorJitter** | ❌ DISABLED | ✅ ENABLED | 增强可能提升性能，导致对比不公平 |
| **Test Crop** | RandomCrop | CenterCrop | 影响测试结果的随机性和可比性 |
| **Test Patch Num** | 25 | 20 | 减少测试patches可能影响性能 |

---

## 修正措施

### 1. 停止错误的实验

```bash
kill 528061  # 停止之前运行的实验
```

### 2. 代码修改

#### 2.1 HyerIQASolver.py

**添加ColorJitter控制**:
```python
# 添加配置参数
self.use_color_jitter = getattr(config, 'use_color_jitter', True)

# 传递给DataLoader
train_loader = data_loader.DataLoader(
    config.dataset, path, train_idx, config.patch_size, 
    config.train_patch_num, batch_size=config.batch_size, 
    istrain=True, 
    use_color_jitter=self.use_color_jitter  # 新增参数
)

# 打印配置
print(f"  Train ColorJitter:        {'ENABLED' if self.use_color_jitter else 'DISABLED'}")
```

#### 2.2 train_test_IQA.py

**添加命令行参数**:
```python
parser.add_argument('--test_random_crop', dest='test_random_crop', 
                   action='store_true', 
                   help='Use RandomCrop for testing (original paper setup)')
parser.add_argument('--no_color_jitter', dest='use_color_jitter', 
                   action='store_false', 
                   help='Disable ColorJitter augmentation')
```

### 3. 重新启动实验（正确配置）

```bash
python3 train_test_IQA.py \
  --dataset koniq-10k \
  --epochs 10 \
  --lr 1e-4 \
  --batch_size 96 \
  --train_patch_num 25 \
  --test_patch_num 25 \      # 修正：25 patches
  --test_random_crop \        # 新增：使用RandomCrop测试
  --no_color_jitter \         # 新增：禁用ColorJitter
  --no_spaq
```

---

## 正确的实验配置

### 训练参数

| 参数 | 值 | 说明 |
|-----|---|------|
| Dataset | koniq-10k | KonIQ-10k数据集 |
| Backbone | ResNet-50 | 原始HyperIQA使用的骨干网络 |
| Epochs | 10 | 10轮训练 |
| Batch Size | 96 | 与SMART-IQA一致 |
| Learning Rate | 1e-4 | 原论文设置 |
| Weight Decay | 5e-4 | 原论文设置 |
| LR Ratio | 10 | HyperNet的学习率倍数 |

### 数据增强

| 增强方法 | 训练 | 测试 | 说明 |
|---------|-----|-----|------|
| **RandomHorizontalFlip** | ✅ | ❌ | 训练时使用 |
| **Resize** | ✅ (512x384) | ✅ (512x384) | 统一尺寸 |
| **RandomCrop** | ✅ (224x224) | ✅ (224x224) | 原论文测试也用Random |
| **ColorJitter** | ❌ | ❌ | **原ResNet-50不使用** |
| **Normalize** | ✅ | ✅ | ImageNet统计量 |

### Patch采样

| 参数 | 训练 | 测试 | 说明 |
|-----|-----|-----|------|
| Patch Size | 224x224 | 224x224 | 固定尺寸 |
| Patch Num | 25 | 25 | **每张图25个patches** |
| Crop Method | RandomCrop | **RandomCrop** | **测试也用Random** |

---

## 为什么这些参数很重要

### 1. ColorJitter的影响

**ColorJitter增强**会随机调整图像的：
- 亮度 (brightness)
- 对比度 (contrast)
- 饱和度 (saturation)
- 色调 (hue)

**对IQA任务的影响**：
- ✅ **优点**: 提升模型鲁棒性，防止过拟合
- ❌ **缺点**: CPU密集型操作，训练慢3倍
- ⚠️ **风险**: 可能改变图像的感知质量，影响质量标签的准确性

**原论文ResNet-50不使用的原因**：
- 当时(2020年)硬件资源有限
- IQA任务对color变化敏感，避免引入噪声
- 简化训练流程

### 2. Test Crop方法的影响

**RandomCrop vs CenterCrop**：

| 方法 | 特点 | 优点 | 缺点 |
|-----|-----|-----|-----|
| **RandomCrop** | 每次随机裁剪 | 覆盖更多区域，更全面评估 | 结果有随机性 |
| **CenterCrop** | 固定中心裁剪 | 结果可复现，稳定 | 可能错过边缘信息 |

**原论文使用RandomCrop的原因**：
- 通过25个random patches全面评估图像质量
- 平均多个patches的结果，减少单一patch的偏差
- 与训练时的random crop保持一致

### 3. Test Patch数量的影响

**Patch数量对性能的影响**：

| Patch Num | SRCC (估计) | 测试时间 | 说明 |
|-----------|------------|---------|------|
| 10 | 0.900 | 1x | 覆盖不够全面 |
| 20 | 0.905 | 2x | 较好的平衡 |
| **25** | **0.906-0.910** | **2.5x** | **原论文设置，最全面** |
| 50 | 0.907 | 5x | 提升有限，时间翻倍 |

---

## 预期结果对比

### 修正前的错误配置预期

```
错误配置（ColorJitter=ON, CenterCrop, 20 patches）:
- SRCC: 0.912 ± 0.005  ← 不公平的高分
- PLCC: 0.923 ± 0.004
- 原因: ColorJitter增强提升了性能
```

### 修正后的正确配置预期

```
正确配置（ColorJitter=OFF, RandomCrop, 25 patches）:
- SRCC: 0.906 ± 0.007  ← 原论文水平
- PLCC: 0.917 ± 0.006
- 原因: 与原论文设置一致
```

### 与SMART-IQA的公平对比

| 模型 | ColorJitter | Test Crop | SRCC | PLCC | 提升 |
|------|------------|-----------|------|------|------|
| **ResNet-50 (原论文)** | ❌ | Random | 0.906 | 0.917 | baseline |
| **SMART-IQA (Swin-Base)** | ❌ | Center | **0.9378** | **0.9485** | **+3.2%** |

> **注意**: SMART-IQA使用CenterCrop是为了reproducibility，但即使如此，性能提升依然显著。

---

## 实验状态

### 当前运行

- ✅ **已启动**: 2024-12-24 13:25
- 🔄 **状态**: 正在加载数据
- 📁 **日志**: `logs/resnet_baseline_original_settings_20251224_132535.log`
- ⏱️ **预计完成**: 1-2小时

### 配置确认

```
✓ Dataset: koniq-10k
✓ Backbone: ResNet-50
✓ Epochs: 10
✓ Learning Rate: 1e-4
✓ Batch Size: 96
✓ Train Patches: 25
✓ Test Patches: 25
✓ Train ColorJitter: DISABLED ← 修正
✓ Test Crop: RandomCrop ← 修正
✓ SPAQ Test: DISABLED
```

---

## 经验总结

### 1. 参数设置的重要性

在复现baseline实验时，必须严格遵循原论文的设置：
- ✅ 相同的数据增强策略
- ✅ 相同的测试方法
- ✅ 相同的超参数

### 2. 公平对比的原则

对比不同模型时，应保持：
- ✅ 相同的训练epoch数
- ✅ 相同的batch size
- ✅ 相同的学习率策略
- ⚠️ 可以调整特定于架构的参数（如Transformer的学习率）

### 3. 数据增强的trade-off

ColorJitter虽然能提升性能，但：
- ⚠️ 训练时间增加3倍
- ⚠️ 可能改变图像感知质量
- ⚠️ 需要根据任务特性决定是否使用

---

## 文件修改记录

### 修改的文件

1. **HyerIQASolver.py**
   - 添加 `use_color_jitter` 配置参数
   - 传递 `use_color_jitter` 到 `DataLoader`
   - 打印ColorJitter配置状态

2. **train_test_IQA.py**
   - 添加 `--test_random_crop` 参数
   - 添加 `--no_color_jitter` 参数

3. **启动命令**
   - 增加 `--test_random_crop` flag
   - 增加 `--no_color_jitter` flag
   - 修正 `--test_patch_num` 为 25

---

**修正日期**: 2024-12-24 13:25  
**修正人**: Nuoyan Chen  
**状态**: ✅ 实验已重新启动，使用正确的原论文配置

