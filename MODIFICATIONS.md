# 代码修改说明

## 修改前后对比

### 原始代码的问题

**数据划分方式（train_test_IQA.py 原版）：**
```python
# 对所有 10073 条CSV记录随机划分
img_num = {
    'koniq-10k': list(range(0, 10073)),
}
sel_num = img_num[config.dataset]

# 随机80/20划分
random.shuffle(sel_num)
train_index = sel_num[0:int(round(0.8 * len(sel_num)))]  # ~8058条
test_index = sel_num[int(round(0.8 * len(sel_num))):]     # ~2015条
```

**问题：**
- CSV中官方train图片(7046)和test图片(2010)是混合的
- 随机划分会把官方train图片分到测试集，test图片分到训练集
- **导致数据泄露，过拟合严重**

### 修改后的方案

**新的数据划分方式：**
```python
# 1. 读取官方划分
def get_koniq_train_test_indices(root_path):
    # 从 koniq_train.json 获取训练集图片列表
    # 从 koniq_test.json 获取测试集图片列表
    # 从CSV中找到对应的索引
    return train_indices (7046), test_indices (2010)

# 2. 只在训练集内划分
train_indices_copy = train_indices_all.copy()
random.shuffle(train_indices_copy)
split_point = int(round(0.8 * len(train_indices_copy)))
train_index = train_indices_copy[0:split_point]  # 5637张
val_index = train_indices_copy[split_point:]     # 1409张

# 3. 测试集完全独立
test_index = test_indices_all  # 2010张
```

**改进：**
- ✅ 使用官方train/test划分
- ✅ 只在训练集内做train/val划分
- ✅ 测试集完全独立，消除数据泄露
- ✅ 同时显示Val和Test性能

### HyperIQASolver 的修改

**原始：**
```python
def __init__(self, config, path, train_idx, test_idx):
    # 只有train和test两个数据集
    self.train_data = train_loader.get_data()
    self.test_data = test_loader.get_data()
```

**修改后：**
```python
def __init__(self, config, path, train_idx, val_idx, test_idx=None):
    # 支持train/val/test三个数据集
    self.train_data = train_loader.get_data()
    self.val_data = val_loader.get_data()
    if test_idx is not None:
        self.test_data = test_loader.get_data()
```

## 默认参数

原始代码和我们的修改都使用相同的默认参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--epochs` | 16 | 训练轮数 |
| `--train_test_num` | 10 | 重复实验次数 |
| `--batch_size` | 96 | 批大小 |
| `--train_patch_num` | 25 | 每张训练图采样patch数 |
| `--test_patch_num` | 25 | 每张测试图采样patch数 |
| `--lr` | 2e-5 | 学习率 |
| `--weight_decay` | 5e-4 | 权重衰减 |
| `--lr_ratio` | 10 | 超网络学习率倍数 |
| `--patch_size` | 224 | Patch大小 |

## 修改效果

### 修复前（数据泄露）
- Train SRCC: 0.98 (过拟合)
- Test SRCC: 0.89-0.90 (不稳定)
- Val和Test差距大
- Test性能在训练过程中波动下降

### 修复后
- Train SRCC: 0.97
- Val SRCC: 0.90
- Test SRCC: 0.90
- **Val和Test性能非常接近（差距<0.01）**
- Test性能稳定
- 轻微过拟合，可控

## 待论文确认的信息

阅读论文后需要确认：
1. 论文的具体训练参数
2. 论文在KonIQ-10k上报告的SRCC/PLCC
3. 论文是否使用了数据增强
4. 论文的train/val/test划分方式
5. 论文是否使用了早停策略

## 文件修改列表

1. `train_test_IQA.py`
   - 新增 `get_koniq_train_test_indices()` 函数
   - 修改KonIQ-10k的数据划分逻辑
   - 支持train/val/test三分

2. `HyperIQASolver.py`
   - 修改 `__init__()` 支持3个数据集
   - 修改 `train()` 同时显示val和test性能
   - 添加模型保存功能（每2个epoch）

3. 新增训练脚本
   - `train_quick.sh` - 快速测试
   - `train_standard.sh` - 标准训练
   - `train_full.sh` - 完整训练

---

## 训练稳定性修复（实验项：当前不启用）

我们曾尝试过三项“训练稳定性修复”（主要针对 Swin 版本）：
- `filter(...)` 改为 `list(filter(...))`（避免 filter 迭代器被重复使用后耗尽）
- Backbone 学习率也按 epoch 衰减（避免 backbone LR 固定）
- 不在每个 epoch 重建 Adam（保留 optimizer state，只更新 param_groups 学习率）

**结论**：在当前实验设置下（以 KonIQ 测试集 SRCC/PLCC 为主），观察到的最终指标差异不明显，因此目前已回退到原始实现，后续如需对比可再单独启用这三项修复。


