# 模型复杂度分析

本目录包含用于分析模型计算复杂度的脚本和工具。

## 📁 文件说明

- `compute_complexity.py` - 完整的复杂度分析脚本（推荐）
- `quick_test.py` - 快速测试脚本（不需要额外依赖）
- `example.JPG` - 测试图片
- `complexity_method.md` - 计算方法参考文档
- `complexity_results.md` - 分析结果报告（运行后生成）

## 🚀 快速开始

### 方法 1：快速测试（推荐先运行）

不需要安装额外依赖，快速得到基本结果：

```bash
cd /root/Perceptual-IQA-CS3324
python complexity/quick_test.py
```

**输出内容**：
- 模型参数量
- 估算的 FLOPs
- 单张图片推理时间
- 吞吐量

### 方法 2：完整分析（需要安装依赖）

安装依赖：

```bash
pip install ptflops thop fvcore
```

运行完整分析：

```bash
cd /root/Perceptual-IQA-CS3324
python complexity/compute_complexity.py
```

**输出内容**：
- 详细的 FLOPs 计算（使用 ptflops 和 thop）
- 参数量统计
- 推理时间统计（平均、标准差、最小、最大、中位数）
- 不同 batch size 的吞吐量测试
- 自动生成 `complexity_results.md` 报告

## 📊 输出示例

### 快速测试输出

```
============================================================
QUICK COMPLEXITY TEST
============================================================

1. Loading model...
   ✅ Model loaded

2. Model Parameters: 88,123,456 (88.12M)

3. Estimated FLOPs: 352.49 GFLOPs

4. Loading test image...
   ✅ Image loaded: (800, 600) -> torch.Size([1, 3, 224, 224])

5. Measuring inference time...

6. Results:
   Average inference time: 45.23 ± 2.15 ms
   Throughput: 22.11 images/sec
   Predicted quality score: 0.7845

============================================================
✅ Quick test completed!
============================================================
```

### 完整分析输出

```
================================================================================
COMPLEXITY ANALYSIS SUMMARY
================================================================================

📊 Model Information:
  Model Name: HyperIQA with Swin Transformer
  Model Size: base
  Total Parameters: 88,123,456 (88.12M)
  Trainable Parameters: 88,123,456 (88.12M)

💻 Computational Complexity:
  FLOPs (ptflops): 352.49G
  Params (ptflops): 88.12M
  FLOPs (thop): 352.47G
  Params (thop): 88.12M

⏱️  Inference Time (single image, 224x224):
  Mean: 45.23 ms
  Std:  2.15 ms
  Min:  42.10 ms
  Max:  51.30 ms
  Median: 44.80 ms

🚀 Throughput:
  Batch size  1:  22.11 images/sec
  Batch size  4:  65.32 images/sec
  Batch size  8: 102.45 images/sec
  Batch size 16: 145.67 images/sec
  Batch size 32: OOM

================================================================================
```

## 🔧 自定义配置

编辑 `compute_complexity.py` 或 `quick_test.py` 中的配置：

```python
# 模型配置
checkpoint_path = "path/to/your/checkpoint.pkl"
model_size = 'base'  # 'tiny', 'small', 'base'

# 测试图片
image_path = "path/to/your/image.jpg"

# 输出文件
output_file = "path/to/output.md"
```

## 📈 测试不同模型

### Swin-Tiny

```python
model = models.HyperNet(
    16, 112, 224, 112, 56, 28, 14, 7,
    use_multiscale=True,
    use_attention=False,
    drop_path_rate=0.2,
    dropout_rate=0.3,
    model_size='tiny'
)
```

预期结果：
- 参数量：~28M
- FLOPs：~120G
- 推理时间：~20ms

### Swin-Small

```python
model = models.HyperNet(
    16, 112, 224, 112, 56, 28, 14, 7,
    use_multiscale=True,
    use_attention=False,
    drop_path_rate=0.2,
    dropout_rate=0.3,
    model_size='small'
)
```

预期结果：
- 参数量：~50M
- FLOPs：~210G
- 推理时间：~30ms

### Swin-Base

```python
model = models.HyperNet(
    16, 112, 224, 112, 56, 28, 14, 7,
    use_multiscale=True,
    use_attention=False,
    drop_path_rate=0.3,
    dropout_rate=0.4,
    model_size='base'
)
```

预期结果：
- 参数量：~88M
- FLOPs：~350G
- 推理时间：~45ms

## 🎯 关键指标说明

### FLOPs (Floating Point Operations)
- 衡量模型的计算复杂度
- 数值越大，计算量越大
- 1 GFLOPs = 10^9 次浮点运算

### 参数量 (Parameters)
- 模型包含的可学习参数总数
- 影响模型大小和内存占用
- 1M = 1,000,000 个参数

### 推理时间 (Inference Time)
- 处理单张图片所需的时间
- 包括前向传播的所有计算
- 通常以毫秒 (ms) 为单位

### 吞吐量 (Throughput)
- 单位时间内可以处理的图片数量
- 以 images/sec 为单位
- 与 batch size 相关

## 📝 注意事项

1. **GPU 性能影响**：推理时间和吞吐量会受 GPU 性能影响
2. **Warmup 重要性**：前几次推理可能较慢，需要 warmup
3. **Batch Size**：增大 batch size 可提高吞吐量，但需要更多显存
4. **图片尺寸**：所有测试使用 224×224 输入尺寸
5. **FLOPs 差异**：不同工具测量的 FLOPs 可能略有差异（通常在 1% 以内）

## 🐛 常见问题

### 1. CUDA out of memory

```bash
# 降低 batch size 或使用 CPU
device = 'cpu'
```

### 2. 依赖安装失败

```bash
# 使用快速测试脚本（不需要额外依赖）
python complexity/quick_test.py
```

### 3. 找不到 checkpoint

```bash
# 检查路径是否正确
ls -lh checkpoints/
```

## 📚 参考文档

- `complexity_method.md` - 包含多种 FLOPs 计算方法
- PyTorch 官方文档：https://pytorch.org/docs/stable/
- ptflops GitHub：https://github.com/sovrasov/flops-counter.pytorch
- thop GitHub：https://github.com/Lyken17/pytorch-OpCounter

