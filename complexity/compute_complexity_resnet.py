#!/usr/bin/env python3
"""
ResNet50-based HyperIQA 复杂度分析
"""

import torch
import torch.nn as nn
import time
import numpy as np
from PIL import Image
import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_iqa.models import hyperiqa as models  # 原始的ResNet-based HyperIQA
from torchvision import transforms

def analyze_resnet_complexity():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建ResNet50-based HyperIQA模型
    print("\nCreating ResNet50-based HyperIQA model...")
    model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model.eval()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 尝试计算FLOPs
    flops_str = "N/A"
    try:
        from ptflops import get_model_complexity_info
        input_size = (3, 224, 224)
        macs, params = get_model_complexity_info(
            model, input_size, 
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        flops_str = f"{macs / 1e9:.2f} GFLOPs"
        print(f"FLOPs: {flops_str}")
    except ImportError:
        print("⚠️  ptflops not installed, skipping FLOPs calculation")
    
    # 测试推理时间
    print("\nMeasuring inference time...")
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # 实际测量
    times = []
    num_iterations = 100
    for _ in range(num_iterations):
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        times.append((end_time - start_time) * 1000)  # ms
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    
    print(f"Mean inference time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {1000/mean_time:.2f} images/sec")
    
    # 测试不同batch size的吞吐量
    print("\nMeasuring throughput for different batch sizes...")
    batch_sizes = [1, 4, 8, 16, 32]
    throughputs = {}
    
    for bs in batch_sizes:
        try:
            input_tensor = torch.randn(bs, 3, 224, 224).to(device)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(input_tensor)
            
            # 测量
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            
            num_batches = max(10, 100 // bs)
            for _ in range(num_batches):
                with torch.no_grad():
                    _ = model(input_tensor)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.time()
            
            total_images = bs * num_batches
            total_time = end_time - start_time
            throughput = total_images / total_time
            
            throughputs[bs] = throughput
            print(f"  Batch size {bs:2d}: {throughput:6.2f} images/sec")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  Batch size {bs:2d}: OOM")
                throughputs[bs] = None
            else:
                raise e
    
    # 生成报告
    report = f"""# 模型复杂度分析报告 - HyperIQA (ResNet50)

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 模型信息

- **模型名称**: HyperIQA (ResNet50 Backbone)
- **模型类型**: CNN-based (ResNet50 + HyperNetwork)
- **总参数量**: {total_params:,} ({total_params/1e6:.2f}M)
- **可训练参数**: {trainable_params:,} ({trainable_params/1e6:.2f}M)
- **输入尺寸**: 224×224×3
- **测试设备**: {device}

## 💻 计算复杂度

- **FLOPs**: {flops_str}
- **Parameters**: {total_params/1e6:.2f}M

## ⏱️ 推理时间

**单张图片推理时间** (224×224):

- **平均值**: {mean_time:.2f} ms
- **标准差**: {std_time:.2f} ms
- **最小值**: {min_time:.2f} ms
- **最大值**: {max_time:.2f} ms
- **中位数**: {median_time:.2f} ms

## 🚀 吞吐量

| Batch Size | 吞吐量 (images/sec) |
|-----------|---------------------|
"""
    
    for bs in batch_sizes:
        if throughputs[bs] is not None:
            report += f"| {bs} | {throughputs[bs]:.2f} |\n"
        else:
            report += f"| {bs} | OOM |\n"
    
    report += """
## 📝 说明

- 本报告分析的是原始HyperIQA模型（使用ResNet50作为backbone）
- FLOPs (Floating Point Operations): 浮点运算数，衡量计算复杂度
- 推理时间：前向传播一次所需的时间
- 吞吐量：单位时间内可以处理的图片数量
- 测试使用了 10 次 warmup 和 100 次迭代来获得稳定的测量结果
"""
    
    # 保存报告
    output_path = '/root/Perceptual-IQA-CS3324/complexity/complexity_results_resnet50.md'
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ 报告已保存: {output_path}")

if __name__ == '__main__':
    analyze_resnet_complexity()
