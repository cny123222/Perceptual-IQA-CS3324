#!/usr/bin/env python3
"""
批量运行所有模型的复杂度分析
包括：HyperIQA (ResNet50), SMART-Tiny, SMART-Small, SMART-Base
"""

import os
import sys
import subprocess
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 定义要测试的模型配置
models_to_test = [
    {
        'name': 'HyperIQA (ResNet50)',
        'type': 'resnet',
        'checkpoint': None,  # ResNet不需要checkpoint，直接创建模型
        'model_size': None,
        'output_file': 'complexity_results_resnet50.md'
    },
    {
        'name': 'SMART-Tiny',
        'type': 'swin',
        'checkpoint': None,  # 需要用户指定或自动查找
        'model_size': 'tiny',
        'output_file': 'complexity_results_swin_tiny.md'
    },
    {
        'name': 'SMART-Small',
        'type': 'swin',
        'checkpoint': None,  # 需要用户指定或自动查找
        'model_size': 'small',
        'output_file': 'complexity_results_swin_small.md'
    },
    {
        'name': 'SMART-Base',
        'type': 'swin',
        'checkpoint': '/root/Perceptual-IQA-CS3324/checkpoints/koniq-10k-swin_20251223_002226/best_model_srcc_0.9378_plcc_0.9485.pkl',
        'model_size': 'base',
        'output_file': 'complexity_results_swin_base.md'
    }
]

def find_best_checkpoint(pattern):
    """查找最佳checkpoint文件"""
    import glob
    checkpoints_dir = '/root/Perceptual-IQA-CS3324/checkpoints'
    
    # 搜索匹配的checkpoint文件
    pattern_path = os.path.join(checkpoints_dir, pattern)
    matches = glob.glob(pattern_path)
    
    if not matches:
        return None
    
    # 返回最新的checkpoint
    return max(matches, key=os.path.getmtime)

def run_resnet_complexity():
    """运行ResNet模型的复杂度分析"""
    print("\n" + "="*80)
    print("分析 HyperIQA (ResNet50) 复杂度")
    print("="*80)
    
    # 创建专门的ResNet复杂度分析脚本
    script_path = '/root/Perceptual-IQA-CS3324/complexity/compute_complexity_resnet.py'
    
    if not os.path.exists(script_path):
        print(f"创建 ResNet 复杂度分析脚本: {script_path}")
        create_resnet_script(script_path)
    
    # 运行脚本
    cmd = f"cd /root/Perceptual-IQA-CS3324 && python {script_path}"
    subprocess.run(cmd, shell=True)

def run_swin_complexity(model_config):
    """运行Swin Transformer模型的复杂度分析"""
    print("\n" + "="*80)
    print(f"分析 {model_config['name']} 复杂度")
    print("="*80)
    
    checkpoint = model_config['checkpoint']
    model_size = model_config['model_size']
    output_file = model_config['output_file']
    
    if checkpoint is None:
        print(f"⚠️  未找到 {model_config['name']} 的 checkpoint")
        print(f"   请手动指定 checkpoint 路径或训练模型")
        
        # 尝试创建无checkpoint的分析（仅参数量和FLOPs）
        create_no_checkpoint_analysis(model_size, output_file)
        return
    
    if not os.path.exists(checkpoint):
        print(f"⚠️  Checkpoint 不存在: {checkpoint}")
        return
    
    # 运行复杂度分析
    cmd = f"""cd /root/Perceptual-IQA-CS3324 && \
python complexity/compute_complexity.py \
--checkpoint {checkpoint} \
--model-size {model_size} \
--output complexity/{output_file} \
--image complexity/example.JPG"""
    
    subprocess.run(cmd, shell=True)

def create_no_checkpoint_analysis(model_size, output_file):
    """为没有checkpoint的模型创建基本分析（参数量和FLOPs）"""
    print(f"创建无checkpoint的基本分析: {model_size}")
    
    import models_swin as models
    import time
    from datetime import datetime
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型（不加载权重）
    if model_size == 'tiny':
        drop_path = 0.2
        dropout = 0.3
    elif model_size == 'small':
        drop_path = 0.2
        dropout = 0.3
    else:  # base
        drop_path = 0.3
        dropout = 0.4
    
    model = models.HyperNet(
        16, 112, 224, 112, 56, 28, 14, 7,
        use_multiscale=True,
        use_attention=True,
        drop_path_rate=drop_path,
        dropout_rate=dropout,
        model_size=model_size
    ).to(device)
    
    model.eval()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 尝试计算FLOPs
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
    except:
        flops_str = "N/A (需要安装 ptflops)"
    
    # 生成报告
    report = f"""# 模型复杂度分析报告 (无checkpoint)

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 模型信息

- **模型名称**: SMART-IQA (Swin-{model_size.capitalize()})
- **模型规模**: {model_size}
- **总参数量**: {total_params:,} ({total_params/1e6:.2f}M)
- **可训练参数**: {trainable_params:,} ({trainable_params/1e6:.2f}M)
- **输入尺寸**: 224×224×3
- **测试设备**: {device}

## 💻 计算复杂度

- **FLOPs (估算)**: {flops_str}
- **Parameters**: {total_params/1e6:.2f}M

## ⚠️ 说明

本报告是基于模型架构的理论分析，未加载训练权重。
实际推理时间和吞吐量需要加载训练好的checkpoint才能测量。

要获取完整的复杂度分析（包括推理时间和吞吐量），请：
1. 训练该模型大小的checkpoint
2. 使用 compute_complexity.py 脚本进行完整分析
"""
    
    # 保存报告
    output_path = f'/root/Perceptual-IQA-CS3324/complexity/{output_file}'
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✅ 报告已保存: {output_path}")
    print(f"   参数量: {total_params/1e6:.2f}M")
    print(f"   FLOPs: {flops_str}")

def create_resnet_script(script_path):
    """创建ResNet复杂度分析脚本"""
    script_content = '''#!/usr/bin/env python3
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
    print("\\nCreating ResNet50-based HyperIQA model...")
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
    print("\\nMeasuring inference time...")
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
    print("\\nMeasuring throughput for different batch sizes...")
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
            report += f"| {bs} | {throughputs[bs]:.2f} |\\n"
        else:
            report += f"| {bs} | OOM |\\n"
    
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
    
    print(f"\\n✅ 报告已保存: {output_path}")

if __name__ == '__main__':
    analyze_resnet_complexity()
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)

def main():
    print("="*80)
    print("批量运行所有模型的复杂度分析")
    print("="*80)
    
    # 1. 运行ResNet分析
    run_resnet_complexity()
    
    # 2. 运行Swin Transformer分析
    for model_config in models_to_test:
        if model_config['type'] == 'swin':
            run_swin_complexity(model_config)
    
    print("\n" + "="*80)
    print("✅ 所有模型复杂度分析完成！")
    print("="*80)
    print("\n查看结果:")
    print("  - complexity/complexity_results_resnet50.md")
    print("  - complexity/complexity_results_swin_tiny.md")
    print("  - complexity/complexity_results_swin_small.md")
    print("  - complexity/complexity_results_swin_base.md")

if __name__ == '__main__':
    main()


