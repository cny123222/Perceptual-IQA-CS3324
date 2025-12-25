#!/usr/bin/env python3
"""
模型复杂度分析脚本
计算 FLOPs、参数量、推理时间和吞吐量
"""

import torch
import torch.nn as nn
import time
import numpy as np
from PIL import Image
import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_iqa.models import smart_iqa as models
from torchvision import transforms


def load_model(checkpoint_path, model_size='base', device='cuda', use_attention=None):
    """加载训练好的模型"""
    print(f"Loading model from: {checkpoint_path}")
    
    # 加载权重以检测是否包含attention
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_hyper' in checkpoint:
        state_dict = checkpoint['model_hyper']
    else:
        state_dict = checkpoint
    
    # 自动检测是否包含attention
    if use_attention is None:
        has_attention = any('multiscale_attention' in key for key in state_dict.keys())
        print(f"Auto-detected attention: {has_attention}")
    else:
        has_attention = use_attention
        print(f"Using manual attention setting: {has_attention}")
    
    # 创建模型
    model = models.HyperNet(
        16, 112, 224, 112, 56, 28, 14, 7,
        use_multiscale=True,
        use_attention=has_attention,
        drop_path_rate=0.3,
        dropout_rate=0.4,
        model_size=model_size
    ).to(device)
    
    # 加载权重
    model.load_state_dict(state_dict)
    
    model.eval()
    print(f"Model loaded successfully (model_size={model_size}, attention={has_attention})")
    return model


def load_image(image_path, device='cuda'):
    """加载并预处理图片"""
    print(f"\nLoading image: {image_path}")
    
    # 加载图片
    img = Image.open(image_path).convert('RGB')
    print(f"Original image size: {img.size}")
    
    # 预处理（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    print(f"Preprocessed tensor shape: {img_tensor.shape}")
    
    return img_tensor, img


def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def compute_flops_ptflops(model, input_size=(3, 224, 224)):
    """使用 ptflops 计算 FLOPs"""
    try:
        from ptflops import get_model_complexity_info
        
        print("\n" + "="*80)
        print("Computing FLOPs using ptflops...")
        print("="*80)
        
        macs, params = get_model_complexity_info(
            model, input_size, 
            as_strings=True,
            print_per_layer_stat=False,  # 设为 True 可以看每层的详细信息
            verbose=False
        )
        
        return macs, params
    except ImportError:
        print("ptflops not installed. Install with: pip install ptflops")
        return None, None
    except Exception as e:
        print(f"Error with ptflops: {e}")
        return None, None


def compute_flops_thop(model, input_tensor):
    """使用 thop 计算 FLOPs"""
    try:
        from thop import profile, clever_format
        
        print("\n" + "="*80)
        print("Computing FLOPs using thop...")
        print("="*80)
        
        # 复制模型到 CPU 以避免 CUDA 问题
        model_cpu = model.cpu()
        input_cpu = input_tensor.cpu()
        
        flops, params = profile(model_cpu, inputs=(input_cpu,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        
        # 移回 GPU
        model.cuda()
        
        return flops, params
    except ImportError:
        print("thop not installed. Install with: pip install thop")
        return None, None
    except Exception as e:
        print(f"Error with thop: {e}")
        return None, None


def measure_inference_time(model, input_tensor, num_warmup=10, num_iterations=100):
    """测量推理时间"""
    print("\n" + "="*80)
    print(f"Measuring inference time (warmup={num_warmup}, iterations={num_iterations})...")
    print("="*80)
    
    device = next(model.parameters()).device
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    # 同步 CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测量时间
    print("Measuring...")
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times)
    }


def measure_throughput(model, input_tensor, batch_sizes=[1, 4, 8, 16, 32], duration=10):
    """测量不同 batch size 的吞吐量"""
    print("\n" + "="*80)
    print(f"Measuring throughput for different batch sizes (duration={duration}s)...")
    print("="*80)
    
    device = next(model.parameters()).device
    results = {}
    
    for bs in batch_sizes:
        try:
            # 创建 batch
            batch_input = input_tensor.repeat(bs, 1, 1, 1)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(batch_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 测量
            num_images = 0
            start_time = time.time()
            
            with torch.no_grad():
                while time.time() - start_time < duration:
                    _ = model(batch_input)
                    num_images += bs
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            throughput = num_images / elapsed_time
            
            results[bs] = {
                'throughput': throughput,
                'num_images': num_images,
                'elapsed_time': elapsed_time
            }
            
            print(f"  Batch size {bs:2d}: {throughput:6.2f} images/sec "
                  f"({num_images} images in {elapsed_time:.2f}s)")
            
        except RuntimeError as e:
            print(f"  Batch size {bs:2d}: Out of memory")
            results[bs] = None
            break
    
    return results


def print_summary(model_name, model_size, total_params, trainable_params, 
                 flops_info, time_stats, throughput_results):
    """打印复杂度分析总结"""
    print("\n" + "="*80)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n📊 Model Information:")
    print(f"  Model Name: {model_name}")
    print(f"  Model Size: {model_size}")
    print(f"  Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    print(f"\n💻 Computational Complexity:")
    if flops_info['ptflops']:
        print(f"  FLOPs (ptflops): {flops_info['ptflops'][0]}")
        print(f"  Params (ptflops): {flops_info['ptflops'][1]}")
    if flops_info['thop']:
        print(f"  FLOPs (thop): {flops_info['thop'][0]}")
        print(f"  Params (thop): {flops_info['thop'][1]}")
    
    print(f"\n⏱️  Inference Time (single image, 224x224):")
    print(f"  Mean: {time_stats['mean']*1000:.2f} ms")
    print(f"  Std:  {time_stats['std']*1000:.2f} ms")
    print(f"  Min:  {time_stats['min']*1000:.2f} ms")
    print(f"  Max:  {time_stats['max']*1000:.2f} ms")
    print(f"  Median: {time_stats['median']*1000:.2f} ms")
    
    print(f"\n🚀 Throughput:")
    for bs, result in throughput_results.items():
        if result:
            print(f"  Batch size {bs:2d}: {result['throughput']:6.2f} images/sec")
    
    print("\n" + "="*80)


def save_results(output_file, model_name, model_size, total_params, trainable_params,
                flops_info, time_stats, throughput_results, device_info):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 模型复杂度分析报告\n\n")
        f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 📊 模型信息\n\n")
        f.write(f"- **模型名称**: {model_name}\n")
        f.write(f"- **模型规模**: {model_size}\n")
        f.write(f"- **总参数量**: {total_params:,} ({total_params/1e6:.2f}M)\n")
        f.write(f"- **可训练参数**: {trainable_params:,} ({trainable_params/1e6:.2f}M)\n")
        f.write(f"- **输入尺寸**: 224×224×3\n")
        f.write(f"- **测试设备**: {device_info}\n\n")
        
        f.write("## 💻 计算复杂度\n\n")
        if flops_info['ptflops']:
            f.write(f"### ptflops 测量结果\n")
            f.write(f"- **FLOPs**: {flops_info['ptflops'][0]}\n")
            f.write(f"- **Parameters**: {flops_info['ptflops'][1]}\n\n")
        
        if flops_info['thop']:
            f.write(f"### thop 测量结果\n")
            f.write(f"- **FLOPs**: {flops_info['thop'][0]}\n")
            f.write(f"- **Parameters**: {flops_info['thop'][1]}\n\n")
        
        f.write("## ⏱️ 推理时间\n\n")
        f.write("**单张图片推理时间** (224×224):\n\n")
        f.write(f"- **平均值**: {time_stats['mean']*1000:.2f} ms\n")
        f.write(f"- **标准差**: {time_stats['std']*1000:.2f} ms\n")
        f.write(f"- **最小值**: {time_stats['min']*1000:.2f} ms\n")
        f.write(f"- **最大值**: {time_stats['max']*1000:.2f} ms\n")
        f.write(f"- **中位数**: {time_stats['median']*1000:.2f} ms\n\n")
        
        f.write("## 🚀 吞吐量\n\n")
        f.write("| Batch Size | 吞吐量 (images/sec) | 测试图片数 | 测试时长 (s) |\n")
        f.write("|-----------|---------------------|-----------|-------------|\n")
        for bs, result in throughput_results.items():
            if result:
                f.write(f"| {bs} | {result['throughput']:.2f} | "
                       f"{result['num_images']} | {result['elapsed_time']:.2f} |\n")
            else:
                f.write(f"| {bs} | OOM | - | - |\n")
        
        f.write("\n## 📝 说明\n\n")
        f.write("- FLOPs (Floating Point Operations): 浮点运算数，衡量计算复杂度\n")
        f.write("- 推理时间：前向传播一次所需的时间\n")
        f.write("- 吞吐量：单位时间内可以处理的图片数量\n")
        f.write("- 测试使用了 10 次 warmup 和 100 次迭代来获得稳定的测量结果\n")
    
    print(f"\n✅ Results saved to: {output_file}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Model Complexity Analysis')
    parser.add_argument('--checkpoint', type=str,
                       default="/root/Perceptual-IQA-CS3324/checkpoints/koniq-10k-swin-ranking-alpha0.5_20251221_155013/best_model_srcc_0.9343_plcc_0.9463.pkl",
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                       default="/root/Perceptual-IQA-CS3324/complexity/example.JPG",
                       help='Path to example image')
    parser.add_argument('--output', type=str,
                       default="/root/Perceptual-IQA-CS3324/complexity/complexity_results.md",
                       help='Output markdown file path')
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['tiny', 'small', 'base'],
                       help='Model size')
    parser.add_argument('--use_attention', action='store_true',
                       help='Use attention fusion (auto-detected if not specified)')
    parser.add_argument('--no_attention', action='store_true',
                       help='Disable attention fusion')
    
    args = parser.parse_args()
    
    # 配置
    checkpoint_path = args.checkpoint
    image_path = args.image
    output_file = args.output
    model_size = args.model_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 确定是否使用attention
    if args.use_attention:
        use_attention = True
    elif args.no_attention:
        use_attention = False
    else:
        use_attention = None  # Auto-detect
    
    print("="*80)
    print("MODEL COMPLEXITY ANALYSIS")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model size: {model_size}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 1. 加载模型
    model = load_model(checkpoint_path, model_size=model_size, device=device, use_attention=use_attention)
    
    # 2. 加载图片
    input_tensor, original_img = load_image(image_path, device=device)
    
    # 3. 统计参数量
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 Parameters:")
    print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 4. 计算 FLOPs
    flops_info = {
        'ptflops': compute_flops_ptflops(model, input_size=(3, 224, 224)),
        'thop': compute_flops_thop(model, input_tensor)
    }
    
    # 5. 测量推理时间
    time_stats = measure_inference_time(model, input_tensor, num_warmup=10, num_iterations=100)
    
    # 6. 测量吞吐量
    throughput_results = measure_throughput(model, input_tensor, 
                                           batch_sizes=[1, 4, 8, 16, 32], 
                                           duration=10)
    
    # 7. 打印总结
    device_info = f"{device}"
    if device == 'cuda':
        device_info += f" ({torch.cuda.get_device_name(0)})"
    
    print_summary(
        model_name="HyperIQA with Swin Transformer",
        model_size=model_size,
        total_params=total_params,
        trainable_params=trainable_params,
        flops_info=flops_info,
        time_stats=time_stats,
        throughput_results=throughput_results
    )
    
    # 8. 保存结果
    save_results(
        output_file=output_file,
        model_name="HyperIQA with Swin Transformer",
        model_size=model_size,
        total_params=total_params,
        trainable_params=trainable_params,
        flops_info=flops_info,
        time_stats=time_stats,
        throughput_results=throughput_results,
        device_info=device_info
    )
    
    print("\n✅ Complexity analysis completed!")


if __name__ == "__main__":
    main()

