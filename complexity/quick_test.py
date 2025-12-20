#!/usr/bin/env python3
"""
快速复杂度测试脚本 - 只测试基本的 FLOPs 和推理时间
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models_swin as models
from PIL import Image
from torchvision import transforms


def simple_flops_estimate(model, input_size=(1, 3, 224, 224)):
    """
    简单估算 FLOPs（不需要额外库）
    基于 Swin Transformer 的典型 FLOPs 值
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # 基于经验值的估算：
    # Swin-Tiny (~28M params): ~4.5 GFLOPs
    # Swin-Small (~50M params): ~8.7 GFLOPs  
    # Swin-Base (~88M params): ~15.4 GFLOPs
    # 考虑到 HyperNet 额外的计算，我们使用 200 的系数
    estimated_flops = total_params * 200
    
    return estimated_flops, total_params


def quick_test():
    checkpoint_path = "/root/Perceptual-IQA-CS3324/checkpoints/koniq-10k-swin-ranking-alpha0.5_20251220_091014/best_model_srcc_0.9336_plcc_0.9464.pkl"
    image_path = "/root/Perceptual-IQA-CS3324/complexity/example.JPG"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("QUICK COMPLEXITY TEST")
    print("="*60)
    
    # 加载模型
    print("\n1. Loading model...")
    model = models.HyperNet(
        16, 112, 224, 112, 56, 28, 14, 7,
        use_multiscale=True,
        use_attention=False,
        drop_path_rate=0.3,
        dropout_rate=0.4,
        model_size='base'
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_hyper' in checkpoint:
        model.load_state_dict(checkpoint['model_hyper'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("   ✅ Model loaded")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n2. Model Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 估算 FLOPs
    estimated_flops, _ = simple_flops_estimate(model)
    print(f"\n3. Estimated FLOPs: {estimated_flops/1e9:.2f} GFLOPs")
    
    # 加载图片
    print(f"\n4. Loading test image...")
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    print(f"   ✅ Image loaded: {img.size} -> {input_tensor.shape}")
    
    # 测试推理时间
    print(f"\n5. Measuring inference time...")
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 测量
    times = []
    with torch.no_grad():
        for _ in range(50):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            output = model(input_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.time() - start)
    
    import numpy as np
    times = np.array(times)
    
    print(f"\n6. Results:")
    print(f"   Average inference time: {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")
    print(f"   Throughput: {1/np.mean(times):.2f} images/sec")
    
    # 预测分数
    if 'target_quality' in output:
        predicted_score = output['target_quality'].item()
        print(f"   Predicted quality score: {predicted_score:.4f}")
    
    print("\n" + "="*60)
    print("✅ Quick test completed!")
    print("="*60)


if __name__ == "__main__":
    quick_test()

