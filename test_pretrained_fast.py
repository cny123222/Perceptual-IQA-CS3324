#!/usr/bin/env python3
"""
快速测试预训练的HyperIQA模型（ResNet-50）
优化：批量处理，减少重复加载
"""

import os
import torch
import models  # Original HyperIQA models (ResNet-50)
from scipy import stats
import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms

def pil_loader(path):
    """加载PIL图像"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def test_on_dataset(dataset_name, json_path, img_dir, device, model_hyper, patch_num):
    """在指定数据集上测试模型"""
    
    # Test transform
    transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225))
    ])
    
    # 加载JSON文件
    if not os.path.exists(json_path):
        print(f"  Warning: {json_path} not found, skipping {dataset_name}")
        return None, None
    
    with open(json_path) as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name}")
    print(f"{'='*60}")
    print(f"  JSON file: {json_path}")
    print(f"  Image dir: {img_dir}")
    print(f"  Test samples: {len(data)} images × {patch_num} patches")
    
    # 测试
    pred_scores = []
    gt_scores = []
    
    model_hyper.eval()
    with torch.no_grad():
        for idx, item in enumerate(data):
            if idx % 100 == 0:
                print(f"  Processing: {idx}/{len(data)} images...")
            
            # 从JSON路径中提取文件名
            img_name = os.path.basename(item['image'])
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            score = float(item['score'])
            
            # 加载图像一次
            img_pil = pil_loader(img_path)
            
            # 对每张图像生成多个patch的预测（批量处理）
            patch_preds = []
            
            # 准备batch
            batch_imgs = []
            for _ in range(patch_num):
                img_tensor = transform(img_pil)
                batch_imgs.append(img_tensor)
            
            # 批量推理（分成小批次以节省内存）
            batch_size = 8
            for i in range(0, patch_num, batch_size):
                batch = torch.stack(batch_imgs[i:i+batch_size]).to(device)
                
                # 生成目标网络权重
                paras = model_hyper(batch)
                
                # 构建目标网络
                model_target = models.TargetNet(paras).to(device)
                model_target.eval()
                
                # 质量预测
                pred = model_target(paras['target_in_vec'])
                
                # 如果是单个样本，pred是标量；如果是batch，pred是向量
                if pred.dim() == 0:
                    patch_preds.append(pred.item())
                else:
                    patch_preds.extend(pred.cpu().numpy().tolist())
            
            # 对多个patch的预测取平均
            avg_pred = np.mean(patch_preds)
            pred_scores.append(avg_pred)
            gt_scores.append(score)
    
    if len(pred_scores) < 2:
        print(f"  Error: Only {len(pred_scores)} valid images found")
        return None, None
    
    # 计算相关系数
    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)
    
    srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    
    print(f"\n  Results:")
    print(f"  Valid images: {len(pred_scores)}")
    print(f"  SRCC: {srcc:.6f}")
    print(f"  PLCC: {plcc:.6f}")
    print(f"{'='*60}")
    
    return srcc, plcc


def main():
    # 配置
    checkpoint_path = '/root/Perceptual-IQA-CS3324/pretrained/koniq_pretrained.pkl'
    patch_num = 25  # 原始HyperIQA的默认值
    
    # 数据集配置
    datasets = {
        'KonIQ-10k': {
            'json': '/root/Perceptual-IQA-CS3324/koniq-10k/koniq_test.json',
            'img_dir': '/root/Perceptual-IQA-CS3324/koniq-test'
        },
        'SPAQ': {
            'json': '/root/Perceptual-IQA-CS3324/spaq-test/spaq_test.json',
            'img_dir': '/root/Perceptual-IQA-CS3324/spaq-test'
        },
        'KADID-10K': {
            'json': '/root/Perceptual-IQA-CS3324/kadid-test/kadid_test.json',
            'img_dir': '/root/Perceptual-IQA-CS3324/kadid-test'
        },
        'AGIQA-3K': {
            'json': '/root/Perceptual-IQA-CS3324/agiqa-test/agiqa_test.json',
            'img_dir': '/root/Perceptual-IQA-CS3324/agiqa-test'
        }
    }
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    print(f'\nLoading pretrained model: {checkpoint_path}')
    if not os.path.exists(checkpoint_path):
        print(f'Error: Checkpoint not found at {checkpoint_path}')
        return
    
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model_hyper.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model_hyper.eval()
    print('Model loaded successfully')
    
    # 在所有数据集上测试
    results = {}
    for dataset_name, config in datasets.items():
        try:
            srcc, plcc = test_on_dataset(
                dataset_name,
                config['json'],
                config['img_dir'],
                device,
                model_hyper,
                patch_num
            )
            if srcc is not None:
                results[dataset_name] = {'srcc': srcc, 'plcc': plcc}
        except Exception as e:
            print(f"\n  Error testing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {'srcc': None, 'plcc': None}
    
    # 打印汇总结果
    print(f"\n\n{'='*80}")
    print(f"PRETRAINED MODEL CROSS-DATASET TEST RESULTS")
    print(f"{'='*80}")
    print(f"Model:           {checkpoint_path}")
    print(f"Test patch num:  {patch_num}")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'SRCC':>12} {'PLCC':>12}")
    print(f"{'-'*80}")
    for dataset_name in ['KonIQ-10k', 'SPAQ', 'KADID-10K', 'AGIQA-3K']:
        if dataset_name in results and results[dataset_name]['srcc'] is not None:
            srcc = results[dataset_name]['srcc']
            plcc = results[dataset_name]['plcc']
            print(f"{dataset_name:<20} {srcc:>12.6f} {plcc:>12.6f}")
        else:
            print(f"{dataset_name:<20} {'N/A':>12} {'N/A':>12}")
    print(f"{'='*80}")
    
    # 保存结果
    output_file = 'logs/pretrained_fast_results.txt'
    os.makedirs('logs', exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"PRETRAINED MODEL CROSS-DATASET TEST RESULTS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Model:           {checkpoint_path}\n")
        f.write(f"Test patch num:  {patch_num}\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Dataset':<20} {'SRCC':>12} {'PLCC':>12}\n")
        f.write(f"{'-'*80}\n")
        for dataset_name in ['KonIQ-10k', 'SPAQ', 'KADID-10K', 'AGIQA-3K']:
            if dataset_name in results and results[dataset_name]['srcc'] is not None:
                srcc = results[dataset_name]['srcc']
                plcc = results[dataset_name]['plcc']
                f.write(f"{dataset_name:<20} {srcc:>12.6f} {plcc:>12.6f}\n")
            else:
                f.write(f"{dataset_name:<20} {'N/A':>12} {'N/A':>12}\n")
        f.write(f"{'='*80}\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()

