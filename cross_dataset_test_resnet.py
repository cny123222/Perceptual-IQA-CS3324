#!/usr/bin/env python3
"""
ResNet-50 Baseline跨数据集测试脚本
Cross-Dataset Testing Script for ResNet-50 HyperIQA Model

测试训练好的ResNet-50模型在四个数据集上的表现：
- KonIQ-10k (in-domain test set)
- SPAQ (cross-domain)
- KADID-10K (cross-domain)
- AGIQA-3K (cross-domain)

Usage:
    python cross_dataset_test_resnet.py --checkpoint path/to/model.pkl --test_patch_num 20
"""

import os
import sys
import argparse
import torch
import models
from scipy import stats
import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
from tqdm import tqdm


def pil_loader(path):
    """加载PIL图像"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class JSONTestDataset(torch.utils.data.Dataset):
    """
    JSON格式测试数据集 - 与原始HyperIQA完全相同的方式
    预加载和缓存resize后的图像以加速测试
    """
    def __init__(self, json_path, img_dir, patch_num, test_random_crop=True):
        # 加载JSON文件
        with open(json_path) as f:
            data = json.load(f)
        
        # 创建样本列表：每张图像重复patch_num次
        samples = []
        unique_imgs = {}
        for item in data:
            img_name = os.path.basename(item['image'])
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            score = float(item['score'])
            unique_imgs[img_path] = score
            # 每张图像重复patch_num次
            for _ in range(patch_num):
                samples.append((img_path, score))
        
        self.samples = samples
        unique_paths = list(unique_imgs.keys())
        print(f"  Found {len(unique_paths)} images, {len(self.samples)} total patches")
        
        # 预加载和缓存resize后的图像
        print(f"  Pre-loading and caching {len(unique_paths)} images...")
        self.resize_transform = transforms.Resize((512, 384))
        
        if test_random_crop:
            self.crop_transform = transforms.RandomCrop(size=384)
        else:
            self.crop_transform = transforms.CenterCrop(size=384)
        
        self._resized_cache = {}
        for img_path in tqdm(unique_paths, desc="  Loading images", unit="img"):
            try:
                img = pil_loader(img_path)
                img_resized = self.resize_transform(img)
                self._resized_cache[img_path] = img_resized
            except Exception as e:
                print(f"  Warning: Failed to load {img_path}: {e}")
        
        # 最终transform
        self.final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, score = self.samples[idx]
        
        # 从缓存中获取resize后的图像
        if img_path in self._resized_cache:
            img_resized = self._resized_cache[img_path]
        else:
            img = pil_loader(img_path)
            img_resized = self.resize_transform(img)
        
        # 应用crop和normalize
        img_cropped = self.crop_transform(img_resized)
        img_tensor = self.final_transform(img_cropped)
        
        return img_tensor, score


def load_model(checkpoint_path, device):
    """加载ResNet-50 HyperIQA模型"""
    print(f"Loading model from: {checkpoint_path}")
    
    # 创建模型
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model_hyper.train(False)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_hyper.load_state_dict(checkpoint['model_hyper'])
    
    print(f"  Model loaded successfully!")
    print(f"  Using device: {device}")
    
    return model_hyper


def test_on_dataset(dataset_name, json_path, img_dir, device, model_hyper, patch_num, test_random_crop):
    """在单个数据集上测试"""
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"  Warning: {json_path} not found, skipping {dataset_name}")
        return None, None
    
    if not os.path.exists(img_dir):
        print(f"  Warning: {img_dir} not found, skipping {dataset_name}")
        return None, None
    
    print(f"\n{'='*80}")
    print(f"Testing on {dataset_name}")
    print(f"{'='*80}")
    
    # 创建数据集和数据加载器
    dataset = JSONTestDataset(json_path, img_dir, patch_num, test_random_crop)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )
    
    # 测试
    pred_scores = []
    gt_scores = []
    
    with torch.no_grad():
        for img, label in tqdm(dataloader, desc=f'  {dataset_name}', unit='batch'):
            img = img.to(device)
            
            # 前向传播
            paras = model_hyper(img)
            
            # 使用TargetNet计算预测分数（使用HyperNet生成的参数）
            model_target = models.TargetNet(paras).to(device)
            pred = model_target(paras['target_in_vec'])
            
            pred_scores.append(float(pred.item()))
            gt_scores.append(float(label.item()))
    
    # 计算SRCC和PLCC
    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)
    
    srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    
    print(f"  SRCC: {srcc:.6f}, PLCC: {plcc:.6f}")
    
    return srcc, plcc


def main():
    parser = argparse.ArgumentParser(description='Cross-dataset testing for ResNet-50 HyperIQA')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint (.pkl)')
    parser.add_argument('--test_patch_num', type=int, default=20,
                        help='Number of patches per image (default: 20)')
    parser.add_argument('--test_random_crop', action='store_true', default=True,
                        help='Use random crop for testing (default: True)')
    parser.add_argument('--test_center_crop', dest='test_random_crop', action='store_false',
                        help='Use center crop for testing')
    
    args = parser.parse_args()
    
    # 检查checkpoint是否存在
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    # 打印配置
    print(f"\n{'='*80}")
    print(f"ResNet-50 BASELINE CROSS-DATASET TEST")
    print(f"{'='*80}")
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Test Patch Num:      {args.test_patch_num}")
    print(f"Test Crop Method:    {'RandomCrop' if args.test_random_crop else 'CenterCrop'}")
    print(f"Device:              {device}")
    print(f"{'='*80}\n")
    
    # 加载模型
    model_hyper = load_model(args.checkpoint, device)
    
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
                args.test_patch_num,
                args.test_random_crop
            )
            if srcc is not None:
                results[dataset_name] = {'srcc': srcc, 'plcc': plcc}
        except Exception as e:
            print(f"\n  Error testing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {'srcc': None, 'plcc': None}
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print(f"RESNET-50 BASELINE CROSS-DATASET TEST RESULTS")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Patch Num: {args.test_patch_num}")
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
    print(f"{'='*80}\n")
    
    # 保存结果到日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/resnet50_baseline_cross_dataset_test_{timestamp}.log'
    
    with open(log_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"RESNET-50 BASELINE CROSS-DATASET TEST RESULTS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test Patch Num: {args.test_patch_num}\n")
        f.write(f"Test Crop Method: {'RandomCrop' if args.test_random_crop else 'CenterCrop'}\n")
        f.write(f"Timestamp: {timestamp}\n")
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
    
    print(f"Results saved to: {log_file}")
    
    # 保存JSON格式的结果
    json_file = f'cross_dataset_results_resnet50_baseline_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'test_patch_num': args.test_patch_num,
            'test_random_crop': args.test_random_crop,
            'timestamp': timestamp,
            'results': results
        }, f, indent=2)
    
    print(f"JSON results saved to: {json_file}\n")


if __name__ == '__main__':
    main()

