#!/usr/bin/env python3
"""
快速测试 SPAQ 数据集（不训练，直接加载模型测试）
用法：python quick_test_spaq_only.py [checkpoint_path]
"""

import os
import sys
import torch
import json
from PIL import Image
import torchvision
import numpy as np
from scipy import stats
from tqdm import tqdm
import models

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def test_spaq_fast(checkpoint_path, spaq_path, test_patch_num=5, num_images=50, device=None):
    """
    快速测试 SPAQ 数据集
    
    Args:
        checkpoint_path: 模型 checkpoint 路径
        spaq_path: SPAQ 数据集路径
        test_patch_num: 每张图片的 patch 数量（减少以加速）
        device: 计算设备
    """
    if device is None:
        device = get_device()
    
    print(f'Using device: {device}')
    
    # 加载模型
    print(f'\nLoading model from: {checkpoint_path}')
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model_hyper.train(False)
    model_hyper.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print('Model loaded successfully!')
    
    # 加载 SPAQ 数据
    json_path = os.path.join(spaq_path, 'spaq_test.json')
    if not os.path.exists(json_path):
        print(f'Error: SPAQ JSON file not found at {json_path}')
        return None, None
    
    with open(json_path) as f:
        spaq_data = json.load(f)
    
    print(f'\nSPAQ dataset: {len(spaq_data)} images')
    print(f'Using {test_patch_num} patches per image (reduced for speed)')
    
    # Transforms (will be split in Dataset class for optimization)
    # Note: Resize will be done once and cached, RandomCrop+ToTensor+Normalize done each time
    transforms = None  # Not used directly, handled in Dataset class
    
    # 创建 samples（只取少量图片快速测试）
    samples = []
    test_images = min(num_images, len(spaq_data))  # 只测试指定数量的图片
    print(f'Testing on first {test_images} images for speed...')
    
    for item in spaq_data[:test_images]:
        img_path = os.path.join(spaq_path, os.path.basename(item['image']))
        if not os.path.exists(img_path):
            continue
        score = float(item['score'])
        for _ in range(test_patch_num):
            samples.append((img_path, score))
    
    if len(samples) == 0:
        print('Error: No valid samples found!')
        return None, None
    
    # Dataset with optimized caching: pre-resize images to avoid slow resize operations
    class SPAQDataset(torch.utils.data.Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            # Split transform: Resize is expensive, so we'll do it once and cache
            # The rest (RandomCrop, ToTensor, Normalize) we do each time
            self.resize_transform = torchvision.transforms.Resize((512, 384))
            self.crop_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
            
            # Cache resized images (much faster than resizing each time)
            self._resized_cache = {}
            # Keep track of unique images
            self._unique_paths = list(set([s[0] for s in samples]))
            print(f'  Pre-loading and resizing {len(self._unique_paths)} unique images...')
            
            # Pre-load and resize all images (this is the expensive operation)
            for path in tqdm(self._unique_paths, desc='  Pre-loading', leave=False):
                if os.path.exists(path):
                    img = pil_loader(path)
                    # Resize once and cache (this is the slow part for large images)
                    resized_img = self.resize_transform(img)
                    self._resized_cache[path] = resized_img
        
        def __getitem__(self, index):
            path, target = self.samples[index]
            # Get pre-resized image from cache
            resized_img = self._resized_cache.get(path)
            if resized_img is None:
                # Fallback: load and resize on the fly
                img = pil_loader(path)
                resized_img = self.resize_transform(img)
                self._resized_cache[path] = resized_img
            
            # Now only do RandomCrop + ToTensor + Normalize (fast operations)
            sample = self.crop_transform(resized_img)
            return sample, target
        
        def __len__(self):
            return len(self.samples)
    
    # DataLoader
    spaq_dataset = SPAQDataset(samples, transforms)
    spaq_loader = torch.utils.data.DataLoader(
        spaq_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
    # 测试
    model_hyper.eval()
    pred_scores = []
    gt_scores = []
    
    print(f'\nTesting...')
    with torch.no_grad():
        for img, label in tqdm(spaq_loader, desc='  SPAQ', unit='batch'):
            img = img.to(device)
            label = label.float().to(device)
            
            paras = model_hyper(img)
            model_target = models.TargetNet(paras).to(device)
            model_target.eval()
            pred = model_target(paras['target_in_vec'])
            
            pred_scores.append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()
    
    # 计算指标
    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, test_patch_num)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, test_patch_num)), axis=1)
    
    spaq_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    spaq_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    
    print(f'\n{"="*60}')
    print(f'SPAQ Test Results (first {test_images} images, {test_patch_num} patches/image):')
    print(f'{"="*60}')
    print(f'  SRCC: {spaq_srcc:.4f}')
    print(f'  PLCC: {spaq_plcc:.4f}')
    print(f'{"="*60}')
    
    return spaq_srcc, spaq_plcc

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Quick test SPAQ dataset without training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (if not provided, will look for latest)')
    parser.add_argument('--spaq_path', type=str, default='./spaq-test',
                        help='Path to SPAQ dataset')
    parser.add_argument('--test_patch_num', type=int, default=5,
                        help='Number of patches per image (default: 5 for speed)')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to test (default: 100 for speed)')
    
    args = parser.parse_args()
    
    # 如果没有指定 checkpoint，尝试找最新的
    if args.checkpoint is None:
        checkpoints_dir = './checkpoints'
        if os.path.exists(checkpoints_dir):
            # 找最新的 checkpoint
            all_checkpoints = []
            for root, dirs, files in os.walk(checkpoints_dir):
                for file in files:
                    if file.endswith('.pkl'):
                        all_checkpoints.append(os.path.join(root, file))
            
            if all_checkpoints:
                # 按修改时间排序
                all_checkpoints.sort(key=os.path.getmtime, reverse=True)
                args.checkpoint = all_checkpoints[0]
                print(f'Using latest checkpoint: {args.checkpoint}')
            else:
                print('Error: No checkpoint found. Please specify --checkpoint')
                sys.exit(1)
        else:
            print('Error: Checkpoints directory not found. Please specify --checkpoint')
            sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f'Error: Checkpoint not found: {args.checkpoint}')
        sys.exit(1)
    
    test_spaq_fast(args.checkpoint, args.spaq_path, args.test_patch_num, args.num_images)

if __name__ == '__main__':
    main()

