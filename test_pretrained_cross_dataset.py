#!/usr/bin/env python3
"""
测试预训练的HyperIQA模型在跨数据集上的表现
Test pretrained HyperIQA (ResNet-50) on cross-datasets
"""

import os
import torch
import models  # Original HyperIQA models (ResNet-50)
from scipy import stats
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

def pil_loader(path):
    """加载PIL图像"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def test_on_dataset(dataset_name, dataset_path, json_file, device, model_hyper, patch_num):
    """在指定数据集上测试模型"""
    
    import torchvision
    
    # Test transform (CenterCrop for reproducibility)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 384)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                       std=(0.229, 0.224, 0.225))
    ])
    
    # Load test data from JSON
    json_path = os.path.join(dataset_path, json_file)
    if not os.path.exists(json_path):
        print(f"  Warning: {json_file} not found, skipping {dataset_name}")
        return None, None
    
    with open(json_path) as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name}")
    print(f"{'='*60}")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Test samples: {len(data)} images × {patch_num} patches")
    
    # Test
    pred_scores_per_img = {}
    gt_scores_per_img = {}
    
    model_hyper.eval()
    with torch.no_grad():
        for item in tqdm(data, desc=f"  {dataset_name}", unit="img"):
            img_name = os.path.basename(item['image'])
            img_path = os.path.join(dataset_path, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            score = float(item['score'])
            
            # Process multiple patches per image
            patch_preds = []
            for _ in range(patch_num):
                img = pil_loader(img_path)
                img = transform(img).unsqueeze(0).to(device)
                
                # Generate weights for target network
                paras = model_hyper(img)
                
                # Building target network
                model_target = models.TargetNet(paras).to(device)
                for param in model_target.parameters():
                    param.requires_grad = False
                
                # Quality prediction
                pred = model_target(paras['target_in_vec'])
                patch_preds.append(pred.item())
            
            # Average predictions over patches
            avg_pred = np.mean(patch_preds)
            pred_scores_per_img[img_name] = avg_pred
            gt_scores_per_img[img_name] = score
    
    # Calculate metrics
    pred_scores = np.array([pred_scores_per_img[k] for k in sorted(pred_scores_per_img.keys())])
    gt_scores = np.array([gt_scores_per_img[k] for k in sorted(gt_scores_per_img.keys())])
    
    srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    
    print(f"\n  Results:")
    print(f"  SRCC: {srcc:.6f}")
    print(f"  PLCC: {plcc:.6f}")
    print(f"{'='*60}")
    
    return srcc, plcc


def main():
    # Configuration
    checkpoint_path = '/root/Perceptual-IQA-CS3324/pretrained/koniq_pretrained.pkl'
    patch_num = 20
    
    # Dataset paths
    datasets = {
        'KonIQ-10k': {
            'path': '/root/Perceptual-IQA-CS3324/koniq-10k',
            'json': 'koniq_test.json'
        },
        'SPAQ': {
            'path': '/root/Perceptual-IQA-CS3324/SPAQ',
            'json': 'spaq_test.json'
        },
        'KADID-10K': {
            'path': '/root/Perceptual-IQA-CS3324/KADID-10K',
            'json': 'kadid_test.json'
        },
        'AGIQA-3K': {
            'path': '/root/Perceptual-IQA-CS3324/AGIQA-3K',
            'json': 'agiqa_test.json'
        }
    }
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'\nLoading pretrained model: {checkpoint_path}')
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model_hyper.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model_hyper.eval()
    print('Model loaded successfully')
    
    # Test on all datasets
    results = {}
    for dataset_name, config in datasets.items():
        try:
            srcc, plcc = test_on_dataset(
                dataset_name,
                config['path'],
                config['json'],
                device,
                model_hyper,
                patch_num
            )
            if srcc is not None:
                results[dataset_name] = {'srcc': srcc, 'plcc': plcc}
        except Exception as e:
            print(f"\n  Error testing {dataset_name}: {e}")
            results[dataset_name] = {'srcc': None, 'plcc': None}
    
    # Print summary
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
    
    # Save results
    output_file = 'logs/pretrained_cross_dataset_results.txt'
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

