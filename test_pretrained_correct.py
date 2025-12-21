#!/usr/bin/env python3
"""
测试预训练的HyperIQA模型（ResNet-50）- 使用与源代码完全相同的方式
"""

import os
import torch
import models
from scipy import stats
import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

def pil_loader(path):
    """加载PIL图像"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class JSONTestDataset(torch.utils.data.Dataset):
    """
    JSON格式测试数据集 - 与原始HyperIQA完全相同的方式
    每张图像重复patch_num次
    """
    def __init__(self, json_path, img_dir, patch_num, transform):
        self.transform = transform
        
        # 加载JSON文件
        with open(json_path) as f:
            data = json.load(f)
        
        # 创建样本列表：每张图像重复patch_num次（与folders.py的Koniq_10kFolder相同）
        samples = []
        for item in data:
            img_name = os.path.basename(item['image'])
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            score = float(item['score'])
            # 每张图像重复patch_num次
            for _ in range(patch_num):
                samples.append((img_path, score))
        
        self.samples = samples
        print(f'  Total samples: {len(self.samples)} (images × patch_num)')
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.samples)


def test_on_dataset(dataset_name, json_path, img_dir, device, model_hyper, patch_num):
    """
    在指定数据集上测试模型 - 与HyerIQASolver.test()完全相同
    """
    
    # Test transform - 与data_loader.py中koniq-10k的test transform完全相同
    transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.RandomCrop(224),  # 原始HyperIQA用RandomCrop
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225))
    ])
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"  Warning: {json_path} not found, skipping {dataset_name}")
        return None, None
    
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name}")
    print(f"{'='*60}")
    print(f"  JSON file: {json_path}")
    print(f"  Image dir: {img_dir}")
    
    # 创建数据集和DataLoader（与HyerIQASolver完全相同）
    dataset = JSONTestDataset(json_path, img_dir, patch_num, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )
    
    # 测试（与HyerIQASolver.test()完全相同）
    model_hyper.eval()
    pred_scores = []
    gt_scores = []
    
    print(f"  Testing {len(dataset)} samples...")
    with torch.no_grad():
        for img, label in tqdm(dataloader, desc=f'  {dataset_name}', unit='batch'):
            img = img.to(device)
            label = label.float().to(device)
            
            # 生成目标网络权重
            paras = model_hyper(img)
            
            # 构建目标网络
            model_target = models.TargetNet(paras).to(device)
            model_target.eval()
            
            # 质量预测
            pred = model_target(paras['target_in_vec'])
            
            pred_scores.append(float(pred.item()))
            gt_scores.extend(label.cpu().tolist())
    
    if len(pred_scores) < patch_num * 2:
        print(f"  Error: Not enough predictions ({len(pred_scores)})")
        return None, None
    
    # 对每张图像的多个patch取平均（与HyerIQASolver.test()完全相同）
    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, patch_num)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, patch_num)), axis=1)
    
    # 计算相关系数
    test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    
    print(f"\n  Results:")
    print(f"  Valid images: {len(pred_scores)}")
    print(f"  SRCC: {test_srcc:.6f}")
    print(f"  PLCC: {test_plcc:.6f}")
    print(f"{'='*60}")
    
    return test_srcc, test_plcc


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
    output_file = 'logs/pretrained_correct_results.txt'
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

