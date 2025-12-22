#!/usr/bin/env python3
"""
测试预训练的StairIQA模型（ResNet-50）在四个数据集上的性能
参考StairIQA的测试方法和数据处理方式
"""

import os
import sys
import torch
import torch.nn as nn
from scipy import stats
import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# 添加StairIQA模型路径到最前面，避免和根目录的models.py冲突
stairiqa_models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmarks/StairIQA/models')
sys.path.insert(0, stairiqa_models_path)

# 导入StairIQA的ResNet模型
import ResNet_staircase


def pil_loader(path):
    """加载PIL图像"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class JSONTestDataset(torch.utils.data.Dataset):
    """
    JSON格式测试数据集 - 使用FiveCrop方法（与StairIQA一致）
    优化：预加载和缓存resize后的图像
    """
    def __init__(self, json_path, img_dir, test_method='five'):
        # 加载JSON文件
        with open(json_path) as f:
            data = json.load(f)
        
        # 创建样本列表
        samples = []
        unique_images = {}
        for item in data:
            img_name = os.path.basename(item['image'])
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            score = float(item['score'])
            samples.append((img_path, score))
            unique_images[img_path] = score
        
        self.samples = samples
        print(f'  Total images: {len(self.samples)}')
        
        # 预处理transforms：先resize，crop在getitem中做
        self.resize_transform = transforms.Resize(384)
        
        if test_method == 'one':
            self.crop_transform = transforms.Compose([
                transforms.CenterCrop(320),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        elif test_method == 'five':
            self.crop_transform = transforms.Compose([
                transforms.FiveCrop(320),
                (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                (lambda crops: torch.stack([transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])(crop) for crop in crops]))
            ])
        
        # 关键优化：预加载和缓存所有resize后的图像
        self._resized_cache = {}
        print(f'  Pre-loading and caching {len(unique_images)} images to memory...')
        for img_path in tqdm(unique_images.keys(), desc='  Loading images', unit='img'):
            try:
                img = pil_loader(img_path)
                self._resized_cache[img_path] = self.resize_transform(img)
            except Exception as e:
                print(f"  Warning: Failed to load {img_path}: {e}")
        print(f'  ✓ Cached {len(self._resized_cache)} images in memory')
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        # 从缓存中获取预先resize的图像
        resized_img = self._resized_cache.get(path)
        if resized_img is None:
            # Fallback：如果缓存中没有，重新加载
            img = pil_loader(path)
            resized_img = self.resize_transform(img)
        
        # 只需要做Crop + ToTensor + Normalize（很快）
        img = self.crop_transform(resized_img)
        return img, target
    
    def __len__(self):
        return len(self.samples)


def test_on_dataset(dataset_name, json_path, img_dir, device, model, output_index, 
                   test_method='five'):
    """
    在指定数据集上测试模型 - 使用StairIQA的测试方法
    
    Args:
        dataset_name: 数据集名称
        json_path: JSON标签文件路径
        img_dir: 图像目录
        device: 计算设备
        model: StairIQA模型
        output_index: 输出头索引 (Koniq10k=3, SPAQ=4, etc.)
        test_method: 测试方法 ('one' for CenterCrop, 'five' for FiveCrop)
    """
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"  Warning: {json_path} not found, skipping {dataset_name}")
        return None, None
    
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name}")
    print(f"{'='*60}")
    print(f"  JSON file: {json_path}")
    print(f"  Image dir: {img_dir}")
    print(f"  Test method: {test_method}")
    
    # 创建数据集和DataLoader（Transform在Dataset内部处理，已经预加载图像）
    dataset = JSONTestDataset(json_path, img_dir, test_method)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    
    # 测试
    model.eval()
    pred_scores = []
    gt_scores = []
    
    print(f"  Testing {len(dataset)} images...")
    with torch.no_grad():
        for img, label in tqdm(dataloader, desc=f'  {dataset_name}', unit='img'):
            img = img.to(device)
            label = label.float()
            
            if test_method == 'one':
                # CenterCrop: 单个crop
                outputs_list = model(img)
                pred = outputs_list[output_index]
                score = pred.item()
            elif test_method == 'five':
                # FiveCrop: 5个crops的平均
                bs, ncrops, c, h, w = img.size()
                outputs_list = model(img.view(-1, c, h, w))
                pred = outputs_list[output_index]
                score = pred.view(bs, ncrops, -1).mean(1).item()
            
            pred_scores.append(score)
            gt_scores.append(label.item())
    
    if len(pred_scores) < 2:
        print(f"  Error: Not enough predictions ({len(pred_scores)})")
        return None, None
    
    # 转换为numpy数组
    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)
    
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
    checkpoint_path = '/root/Perceptual-IQA-CS3324/benchmarks/StairIQA/pretrained/ResNet_staircase_50-EXP1-Koniq10k.pkl'
    test_method = 'five'  # 使用FiveCrop（与论文一致）
    
    # 数据集配置
    # StairIQA模型有6个输出头，对应不同的数据集
    # 根据test_staircase.py: Koniq10k使用索引3
    # 注意：KonIQ-10k是训练集，跳过不测试，只测试跨数据集泛化
    datasets = {
        'SPAQ': {
            'json': '/root/Perceptual-IQA-CS3324/spaq-test/spaq_test.json',
            'img_dir': '/root/Perceptual-IQA-CS3324/spaq-test',
            'output_index': 3  # 使用Koniq10k的输出头（泛化测试）
        },
        'KADID-10K': {
            'json': '/root/Perceptual-IQA-CS3324/kadid-test/kadid_test.json',
            'img_dir': '/root/Perceptual-IQA-CS3324/kadid-test',
            'output_index': 3  # 使用Koniq10k的输出头（泛化测试）
        },
        'AGIQA-3K': {
            'json': '/root/Perceptual-IQA-CS3324/agiqa-test/agiqa_test.json',
            'img_dir': '/root/Perceptual-IQA-CS3324/agiqa-test',
            'output_index': 3  # 使用Koniq10k的输出头（泛化测试）
        }
    }
    
    # KonIQ-10k: 使用论文报告的结果（这是训练集）
    # StairIQA论文报告: SRCC=0.906, PLCC=0.921 (在KonIQ-10k上)
    koniq_paper_results = {
        'srcc': 0.906,
        'plcc': 0.921
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
    print(f'\nLoading StairIQA pretrained model: {checkpoint_path}')
    if not os.path.exists(checkpoint_path):
        print(f'Error: Checkpoint not found at {checkpoint_path}')
        return
    
    # 创建模型（与test_staircase.py一致）
    model = ResNet_staircase.resnet50(pretrained=False)
    model = nn.DataParallel(model)  # StairIQA使用DataParallel
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print('Model loaded successfully')
    print(f'Model has DataParallel wrapper: {isinstance(model, nn.DataParallel)}')
    
    # KonIQ-10k使用论文报告的结果（训练集）
    results = {
        'KonIQ-10k': {
            'srcc': koniq_paper_results['srcc'],
            'plcc': koniq_paper_results['plcc'],
            'source': 'paper'
        }
    }
    
    # 在其他数据集上测试（跨数据集泛化）
    for dataset_name, config in datasets.items():
        try:
            srcc, plcc = test_on_dataset(
                dataset_name,
                config['json'],
                config['img_dir'],
                device,
                model,
                config['output_index'],
                test_method
            )
            if srcc is not None:
                results[dataset_name] = {'srcc': srcc, 'plcc': plcc, 'source': 'tested'}
        except Exception as e:
            print(f"\n  Error testing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {'srcc': None, 'plcc': None, 'source': 'error'}
    
    # 打印汇总结果
    print(f"\n\n{'='*80}")
    print(f"STAIRIQA PRETRAINED MODEL CROSS-DATASET TEST RESULTS")
    print(f"{'='*80}")
    print(f"Model:           StairIQA ResNet-50")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Test method:     {test_method} (5 crops average)")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'SRCC':>12} {'PLCC':>12} {'Source':>15}")
    print(f"{'-'*80}")
    for dataset_name in ['KonIQ-10k', 'SPAQ', 'KADID-10K', 'AGIQA-3K']:
        if dataset_name in results and results[dataset_name]['srcc'] is not None:
            srcc = results[dataset_name]['srcc']
            plcc = results[dataset_name]['plcc']
            source = results[dataset_name].get('source', 'tested')
            source_str = 'Paper (train)' if source == 'paper' else 'Cross-dataset'
            print(f"{dataset_name:<20} {srcc:>12.4f} {plcc:>12.4f} {source_str:>15}")
        else:
            print(f"{dataset_name:<20} {'N/A':>12} {'N/A':>12} {'Failed':>15}")
    print(f"{'='*80}")
    
    # 计算平均性能（只计算跨数据集测试的结果，不包括训练集）
    cross_dataset_results = [r for r in results.values() 
                            if r['srcc'] is not None and r.get('source') == 'tested']
    if len(cross_dataset_results) > 0:
        avg_srcc = np.mean([r['srcc'] for r in cross_dataset_results])
        avg_plcc = np.mean([r['plcc'] for r in cross_dataset_results])
        print(f"\nAverage across {len(cross_dataset_results)} cross-dataset tests:")
        print(f"  Average SRCC: {avg_srcc:.4f}")
        print(f"  Average PLCC: {avg_plcc:.4f}")
        print(f"{'='*80}")
    
    # 保存结果
    output_file = 'logs/stairiqa_pretrained_results.txt'
    os.makedirs('logs', exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"STAIRIQA PRETRAINED MODEL CROSS-DATASET TEST RESULTS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Model:           StairIQA ResNet-50\n")
        f.write(f"Checkpoint:      {checkpoint_path}\n")
        f.write(f"Test method:     {test_method} (5 crops average)\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Dataset':<20} {'SRCC':>12} {'PLCC':>12} {'Source':>15}\n")
        f.write(f"{'-'*80}\n")
        for dataset_name in ['KonIQ-10k', 'SPAQ', 'KADID-10K', 'AGIQA-3K']:
            if dataset_name in results and results[dataset_name]['srcc'] is not None:
                srcc = results[dataset_name]['srcc']
                plcc = results[dataset_name]['plcc']
                source = results[dataset_name].get('source', 'tested')
                source_str = 'Paper (train)' if source == 'paper' else 'Cross-dataset'
                f.write(f"{dataset_name:<20} {srcc:>12.4f} {plcc:>12.4f} {source_str:>15}\n")
            else:
                f.write(f"{dataset_name:<20} {'N/A':>12} {'N/A':>12} {'Failed':>15}\n")
        f.write(f"{'='*80}\n")
        
        if len(cross_dataset_results) > 0:
            f.write(f"\nAverage across {len(cross_dataset_results)} cross-dataset tests:\n")
            f.write(f"  Average SRCC: {avg_srcc:.4f}\n")
            f.write(f"  Average PLCC: {avg_plcc:.4f}\n")
        
        f.write(f"\nNotes:\n")
        f.write(f"- KonIQ-10k: Paper reported results (model trained on this dataset)\n")
        f.write(f"- Other datasets: Cross-dataset generalization test\n")
        f.write(f"{'='*80}\n")
    
    print(f"\nResults saved to: {output_file}")
    
    # 同时保存JSON格式
    json_output = 'logs/stairiqa_pretrained_results.json'
    with open(json_output, 'w') as f:
        json.dump({
            'model': 'StairIQA ResNet-50',
            'checkpoint': checkpoint_path,
            'test_method': test_method,
            'results': results
        }, f, indent=2)
    print(f"JSON results saved to: {json_output}")


if __name__ == '__main__':
    main()

