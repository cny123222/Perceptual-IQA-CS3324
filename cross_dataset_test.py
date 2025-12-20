#!/usr/bin/env python3
"""
跨数据集测试脚本 - Cross-Dataset Testing Script

用于在多个IQA数据集上评估训练好的模型
Supports: KonIQ-10k, SPAQ, KADID-10K, AGIQA-3K

Usage:
    python cross_dataset_test.py --checkpoint path/to/model.pkl --model_size tiny
    python cross_dataset_test.py --checkpoint path/to/model.pkl --model_size small --test_patch_num 20
"""

import os
import argparse
import torch
import torchvision
import models_swin as models
from scipy import stats
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import glob


def get_device():
    """自动检测可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def pil_loader(path):
    """加载PIL图像"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class JSONTestDataset(torch.utils.data.Dataset):
    """
    通用JSON格式测试数据集加载器
    与训练时的测试方法完全一致
    """
    def __init__(self, root, json_file, transform, patch_num):
        self.root = root
        self.transform = transform
        self.patch_num = patch_num
        
        # 加载JSON文件
        json_path = os.path.join(root, json_file)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
            
        with open(json_path) as f:
            data = json.load(f)
        
        # 生成样本列表：每个图像生成patch_num个样本
        self.samples = []
        for item in data:
            img_path = os.path.join(root, os.path.basename(item['image']))
            if not os.path.exists(img_path):
                print(f"  Warning: Image not found: {img_path}")
                continue
            score = float(item['score'])
            for _ in range(patch_num):
                self.samples.append((img_path, score))
        
        # 计算实际图像数量
        unique_images = len(set([s[0] for s in self.samples]))
        print(f"  Loaded {unique_images} images, {len(self.samples)} total patches")
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __len__(self):
        return len(self.samples)


def test_on_dataset(dataset_name, dataset_path, json_file, device, model_hyper, 
                   patch_num, dropout_rate, test_random_crop=False):
    """
    在指定数据集上测试模型
    
    Args:
        dataset_name: 数据集名称
        dataset_path: 数据集路径
        json_file: JSON文件名
        device: 计算设备
        model_hyper: HyperNet模型
        patch_num: 每张图像的patch数量
        dropout_rate: Dropout率
        test_random_crop: 是否使用RandomCrop（默认False，使用CenterCrop以保证可复现性）
    
    Returns:
        srcc, plcc, num_images
    """
    print(f'\n{"="*60}')
    print(f'Testing on {dataset_name.upper()} dataset')
    print(f'{"="*60}')
    print(f'Dataset path: {dataset_path}')
    print(f'JSON file: {json_file}')
    print(f'Patch number: {patch_num}')
    print(f'Crop method: {"RandomCrop" if test_random_crop else "CenterCrop"}')
    
    # 构建transform（与训练时的测试transform完全一致）
    # 参考 data_loader.py 中 koniq-10k 的测试transform
    if test_random_crop:
        # RandomCrop（原论文方法，但结果不可复现）
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                           std=(0.229, 0.224, 0.225))])
    else:
        # CenterCrop（推荐，结果可复现）
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                           std=(0.229, 0.224, 0.225))])
    
    # 加载数据集
    try:
        dataset = JSONTestDataset(dataset_path, json_file, transforms, patch_num)
    except FileNotFoundError as e:
        print(f"  ❌ Error: {e}")
        return None, None, 0
    
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
    # 测试（与 HyperIQASolver_swin.py 中的 test() 方法完全一致）
    model_hyper.train(False)
    pred_scores = []
    gt_scores = []
    
    print(f'  Total batches: {len(test_loader)}')
    
    with torch.no_grad():  # 禁用梯度计算以加速推理
        test_loader_with_progress = tqdm(
            test_loader,
            desc=f'  Testing {dataset_name.upper()}',
            total=len(test_loader),
            unit='batch',
            mininterval=1.0
        )
        
        for img, label in test_loader_with_progress:
            img = img.to(device)
            label = label.float().to(device)
            
            # 生成目标网络权重
            paras = model_hyper(img)
            
            # 构建目标网络
            model_target = models.TargetNet(paras, dropout_rate=dropout_rate).to(device)
            model_target.train(False)
            
            # 质量预测
            pred = model_target(paras['target_in_vec'])
            
            pred_scores.append(float(pred.item()))
            gt_scores.extend(label.cpu().tolist())
    
    if len(pred_scores) == 0:
        print(f"  ❌ Error: No predictions generated")
        return None, None, 0
    
    # 计算每个图像的平均分数（与训练时完全一致）
    # 参考 HyperIQASolver_swin.py 的 test() 方法
    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, patch_num)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, patch_num)), axis=1)
    
    num_images = len(pred_scores)
    
    # 计算相关系数（与训练时完全一致）
    # SRCC (Spearman Rank Correlation Coefficient)
    srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    
    # PLCC (Pearson Linear Correlation Coefficient)
    plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    
    print(f'  ✅ {dataset_name.upper()} Results:')
    print(f'     Images: {num_images}')
    print(f'     SRCC: {srcc:.4f}')
    print(f'     PLCC: {plcc:.4f}')
    
    return srcc, plcc, num_images


def load_model(checkpoint_path, model_size, device):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        model_size: 模型大小 (tiny/small/base)
        device: 计算设备
    
    Returns:
        model_hyper: 加载好的HyperNet模型
        dropout_rate: 模型的dropout率
    """
    print(f'\n{"="*60}')
    print(f'Loading Model')
    print(f'{"="*60}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Model size: {model_size}')
    print(f'Device: {device}')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 从checkpoint文件名中提取dropout_rate（如果有的话）
    # 例如: best_model_srcc_0.9236_plcc_0.9406.pkl
    dropout_rate = 0.3  # 默认值
    
    # 创建模型（与训练时的HyperNet一致）
    # 参考 HyperIQASolver_swin.py 的 __init__ 方法
    model_hyper = models.HyperNet(
        16, 112, 224, 112, 56, 28, 14, 7,
        use_multiscale=True,  # 默认启用多尺度特征融合
        drop_path_rate=0.2,   # 默认值（训练时的值）
        dropout_rate=dropout_rate,
        model_size=model_size
    ).to(device)
    
    # 加载权重
    print(f'  Loading checkpoint...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_hyper.load_state_dict(checkpoint)
    model_hyper.train(False)  # 设置为评估模式
    
    print(f'  ✅ Model loaded successfully!')
    
    return model_hyper, dropout_rate


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Dataset Testing for Swin Transformer IQA Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on all datasets with Swin-Tiny model
  python cross_dataset_test.py --checkpoint checkpoints/best_model.pkl --model_size tiny
  
  # Test on specific datasets with Swin-Small model
  python cross_dataset_test.py --checkpoint checkpoints/best_model.pkl --model_size small --datasets koniq spaq
  
  # Use RandomCrop for testing (less reproducible but matches original paper)
  python cross_dataset_test.py --checkpoint checkpoints/best_model.pkl --model_size tiny --test_random_crop
        """
    )
    
    # 必需参数
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, required=True,
                       help='Path to checkpoint file (.pkl)')
    parser.add_argument('--model_size', dest='model_size', type=str, required=True,
                       choices=['tiny', 'small', 'base'],
                       help='Swin Transformer model size: tiny, small, or base')
    
    # 可选参数
    parser.add_argument('--datasets', dest='datasets', type=str, nargs='+',
                       default=['koniq', 'spaq', 'kadid', 'agiqa'],
                       choices=['koniq', 'spaq', 'kadid', 'agiqa'],
                       help='Datasets to test on (default: all)')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=20,
                       help='Number of patches per image (default: 20)')
    parser.add_argument('--test_random_crop', dest='test_random_crop', action='store_true',
                       help='Use RandomCrop for testing (default: CenterCrop for reproducibility)')
    parser.add_argument('--base_dir', dest='base_dir', type=str, default=None,
                       help='Base directory for datasets (default: script directory)')
    
    config = parser.parse_args()
    
    # 设置基础目录
    if config.base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        base_dir = config.base_dir
    
    # 数据集配置
    datasets_config = {
        'koniq': {
            'path': os.path.join(base_dir, 'koniq-10k') + '/',
            'json': 'koniq_test.json',  # 使用KonIQ-10k的测试集
            'name': 'KonIQ-10k Test Set'
        },
        'spaq': {
            'path': os.path.join(base_dir, 'spaq-test') + '/',
            'json': 'spaq_test.json',
            'name': 'SPAQ'
        },
        'kadid': {
            'path': os.path.join(base_dir, 'kadid-test') + '/',
            'json': 'kadid_test.json',
            'name': 'KADID-10K'
        },
        'agiqa': {
            'path': os.path.join(base_dir, 'agiqa-test') + '/',
            'json': 'agiqa_test.json',
            'name': 'AGIQA-3K'
        }
    }
    
    # 打印配置信息
    print('\n' + '='*60)
    print('Cross-Dataset Testing Configuration')
    print('='*60)
    print(f'Checkpoint: {config.checkpoint}')
    print(f'Model size: {config.model_size}')
    print(f'Test patch num: {config.test_patch_num}')
    print(f'Crop method: {"RandomCrop" if config.test_random_crop else "CenterCrop"}')
    print(f'Datasets: {", ".join([d.upper() for d in config.datasets])}')
    print('='*60)
    
    # 获取设备
    device = get_device()
    print(f'Using device: {device}')
    
    # 加载模型
    try:
        model_hyper, dropout_rate = load_model(config.checkpoint, config.model_size, device)
    except Exception as e:
        print(f'\n❌ Error loading model: {e}')
        import traceback
        traceback.print_exc()
        return
    
    # 在各数据集上测试
    results = {}
    for dataset_name in config.datasets:
        dataset_info = datasets_config[dataset_name]
        try:
            srcc, plcc, num_images = test_on_dataset(
                dataset_info['name'],
                dataset_info['path'],
                dataset_info['json'],
                device,
                model_hyper,
                config.test_patch_num,
                dropout_rate,
                config.test_random_crop
            )
            
            if srcc is not None and plcc is not None:
                results[dataset_info['name']] = {
                    'srcc': srcc,
                    'plcc': plcc,
                    'num_images': num_images
                }
            else:
                results[dataset_info['name']] = None
                
        except Exception as e:
            print(f'\n❌ Error testing on {dataset_info["name"]}: {e}')
            import traceback
            traceback.print_exc()
            results[dataset_info['name']] = None
    
    # 打印最终结果汇总
    print('\n' + '='*60)
    print('FINAL RESULTS SUMMARY')
    print('='*60)
    print(f'Model: Swin-{config.model_size.capitalize()}')
    print(f'Checkpoint: {os.path.basename(config.checkpoint)}')
    print(f'Test patch num: {config.test_patch_num}')
    print('-'*60)
    
    # 表格形式打印
    print(f'{"Dataset":<20} {"Images":<10} {"SRCC":<10} {"PLCC":<10}')
    print('-'*60)
    
    for dataset_name, result in results.items():
        if result is not None:
            print(f'{dataset_name:<20} {result["num_images"]:<10} '
                  f'{result["srcc"]:<10.4f} {result["plcc"]:<10.4f}')
        else:
            print(f'{dataset_name:<20} {"Failed":<10} {"-":<10} {"-":<10}')
    
    print('='*60)
    
    # 计算平均性能（仅针对成功的数据集）
    successful_results = [r for r in results.values() if r is not None]
    if len(successful_results) > 0:
        avg_srcc = np.mean([r['srcc'] for r in successful_results])
        avg_plcc = np.mean([r['plcc'] for r in successful_results])
        print(f'\nAverage across {len(successful_results)} datasets:')
        print(f'  Average SRCC: {avg_srcc:.4f}')
        print(f'  Average PLCC: {avg_plcc:.4f}')
        print('='*60)
    
    # 保存结果到JSON文件
    output_file = f'cross_dataset_results_{config.model_size}_{os.path.basename(config.checkpoint).replace(".pkl", "")}.json'
    output_path = os.path.join(base_dir, output_file)
    
    output_data = {
        'checkpoint': config.checkpoint,
        'model_size': config.model_size,
        'test_patch_num': config.test_patch_num,
        'test_random_crop': config.test_random_crop,
        'results': {k: v for k, v in results.items()}
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f'\n✅ Results saved to: {output_path}')


if __name__ == '__main__':
    main()

