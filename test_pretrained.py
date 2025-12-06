import os
import argparse
import torch
import torchvision
import models
import data_loader
from scipy import stats
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

# 自动检测可用设备：优先使用 CUDA，然后是 MPS (macOS GPU)，最后使用 CPU
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


class JSONTestFolder(torch.utils.data.Dataset):
    """通用JSON格式测试数据集加载器"""
    def __init__(self, root, json_file, transform, patch_num):
        self.root = root
        self.transform = transform
        self.patch_num = patch_num
        
        # 加载JSON文件
        json_path = os.path.join(root, json_file)
        with open(json_path) as f:
            data = json.load(f)
        
        self.samples = []
        for item in data:
            img_path = os.path.join(root, os.path.basename(item['image']))
            # 处理score可能是字符串或数字的情况
            score = float(item['score'])
            for aug in range(patch_num):
                self.samples.append((img_path, score))
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __len__(self):
        return len(self.samples)


def test_on_dataset(dataset_name, dataset_path, json_file, device, model_hyper, patch_num=25):
    """在指定数据集上测试预训练模型"""
    print(f'\nTesting on {dataset_name} dataset...')
    print(f'Dataset path: {dataset_path}')
    
    # 所有数据集使用统一的transform（参考demo.py和data_loader.py中koniq-10k的transform）
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 384)),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])
    
    
    # 所有数据集都使用统一的JSON加载器
    dataset = JSONTestFolder(dataset_path, json_file, transforms, patch_num)
    test_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)
    
    # 测试
    model_hyper.eval()
    pred_scores = []
    gt_scores = []
    
    print(f'  Total batches: {len(test_data_loader)}')
    with torch.no_grad():
        # 使用 tqdm 显示进度条
        for batch_idx, (img, label) in enumerate(tqdm(test_data_loader, desc=f'  Testing {dataset_name}')):
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

    
    # 计算每个图片的平均分数（每个图片有patch_num个patch）
    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)
    
    # 获取实际图片数量
    num_images = len(gt_scores) // patch_num
    
    # 重塑为 (num_images, patch_num)
    pred_scores = np.reshape(pred_scores[:num_images * patch_num], (num_images, patch_num))
    gt_scores = np.reshape(gt_scores[:num_images * patch_num], (num_images, patch_num))
    
    # 计算每个图片的平均分数
    pred_scores_mean = np.mean(pred_scores, axis=1)
    gt_scores_mean = np.mean(gt_scores, axis=1)
    
    # 计算 SRCC (Spearman Rank Correlation Coefficient)
    # 公式: SRCC = 1 - (6 * Σ(d_i^2)) / (n * (n^2 - 1))
    # 其中 d_i 是排名差，n 是样本数
    # scipy.stats.spearmanr 使用正确的公式
    srcc, _ = stats.spearmanr(pred_scores_mean, gt_scores_mean)
    
    # 计算 PLCC (Pearson Linear Correlation Coefficient)
    # 公式: PLCC = Σ((y_i - y_bar) * (y_hat_i - y_hat_bar)) / 
    #              (sqrt(Σ((y_i - y_bar)^2)) * sqrt(Σ((y_hat_i - y_hat_bar)^2)))
    # scipy.stats.pearsonr 使用正确的公式
    plcc, _ = stats.pearsonr(pred_scores_mean, gt_scores_mean)
    
    return srcc, plcc, num_images


def main():
    parser = argparse.ArgumentParser(description='Test pretrained model on test datasets')
    parser.add_argument('--dataset', dest='dataset', type=str, 
                       default='koniq', 
                       help='Test dataset: koniq | spaq | kadid | agiqa | all')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25,
                       help='Number of sample patches from testing image')
    parser.add_argument('--model_path', dest='model_path', type=str,
                       default='./pretrained/koniq_pretrained.pkl',
                       help='Path to pretrained model')
    
    config = parser.parse_args()
    
    # 获取当前脚本所在目录的绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据集配置（所有测试集统一结构）
    datasets_config = {
        'koniq': {
            'path': os.path.join(base_dir, 'koniq-test') + '/',
            'json': 'koniq_test.json'
        },
        'spaq': {
            'path': os.path.join(base_dir, 'spaq-test') + '/',
            'json': 'spaq_test.json'
        },
        'kadid': {
            'path': os.path.join(base_dir, 'kadid-test') + '/',
            'json': 'kadid_test.json'
        },
        'agiqa': {
            'path': os.path.join(base_dir, 'agiqa-test') + '/',
            'json': 'agiqa_test.json'
        }
    }
    
    # 设置设备
    device = get_device()
    print(f'Using device: {device}')
    
    # 加载预训练模型
    print(f'\nLoading pretrained model from {config.model_path}...')
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model_hyper.train(False)
    model_hyper.load_state_dict(torch.load(config.model_path, map_location=device))
    print('Pretrained model loaded successfully!')
    
    # 确定要测试的数据集
    if config.dataset == 'all':
        test_datasets = list(datasets_config.keys())
    else:
        if config.dataset not in datasets_config:
            print(f'Error: Unknown dataset {config.dataset}')
            print(f'Available datasets: {list(datasets_config.keys())} | all')
            return
        test_datasets = [config.dataset]
    
    # 测试所有指定的数据集
    results = {}
    for dataset_name in test_datasets:
        dataset_info = datasets_config[dataset_name]
        try:
            srcc, plcc, num_images = test_on_dataset(
                dataset_name,
                dataset_info['path'],
                dataset_info['json'],
                device,
                model_hyper,
                config.test_patch_num
            )
            results[dataset_name] = {
                'srcc': srcc,
                'plcc': plcc,
                'num_images': num_images
            }
        except Exception as e:
            print(f'Error testing on {dataset_name}: {e}')
            import traceback
            traceback.print_exc()
            results[dataset_name] = None
    
    # 打印结果
    print('\n' + '='*60)
    print('Test Results Summary:')
    print('='*60)
    for dataset_name, result in results.items():
        if result is not None:
            print(f'\n{dataset_name.upper()}:')
            print(f'  Number of test images: {result["num_images"]}')
            print(f'  SRCC (Spearman Rank Correlation): {result["srcc"]:.4f}')
            print(f'  PLCC (Pearson Linear Correlation): {result["plcc"]:.4f}')
        else:
            print(f'\n{dataset_name.upper()}: Failed to test')
    print('='*60)


if __name__ == '__main__':
    main()
