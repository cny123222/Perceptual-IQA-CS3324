"""
特征图热力图可视化
显示同一图像在4个Swin Transformer stage的特征激活
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from PIL import Image
import torchvision.transforms as transforms
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_swin as models


class FeatureExtractor:
    """提取Swin Transformer各个stage的特征图"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载模型
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = models.HyperNet(
            16, 112, 224, 112, 56, 28, 14, 7,
            use_multiscale=True,
            use_attention=True,
            model_size='base'
        ).to(self.device)
        
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("✓ Model loaded\n")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 用于存储中间特征
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册hook以提取中间特征"""
        def get_activation(name):
            def hook(module, input, output):
                # 如果output是字典（Swin backbone的情况）
                if isinstance(output, dict) and 'hyper_in_feat_multi' in output:
                    feat0, feat1, feat2, feat3 = output['hyper_in_feat_multi']
                    self.features['stage0'] = feat0.detach()
                    self.features['stage1'] = feat1.detach()
                    self.features['stage2'] = feat2.detach()
                    self.features['stage3'] = feat3.detach()
                else:
                    self.features[name] = output.detach()
            return hook
        
        # 注册到Swin backbone
        self.model.swin.register_forward_hook(get_activation('swin'))
    
    def extract_features(self, img_path):
        """提取一张图片的4个stage特征"""
        # 读取并预处理图片
        img = Image.open(img_path).convert('RGB')
        img_original = np.array(img)
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        # 返回特征和原始图片
        return self.features, img_original


def visualize_feature_heatmaps(features, original_img, save_path):
    """
    可视化4个stage的特征热力图
    
    Args:
        features: dict with keys 'stage0', 'stage1', 'stage2', 'stage3'
        original_img: 原始图片 (numpy array)
        save_path: 保存路径
    """
    # 设置字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    
    # 创建图表 - 3行3列布局
    fig = plt.figure(figsize=(12, 10))
    
    # 顶部跨两列: 原始图片
    ax_img = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax_img.imshow(original_img)
    ax_img.set_title('Original Image with Distortion', fontsize=14, fontweight='bold')
    ax_img.axis('off')
    
    # 右上角: 图例说明
    ax_legend = plt.subplot2grid((3, 3), (0, 2))
    ax_legend.axis('off')
    ax_legend.text(0.1, 0.9, 'Feature Activation Heatmap', 
                   fontsize=11, fontweight='bold', transform=ax_legend.transAxes)
    ax_legend.text(0.1, 0.7, '🔴 Red = High activation\n    (Model focuses here)', 
                   fontsize=9, transform=ax_legend.transAxes)
    ax_legend.text(0.1, 0.4, '🔵 Blue = Low activation\n    (Less important)', 
                   fontsize=9, transform=ax_legend.transAxes)
    
    # 4个stage的特征图
    stage_names = ['Stage 0\n(Low-level: 56×56)', 
                   'Stage 1\n(Mid-level: 28×28)', 
                   'Stage 2\n(High-level: 14×14)', 
                   'Stage 3\n(Semantic: 7×7)']
    
    positions = [
        (1, 0),  # Stage 0: 第2行第1列
        (1, 1),  # Stage 1: 第2行第2列
        (2, 0),  # Stage 2: 第3行第1列
        (2, 1),  # Stage 3: 第3行第2列
    ]
    
    # 提取并可视化每个stage
    for i, (stage_key, stage_name, pos) in enumerate(zip(
        ['stage0', 'stage1', 'stage2', 'stage3'], 
        stage_names, 
        positions
    )):
        if stage_key not in features:
            continue
        
        feat = features[stage_key]  # Shape: (1, C, H, W)
        
        # 对所有通道取平均，得到激活强度
        feat_mean = feat[0].mean(dim=0).cpu().numpy()  # (H, W)
        
        # 归一化到0-1
        feat_min, feat_max = feat_mean.min(), feat_mean.max()
        if feat_max > feat_min:
            feat_norm = (feat_mean - feat_min) / (feat_max - feat_min)
        else:
            feat_norm = feat_mean
        
        # 绘制热力图
        ax = plt.subplot2grid((3, 3), pos)
        im = ax.imshow(feat_norm, cmap='jet', interpolation='bilinear')
        
        # 标题显示stage信息
        ax.set_title(stage_name, fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation', fontsize=9)
    
    # 右下角: 统计信息
    ax_stats = plt.subplot2grid((3, 3), (2, 2))
    ax_stats.axis('off')
    ax_stats.text(0.1, 0.9, 'Feature Statistics:', 
                  fontsize=10, fontweight='bold', transform=ax_stats.transAxes)
    
    for i, stage_key in enumerate(['stage0', 'stage1', 'stage2', 'stage3']):
        if stage_key not in features:
            continue
        feat = features[stage_key]
        channels = feat.shape[1]
        spatial = f"{feat.shape[2]}×{feat.shape[3]}"
        ax_stats.text(0.1, 0.75 - i*0.15, 
                     f'S{i}: {channels}ch, {spatial}',
                     fontsize=8, transform=ax_stats.transAxes)
    
    plt.tight_layout()
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature heatmap to: {save_path}")
    
    # 也保存PNG版本
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PNG version to: {png_path}")
    
    plt.close()


def main():
    print("=" * 80)
    print("Feature Map Heatmap Visualization")
    print("=" * 80)
    
    # 配置
    model_path = 'checkpoints/koniq-10k-swin_20251223_002226/best_model_srcc_0.9378_plcc_0.9485.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # 使用注意力分析中表现好的图片（质量对比明显）
    # 明确选择：1高质量 + 1中低质量 + 1极低质量
    test_images = [
        ('koniq-10k/test/320987228.jpg', 'high_quality_MOS73'),      # 高质量
        ('koniq-10k/train/5348237812.jpg', 'low_quality_MOS38'),     # 低质量  
        ('koniq-10k/train/5135908583.jpg', 'very_low_quality_MOS26'), # 极低质量
    ]
    
    # 检查文件，如果不存在尝试另一个目录
    verified_images = []
    for path, label in test_images:
        if os.path.exists(path):
            verified_images.append((path, label))
        else:
            # 尝试另一个目录
            alt_path = path.replace('/train/', '/test/') if '/train/' in path else path.replace('/test/', '/train/')
            if os.path.exists(alt_path):
                verified_images.append((alt_path, label))
    
    test_images = verified_images
    
    # 创建特征提取器
    extractor = FeatureExtractor(model_path)
    
    # 创建输出目录
    output_dir = 'feature_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每张图片
    for img_path, quality_label in test_images:
        if not os.path.exists(img_path):
            print(f"⚠ Image not found: {img_path}")
            continue
        
        print(f"\n Processing: {os.path.basename(img_path)} ({quality_label})")
        
        # 提取特征
        features, original_img = extractor.extract_features(img_path)
        
        # 可视化
        save_path = os.path.join(output_dir, f'feature_heatmap_{quality_label}.pdf')
        visualize_feature_heatmaps(features, original_img, save_path)
    
    print("\n" + "=" * 80)
    print("✅ Feature visualization completed!")
    print("=" * 80)
    print(f"\nGenerated files in: {output_dir}/")
    print("  - feature_heatmap_*.pdf (for paper)")
    print("  - feature_heatmap_*.png (preview)")


if __name__ == '__main__':
    main()

