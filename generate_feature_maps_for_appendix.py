"""
生成简洁版特征图用于论文附录
去掉不必要的标注，统一字体
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
                if isinstance(output, dict) and 'hyper_in_feat_multi' in output:
                    feat0, feat1, feat2, feat3 = output['hyper_in_feat_multi']
                    self.features['stage0'] = feat0.detach()
                    self.features['stage1'] = feat1.detach()
                    self.features['stage2'] = feat2.detach()
                    self.features['stage3'] = feat3.detach()
                else:
                    self.features[name] = output.detach()
            return hook
        
        self.model.swin.register_forward_hook(get_activation('swin'))
    
    def extract_features(self, img_path):
        """提取一张图片的4个stage特征"""
        img = Image.open(img_path).convert('RGB')
        img_original = np.array(img)
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        return self.features, img_original


def visualize_feature_heatmaps_clean(features, original_img, save_path, quality_label):
    """
    生成简洁版特征热力图用于附录
    
    Args:
        features: dict with keys 'stage0', 'stage1', 'stage2', 'stage3'
        original_img: 原始图片 (numpy array)
        save_path: 保存路径
        quality_label: 质量标签（用于标题）
    """
    # 统一字体为Times
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 11
    
    # 创建图表 - 3行2列布局（更紧凑）
    fig = plt.figure(figsize=(9, 9))
    
    # 顶部：原始图片（居中，跨两列）
    ax_img = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_img.imshow(original_img)
    ax_img.set_title(f'Original Image ({quality_label})', fontsize=13, fontweight='bold')
    ax_img.axis('off')
    
    # 4个stage的特征图 - 2x2布局
    stage_names = [
        'Stage 0 (Low-level: 56×56)', 
        'Stage 1 (Mid-level: 28×28)', 
        'Stage 2 (High-level: 14×14)', 
        'Stage 3 (Semantic: 7×7)'
    ]
    
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
        
        # 对所有通道取平均
        feat_mean = feat[0].mean(dim=0).cpu().numpy()  # (H, W)
        
        # 归一化到0-1
        feat_min, feat_max = feat_mean.min(), feat_mean.max()
        if feat_max > feat_min:
            feat_norm = (feat_mean - feat_min) / (feat_max - feat_min)
        else:
            feat_norm = feat_mean
        
        # 绘制热力图
        ax = plt.subplot2grid((3, 2), pos)
        im = ax.imshow(feat_norm, cmap='jet', interpolation='bilinear')
        
        # 标题
        ax.set_title(stage_name, fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout(w_pad=0.5)  # 减小横向间距
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    # PNG版本
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def main():
    print("=" * 80)
    print("生成简洁版特征图用于附录")
    print("=" * 80)
    
    model_path = 'checkpoints/koniq-10k-swin_20251223_002226/best_model_srcc_0.9378_plcc_0.9485.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # 选择之前效果比较好的两张
    test_images = [
        ('koniq-10k/train/7178199774.jpg', 'High Quality', 'high_quality'),     # MOS 81
        ('koniq-10k/train/244484608.jpg', 'Low Quality', 'low_quality'),        # MOS 16
    ]
    
    # 创建特征提取器
    extractor = FeatureExtractor(model_path)
    
    # 创建输出目录
    output_dir = 'paper_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每张图片
    for img_path, quality_label, filename_prefix in test_images:
        # 尝试train和test两个目录
        if not os.path.exists(img_path):
            alt_path = img_path.replace('/train/', '/test/')
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                print(f"⚠ Image not found: {img_path}")
                continue
        
        print(f"\nProcessing: {os.path.basename(img_path)} ({quality_label})")
        
        # 提取特征
        features, original_img = extractor.extract_features(img_path)
        
        # 可视化
        save_path = os.path.join(output_dir, f'feature_map_{filename_prefix}_appendix.pdf')
        visualize_feature_heatmaps_clean(features, original_img, save_path, quality_label)
    
    print("\n" + "=" * 80)
    print("✅ 附录特征图生成完成！")
    print("=" * 80)
    print(f"\n生成文件位置: {output_dir}/")
    print("  - feature_map_high_quality_appendix.pdf")
    print("  - feature_map_low_quality_appendix.pdf")
    print("  - (对应的PNG预览版)")


if __name__ == '__main__':
    main()

