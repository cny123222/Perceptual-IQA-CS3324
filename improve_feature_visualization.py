"""
改进版特征图可视化
解决可能的问题：
1. 特征图可能过于抽象，难以解释
2. 不同stage的对比度可能不明显
3. 缺少与原图的空间对应关系
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_swin as models


class ImprovedFeatureExtractor:
    """改进的特征提取器"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
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
        
        # 图像预处理（保持长宽比）
        self.transform = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
            return hook
        
        self.model.swin.register_forward_hook(get_activation('swin'))
    
    def extract_features(self, img_path):
        """提取特征"""
        img = Image.open(img_path).convert('RGB')
        img_original = np.array(img)
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        return self.features, img_original


def create_overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    创建热力图叠加到原图上的效果
    这样更容易看出模型关注的区域
    """
    # Resize heatmap to match original image
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize to 0-255
    heatmap_norm = ((heatmap_resized - heatmap_resized.min()) / 
                    (heatmap_resized.max() - heatmap_resized.min()) * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlay = cv2.addWeighted(original_img, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlay


def visualize_improved_features(features, original_img, save_path, quality_label):
    """
    改进版可视化：
    1. 使用更好的布局
    2. 添加overlay显示
    3. 只展示最有意义的特征
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    
    # 创建更紧凑的布局
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.25, wspace=0.3)
    
    # 第一行：原始图片 + 4个stage的纯热力图
    stage_info = [
        ('stage0', 'Stage 0\n(56×56)', 0),
        ('stage1', 'Stage 1\n(28×28)', 1),
        ('stage2', 'Stage 2\n(14×14)', 2),
        ('stage3', 'Stage 3\n(7×7)', 3),
    ]
    
    # 原图（左上，跨两行）
    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(original_img)
    ax_orig.set_title(f'Original Image\n({quality_label.replace("_", " ").title()})', 
                     fontsize=12, fontweight='bold')
    ax_orig.axis('off')
    
    # 第一行：4个stage的热力图
    for stage_key, stage_name, col_idx in stage_info:
        if stage_key not in features:
            continue
        
        feat = features[stage_key][0].mean(dim=0).cpu().numpy()
        
        # 归一化
        feat_norm = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        
        # 第一行：纯热力图
        ax1 = fig.add_subplot(gs[0, col_idx+1])
        im1 = ax1.imshow(feat_norm, cmap='jet', interpolation='bilinear')
        ax1.set_title(stage_name, fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        # 添加小的colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        
        # 第二行：overlay (叠加到原图)
        ax2 = fig.add_subplot(gs[1, col_idx+1])
        overlay = create_overlay_heatmap(original_img, feat_norm, alpha=0.5)
        ax2.imshow(overlay)
        ax2.set_title('Overlay', fontsize=9, style='italic')
        ax2.axis('off')
    
    # 总标题
    fig.suptitle('Multi-Scale Feature Activation Analysis', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved improved visualization to: {save_path}")
    
    # PNG版本
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def main():
    print("=" * 80)
    print("Improved Feature Map Visualization")
    print("=" * 80)
    
    model_path = 'checkpoints/koniq-10k-swin_20251223_002226/best_model_srcc_0.9378_plcc_0.9485.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # 测试图片
    test_images = [
        ('koniq-10k/test/7358286276.jpg', 'low_quality'),
        ('koniq-10k/test/320987228.jpg', 'high_quality'),
    ]
    
    extractor = ImprovedFeatureExtractor(model_path)
    
    output_dir = 'feature_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path, quality_label in test_images:
        if not os.path.exists(img_path):
            print(f"⚠ Image not found: {img_path}")
            continue
        
        print(f"\nProcessing: {os.path.basename(img_path)} ({quality_label})")
        
        features, original_img = extractor.extract_features(img_path)
        
        save_path = os.path.join(output_dir, f'feature_improved_{quality_label}.pdf')
        visualize_improved_features(features, original_img, save_path, quality_label)
    
    print("\n" + "=" * 80)
    print("✅ Improved visualization completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

