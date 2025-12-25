#!/usr/bin/env python3
"""
注意力可视化脚本 - SMART-IQA
可视化Swin Transformer中的channel attention权重
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
import os
import sys
from torchvision import transforms
import cv2

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from smart_iqa.models import smart_iqa as models


class AttentionVisualizer:
    def __init__(self, model_path, device='cuda'):
        """
        初始化注意力可视化器
        
        Args:
            model_path: 模型checkpoint路径
            device: 运行设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载模型
        print(f"Loading model from: {model_path}")
        # 根据checkpoint中的信息自动检测模型配置
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型（与训练时相同的配置）
        self.model = models.HyperNet(
            16, 112, 224, 112, 56, 28, 14, 7,
            use_multiscale=True,
            use_attention=True,
            model_size='base'  # 从checkpoint文件名判断
        ).to(self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("✓ Model loaded successfully")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 存储注意力权重
        self.attention_weights = None
        self.feature_maps = {}
        
        # 注册hook来提取注意力权重
        self._register_hooks()
    
    def _register_hooks(self):
        """注册forward hooks来提取中间特征和注意力权重"""
        
        def save_attention_hook(module, input, output):
            """保存attention权重"""
            # MultiScaleAttention的输出是(fused_feat, attention_weights)
            if isinstance(output, tuple) and len(output) == 2:
                features, weights = output
                self.attention_weights = weights.detach().cpu()
                print(f"  [Hook] Captured attention weights: {weights.squeeze().detach().cpu().numpy()}")
        
        def save_feature_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.feature_maps[name] = output.detach().cpu()
            return hook
        
        # 注册hook到HyperNet中的multiscale_attention模块
        if hasattr(self.model, 'multiscale_attention'):
            self.model.multiscale_attention.register_forward_hook(save_attention_hook)
            print("✓ Registered hook for model.multiscale_attention")
        
        # 注册hook到Swin backbone的各个stage输出
        if hasattr(self.model, 'swin'):
            if hasattr(self.model.swin, 'model') and hasattr(self.model.swin.model, 'layers'):
                for i, layer in enumerate(self.model.swin.model.layers):
                    layer.register_forward_hook(save_feature_hook(f'stage_{i}'))
                print(f"✓ Registered hooks for {len(self.model.swin.model.layers)} backbone stages")
    
    def preprocess_image(self, image_path):
        """
        预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            tensor: 预处理后的图像tensor
            original: 原始PIL图像
        """
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor, image
    
    def forward_and_extract(self, image_tensor):
        """
        前向传播并提取注意力权重
        
        Args:
            image_tensor: 输入图像tensor
            
        Returns:
            quality_score: 预测的质量分数
            attention_weights: 注意力权重
        """
        with torch.no_grad():
            # HyperNet forward returns dict of weights
            paras = self.model(image_tensor)
            
            # Use TargetNet to compute quality score
            target_net = models.TargetNet(paras, dropout_rate=0).to(self.device)
            target_net.eval()
            quality_score = target_net(paras['target_in_vec'])
        
        return quality_score.mean().item(), self.attention_weights
    
    def visualize_attention_weights(self, attention_weights, save_path=None, title="Channel Attention Weights"):
        """
        可视化注意力权重（柱状图）
        
        Args:
            attention_weights: 注意力权重 (1, num_scales)
            save_path: 保存路径
            title: 图表标题
        """
        if attention_weights is None:
            print("Warning: No attention weights to visualize")
            return
        
        weights = attention_weights.squeeze().numpy()
        num_scales = len(weights)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 颜色映射
        colors = ['#4ECDC4', '#95E1D3', '#FFD93D'][:num_scales]
        
        # 绘制柱状图
        bars = ax.bar(range(num_scales), weights, color=colors, 
                     edgecolor='black', linewidth=2, alpha=0.8, width=0.6)
        
        # 设置标签
        scale_names = [f'Stage {i+1}\n(Low-level)' if i == 0 
                      else f'Stage {i+1}\n(Mid-level)' if i == 1
                      else f'Stage {i+1}\n(High-level)'
                      for i in range(num_scales)]
        
        ax.set_xticks(range(num_scales))
        ax.set_xticklabels(scale_names, fontsize=11, weight='bold')
        ax.set_ylabel('Attention Weight', fontsize=13, weight='bold')
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_ylim([0, max(weights) * 1.2])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 在柱子上标注数值
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{weight:.3f}',
                   ha='center', va='bottom', fontsize=11, weight='bold')
            
            # 添加百分比
            percentage = weight / weights.sum() * 100
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{percentage:.1f}%',
                   ha='center', va='center', fontsize=10, color='white', weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Attention weights visualization saved to: {save_path}")
        
        plt.savefig(save_path.replace('.png', '_show.png'), dpi=150)
        plt.close()
    
    def visualize_attention_heatmap(self, image_path, attention_weights, save_path=None):
        """
        将注意力权重叠加到原始图像上（热力图效果）
        
        Args:
            image_path: 原始图像路径
            attention_weights: 注意力权重
            save_path: 保存路径
        """
        # 读取原始图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if attention_weights is None:
            print("Warning: No attention weights to visualize")
            return
        
        weights = attention_weights.squeeze().numpy()
        
        # 创建可视化
        fig, axes = plt.subplots(1, len(weights) + 1, figsize=(4 * (len(weights) + 1), 4))
        
        # 显示原图
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=12, weight='bold')
        axes[0].axis('off')
        
        # 为每个scale创建热力图效果
        for i, weight in enumerate(weights):
            # 创建热力图（简单版本：根据权重调整图像亮度）
            alpha = weight / weights.sum()  # 归一化权重
            
            # 创建彩色覆盖层
            overlay = np.ones_like(image, dtype=np.float32)
            colors = [[0.3, 0.8, 0.77], [0.58, 0.88, 0.83], [1.0, 0.85, 0.24]]  # 对应RGB
            if i < len(colors):
                overlay = overlay * np.array(colors[i])
            
            # 混合原图和覆盖层
            blended = (image.astype(np.float32) * (1 - alpha * 0.5) + 
                      overlay * 255 * alpha * 0.5).astype(np.uint8)
            
            axes[i + 1].imshow(blended)
            axes[i + 1].set_title(f'Stage {i+1} (α={weight:.3f})', 
                                 fontsize=12, weight='bold')
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Attention heatmap saved to: {save_path}")
        
        plt.close()
    
    def process_image(self, image_path, output_dir='attention_visualizations'):
        """
        处理单张图像并生成所有可视化
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取图像名称
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        print(f"\n{'='*80}")
        print(f"Processing: {image_name}")
        print(f"{'='*80}")
        
        # 预处理图像
        image_tensor, original_image = self.preprocess_image(image_path)
        print(f"✓ Image loaded: {original_image.size}")
        
        # 前向传播
        quality_score, attention_weights = self.forward_and_extract(image_tensor)
        print(f"✓ Predicted quality score: {quality_score:.4f}")
        
        if attention_weights is not None:
            weights = attention_weights.squeeze().numpy()
            print(f"✓ Attention weights: {weights}")
            print(f"   Stage distribution: {weights / weights.sum() * 100}")
        
        # 生成可视化
        # 1. 注意力权重柱状图
        weights_path = os.path.join(output_dir, f'{image_name}_attention_weights.png')
        self.visualize_attention_weights(
            attention_weights, 
            save_path=weights_path,
            title=f'Channel Attention Weights (Quality: {quality_score:.4f})'
        )
        
        # 2. 注意力热力图
        heatmap_path = os.path.join(output_dir, f'{image_name}_attention_heatmap.png')
        self.visualize_attention_heatmap(image_path, attention_weights, save_path=heatmap_path)
        
        print(f"{'='*80}\n")
        
        return quality_score, attention_weights


def main():
    """主函数"""
    print("=" * 80)
    print("SMART-IQA 注意力可视化工具")
    print("=" * 80)
    
    # 配置
    model_path = 'checkpoints/koniq-10k-swin_20251223_002226/best_model_srcc_0.9378_plcc_0.9485.pkl'
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please specify the correct model path")
        return
    
    # 读取选定的图片
    with open('selected_images_for_viz.json') as f:
        selected_images = json.load(f)
    
    # 查找图片实际路径（可能在train或test目录）
    images_to_process = []
    for quality in ['low', 'mid', 'high']:
        img_name = selected_images[quality]['image']
        mos = selected_images[quality]['mos']
        
        # 检查train和test目录
        img_path = None
        for subdir in ['train', 'test']:
            candidate_path = os.path.join('koniq-10k', subdir, img_name)
            if os.path.exists(candidate_path):
                img_path = candidate_path
                break
        
        if img_path:
            images_to_process.append((quality, img_path, mos))
        else:
            print(f"Warning: Image not found: {img_name}")
    
    if not images_to_process:
        print("❌ No images found!")
        return
    
    # 创建可视化器
    print("\nInitializing visualizer...")
    visualizer = AttentionVisualizer(model_path)
    
    # 处理每张图片
    results = []
    for quality, img_path, mos in images_to_process:
        print(f"\n\n{'#'*80}")
        print(f"# {quality.upper()} QUALITY IMAGE (GT MOS: {mos:.4f})")
        print(f"{'#'*80}")
        
        score, weights = visualizer.process_image(img_path)
        results.append({
            'quality': quality,
            'image': os.path.basename(img_path),
            'gt_mos': mos,
            'pred_score': score,
            'attention_weights': weights.squeeze().numpy().tolist() if weights is not None else None
        })
    
    # 保存结果
    with open('attention_visualization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n\n" + "=" * 80)
    print("✅ 注意力可视化完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print("  - attention_visualizations/*_attention_weights.png  (注意力权重柱状图)")
    print("  - attention_visualizations/*_attention_heatmap.png  (注意力热力图)")
    print("  - attention_visualization_results.json              (数值结果)")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

