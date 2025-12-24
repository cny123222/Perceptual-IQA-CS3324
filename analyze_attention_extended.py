"""
扩展的注意力分析 - 不生成可视化图片，只输出数值结果
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_swin as models


class AttentionAnalyzer:
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
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def analyze_image(self, img_path, num_patches=10):
        """
        分析单张图片的注意力权重
        
        Args:
            img_path: 图片路径
            num_patches: 采样patch数量
            
        Returns:
            pred_score: 预测分数
            avg_weights: 平均注意力权重 (4,)
            std_weights: 注意力权重标准差 (4,)
        """
        img = Image.open(img_path).convert('RGB')
        
        # 生成多个patches
        patches = []
        for _ in range(num_patches):
            patch = self.transform(img)
            patches.append(patch)
        
        patches_tensor = torch.stack(patches).to(self.device)  # (N, 3, 224, 224)
        
        with torch.no_grad():
            # Forward pass
            paras = self.model(patches_tensor)
            
            # 获取注意力权重 (存储在模型属性中)
            if hasattr(self.model, 'last_attention_weights'):
                weights = self.model.last_attention_weights  # (N, 4)
            else:
                weights = None
            
            # 获取预测分数
            model_target = models.TargetNet(paras).to(self.device)
            model_target.eval()
            pred = model_target(paras['target_in_vec'])
            pred_score = float(pred.mean())
        
        if weights is not None:
            # 计算平均和标准差
            weights_np = weights.cpu().numpy()  # (N, 4)
            avg_weights = weights_np.mean(axis=0)
            std_weights = weights_np.std(axis=0)
            return pred_score, avg_weights, std_weights
        else:
            return pred_score, None, None


def main():
    print("=" * 80)
    print("扩展注意力分析 - 25张图片")
    print("=" * 80)
    
    model_path = 'checkpoints/koniq-10k-swin_20251223_002226/best_model_srcc_0.9378_plcc_0.9485.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # 读取扩展的图片列表
    with open('selected_images_for_viz_extended.json') as f:
        selected_images = json.load(f)
    
    # 创建分析器
    print("\nInitializing analyzer...")
    analyzer = AttentionAnalyzer(model_path)
    
    # 处理所有图片
    all_results = []
    
    for quality_level in ['very_low', 'low', 'mid', 'high', 'very_high']:
        print(f"\n{'='*80}")
        print(f"Quality Level: {quality_level.upper()}")
        print(f"{'='*80}")
        
        images = selected_images[quality_level]
        
        for img_info in images:
            img_name = img_info['image']
            gt_mos = img_info['mos']
            
            # 查找图片路径
            img_path = None
            for subdir in ['train', 'test']:
                candidate = os.path.join('koniq-10k', subdir, img_name)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            
            if not img_path:
                print(f"  ⚠ Image not found: {img_name}")
                continue
            
            # 分析
            pred_score, avg_weights, std_weights = analyzer.analyze_image(img_path, num_patches=10)
            
            result = {
                'quality_level': quality_level,
                'image': img_name,
                'gt_mos': gt_mos,
                'pred_score': pred_score,
                'avg_attention_weights': avg_weights.tolist() if avg_weights is not None else None,
                'std_attention_weights': std_weights.tolist() if std_weights is not None else None
            }
            
            all_results.append(result)
            
            # 打印结果
            print(f"\n  Image: {img_name}")
            print(f"    GT MOS:  {gt_mos:.2f}")
            print(f"    Pred:    {pred_score:.2f}")
            if avg_weights is not None:
                print(f"    Attention weights (avg ± std):")
                for i, (avg, std) in enumerate(zip(avg_weights, std_weights)):
                    print(f"      Stage {i+1}: {avg:.4f} ± {std:.4f}")
    
    # 保存结果
    output_file = 'attention_analysis_extended.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print(f"✅ 分析完成！")
    print(f"{'='*80}")
    print(f"处理了 {len(all_results)} 张图片")
    print(f"结果保存到: {output_file}")
    
    # 统计分析
    print(f"\n{'='*80}")
    print("统计摘要")
    print(f"{'='*80}")
    
    for quality_level in ['very_low', 'low', 'mid', 'high', 'very_high']:
        level_results = [r for r in all_results if r['quality_level'] == quality_level]
        if level_results:
            weights_list = [r['avg_attention_weights'] for r in level_results if r['avg_attention_weights']]
            if weights_list:
                weights_array = np.array(weights_list)  # (N, 4)
                mean_weights = weights_array.mean(axis=0)
                
                print(f"\n{quality_level.upper()}:")
                print(f"  Average attention weights across {len(weights_list)} images:")
                for i, w in enumerate(mean_weights):
                    print(f"    Stage {i+1}: {w:.4f}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()

