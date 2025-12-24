#!/usr/bin/env python3
"""
创建注意力可视化对比图用于论文
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
from PIL import Image
import os

# 设置matplotlib参数
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['text.usetex'] = False

# 读取结果
with open('attention_visualization_results.json') as f:
    results = json.load(f)

# 创建大图
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)

# 颜色方案
colors = ['#4ECDC4', '#95E1D3', '#FFD93D', '#FF6B6B']
quality_labels = ['Low Quality', 'Medium Quality', 'High Quality']

# 第一行：注意力权重柱状图
for idx, (result, quality_label) in enumerate(zip(results, quality_labels)):
    ax = plt.subplot(gs[0, idx])
    
    weights = result['attention_weights']
    stage_names = [f'S{i+1}' for i in range(4)]
    
    bars = ax.bar(stage_names, weights, color=colors, edgecolor='black', 
                  linewidth=1.5, alpha=0.85, width=0.6)
    
    # 标注数值和百分比
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        height = bar.get_height()
        # 数值标注
        ax.text(bar.get_x() + bar.get_width()/2., height + max(weights)*0.02,
               f'{weight:.3f}',
               ha='center', va='bottom', fontsize=9, weight='bold')
        
        # 百分比标注（如果权重足够大）
        percentage = weight / sum(weights) * 100
        if weight > 0.1:  # 只在权重>10%时显示百分比
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{percentage:.0f}%',
                   ha='center', va='center', fontsize=9, 
                   color='white', weight='bold')
    
    ax.set_ylim([0, max(weights) * 1.15])
    ax.set_ylabel('Attention Weight', fontsize=11, weight='bold')
    ax.set_xlabel('Swin Stage', fontsize=11, weight='bold')
    
    # 标题包含质量信息
    mos = result['gt_mos']
    pred = result['pred_score']
    ax.set_title(f'{quality_label}\nGT MOS: {mos:.2f}, Pred: {pred:.1f}', 
                fontsize=11, weight='bold')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 第二行：图片示例
image_dir = 'koniq-10k/test'
for idx, result in enumerate(results):
    ax = plt.subplot(gs[1, idx])
    
    # 读取并显示图片
    img_path = os.path.join(image_dir, result['image'])
    img = Image.open(img_path)
    
    # 调整图片大小以适应显示
    img.thumbnail((800, 800))
    
    ax.imshow(img)
    ax.axis('off')
    
    # 添加质量标签
    quality_texts = ['Low Quality', 'Medium Quality', 'High Quality']
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='black', linewidth=2, alpha=0.9)
    ax.text(0.5, 0.05, quality_texts[idx], 
           transform=ax.transAxes,
           ha='center', va='bottom',
           fontsize=12, weight='bold',
           bbox=bbox_props)

# 总标题
fig.suptitle('Channel Attention Analysis: Adaptive Feature Selection Based on Image Quality',
            fontsize=14, weight='bold', y=0.98)

# 添加说明文本
textstr = 'Key Findings: Low-quality images use balanced multi-scale features (all stages ~25%), ' + \
          'while high-quality images concentrate on high-level features (Stage 3: 99.6%+)'
fig.text(0.5, 0.02, textstr, ha='center', va='bottom', fontsize=10, 
        style='italic', wrap=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# 保存为PDF和PNG
output_dir = 'attention_visualizations'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'attention_comparison_combined.pdf'), 
           dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig(os.path.join(output_dir, 'attention_comparison_combined.png'), 
           dpi=300, bbox_inches='tight', pad_inches=0.1)

print(f"✓ Combined attention visualization saved to {output_dir}/")
print("  - attention_comparison_combined.pdf (for paper)")
print("  - attention_comparison_combined.png (preview)")

plt.close()

