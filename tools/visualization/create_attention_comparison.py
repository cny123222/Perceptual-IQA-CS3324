import sys
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
import csv

# 设置matplotlib参数 - 统一使用Times字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 读取结果
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'attention_visualization_results.json')
with open(json_path) as f:
    results = json.load(f)

# 读取MOS_zscore (0-100范围)
mos_zscore_map = {}
with open('koniq-10k/koniq10k_scores_and_distributions.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        mos_zscore_map[row['image_name']] = float(row['MOS_zscore'])

# 更新results中的gt_mos为MOS_zscore
for result in results:
    if result['image'] in mos_zscore_map:
        result['gt_mos'] = mos_zscore_map[result['image']]

# 创建大图
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2], hspace=0.15, wspace=0.3)

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
    
    # 标题包含质量信息 (只显示质量等级)
    ax.set_title(f'{quality_label}', 
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
    
    # 添加MOS信息（不显示质量标签文字）
    mos = result['gt_mos']
    pred = result['pred_score']
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='black', linewidth=2, alpha=0.9)
    ax.text(0.5, 0.05, f'GT: {mos:.1f}, Pred: {pred:.1f}', 
           transform=ax.transAxes,
           ha='center', va='bottom',
           fontsize=11, weight='bold',
           bbox=bbox_props)

# 不添加总标题和说明文本
plt.tight_layout()

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

