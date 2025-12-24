#!/usr/bin/env python3
"""
绘制SMART-IQA架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# 设置matplotlib参数
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 定义颜色方案
COLORS = {
    'swin': ['#E3F2FD', '#90CAF9', '#42A5F5', '#1976D2'],  # 蓝色渐变
    'afa': '#E8F5E9',  # 浅绿
    'afa_border': '#4CAF50',  # 绿色边框
    'attention': '#FFE0B2',  # 浅橙
    'attention_border': '#FF6B00',  # 橙色边框
    'hypernet': '#C8E6C9',  # 浅绿
    'hypernet_border': '#388E3C',  # 深绿边框
    'targetnet': '#E1BEE7',  # 浅紫
    'targetnet_border': '#7B1FA2',  # 紫色边框
    'arrow': '#424242',  # 深灰
    'arrow_weight': '#F44336',  # 红色（权重箭头）
}

def draw_main_architecture():
    """绘制主架构图"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 定义各模块的位置
    positions = {
        'input': (0.5, 4),
        'swin': (2, 4),
        'afa': (5, 4),
        'attention': (8, 4),
        'hypernet': (11, 2.5),
        'targetnet': (11, 5.5),
        'output': (14.5, 4),
    }
    
    # 1. 输入图像（简化表示）
    input_box = FancyBboxPatch((0.2, 3), 0.6, 2, 
                               boxstyle="round,pad=0.05", 
                               facecolor='white', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(0.5, 4, 'Input\nImage', ha='center', va='center', fontsize=10, weight='bold')
    
    # 2. Swin Transformer (4个stage)
    stage_heights = [0.8, 0.7, 0.6, 0.5]
    stage_widths = [0.6, 0.55, 0.5, 0.45]
    stage_y_start = 2.5
    
    for i, (h, w) in enumerate(zip(stage_heights, stage_widths)):
        y = stage_y_start + i * 1.1
        color = COLORS['swin'][i]
        # 画立方体效果（简化为矩形）
        stage_box = FancyBboxPatch((2 + i*0.15, y), w, h, 
                                   boxstyle="round,pad=0.02", 
                                   facecolor=color, 
                                   edgecolor='#1565C0', linewidth=1.5)
        ax.add_patch(stage_box)
        ax.text(2 + i*0.15 + w/2, y + h/2, f'Stage {i+1}', 
               ha='center', va='center', fontsize=8, weight='bold')
    
    # Swin标题
    ax.text(2.5, 6.8, 'Swin Transformer', ha='center', fontsize=11, weight='bold')
    ax.text(2.5, 6.5, 'Backbone', ha='center', fontsize=9, style='italic')
    
    # 3. AFA Module（简化框）
    afa_box = FancyBboxPatch((4.5, 3), 1.2, 2, 
                            boxstyle="round,pad=0.05", 
                            facecolor=COLORS['afa'], 
                            edgecolor=COLORS['afa_border'], linewidth=2)
    ax.add_patch(afa_box)
    ax.text(5.1, 4.7, 'AFA', ha='center', fontsize=10, weight='bold')
    ax.text(5.1, 4.4, 'Module', ha='center', fontsize=9)
    ax.text(5.1, 3.8, '(Adaptive', ha='center', fontsize=7)
    ax.text(5.1, 3.5, 'Feature', ha='center', fontsize=7)
    ax.text(5.1, 3.2, 'Aggregation)', ha='center', fontsize=7)
    
    # 4. Channel Attention（展开画！）⭐
    # 外框 - 醒目的橙色边框
    attention_box = FancyBboxPatch((7, 2.5), 2.5, 3, 
                                   boxstyle="round,pad=0.08", 
                                   facecolor=COLORS['attention'], 
                                   edgecolor=COLORS['attention_border'], 
                                   linewidth=3)
    ax.add_patch(attention_box)
    
    # 星标
    star = ax.text(7.15, 5.3, '⭐', fontsize=14)
    
    # 标题
    ax.text(8.25, 5.3, 'Channel Attention', ha='center', fontsize=10, weight='bold')
    ax.text(8.25, 5.05, 'Fusion', ha='center', fontsize=10, weight='bold')
    
    # Attention内部结构
    # Stage 4输入
    ax.text(8.25, 4.6, 'Stage 4 → GAP', ha='center', fontsize=8)
    ax.text(8.25, 4.35, '↓', ha='center', fontsize=10)
    
    # Attention Network
    att_net_box = FancyBboxPatch((7.5, 3.7), 1.5, 0.5, 
                                boxstyle="round,pad=0.02", 
                                facecolor='#F5F5F5', 
                                edgecolor='#9E9E9E', linewidth=1)
    ax.add_patch(att_net_box)
    ax.text(8.25, 4, 'Attention Net', ha='center', fontsize=7, weight='bold')
    ax.text(8.25, 3.85, 'FC→ReLU→FC→Softmax', ha='center', fontsize=6)
    
    ax.text(8.25, 3.5, '↓', ha='center', fontsize=10)
    
    # 权重
    ax.text(8.25, 3.3, 'Weights: [w₁  w₂  w₃]', ha='center', fontsize=7, weight='bold')
    ax.text(8.25, 3.05, '↓', ha='center', fontsize=10)
    ax.text(8.25, 2.85, 'Apply to f₁, f₂, f₃', ha='center', fontsize=7, style='italic')
    
    # 5. Concat节点
    concat_circle = Circle((10, 4), 0.2, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(concat_circle)
    ax.text(10, 4, '⊕', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(10, 3.4, 'Concat', ha='center', fontsize=7)
    
    # 6. HyperNet
    hypernet_box = FancyBboxPatch((10.5, 1.5), 1.2, 1.5, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=COLORS['hypernet'], 
                                 edgecolor=COLORS['hypernet_border'], linewidth=2)
    ax.add_patch(hypernet_box)
    ax.text(11.1, 2.7, 'HyperNet', ha='center', fontsize=9, weight='bold')
    ax.text(11.1, 2.45, 'Conv 1×1', ha='center', fontsize=7)
    ax.text(11.1, 2.25, '×3 layers', ha='center', fontsize=7)
    ax.text(11.1, 2.0, 'Generate', ha='center', fontsize=6, style='italic')
    ax.text(11.1, 1.8, 'Weights', ha='center', fontsize=6, style='italic')
    
    # 7. TargetNet
    targetnet_box = FancyBboxPatch((10.5, 4.5), 1.2, 2, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=COLORS['targetnet'], 
                                  edgecolor=COLORS['targetnet_border'], linewidth=2)
    ax.add_patch(targetnet_box)
    ax.text(11.1, 6.2, 'TargetNet', ha='center', fontsize=9, weight='bold')
    
    # TargetNet内部的4个FC层
    fc_y = [5.9, 5.5, 5.1, 4.7]
    for i, y in enumerate(fc_y):
        fc_box = FancyBboxPatch((10.65, y-0.15), 0.9, 0.25, 
                               facecolor='#F3E5F5', 
                               edgecolor='#9C27B0', linewidth=0.5)
        ax.add_patch(fc_box)
        ax.text(11.1, y, f'FC{i+1}', ha='center', fontsize=7)
    
    # 8. 输出
    output_box = FancyBboxPatch((14.2, 3), 0.6, 2, 
                               boxstyle="round,pad=0.05", 
                               facecolor='white', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(14.5, 4, 'Quality\nScore', ha='center', va='center', fontsize=10, weight='bold')
    
    # 绘制箭头连接
    # Input → Swin
    ax.annotate('', xy=(2, 4), xytext=(0.8, 4),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['arrow']))
    
    # Swin → AFA (Stage 1,2,3)
    for i in range(3):
        y = 2.5 + i * 1.1 + stage_heights[i]/2
        ax.annotate('', xy=(4.5, 3.5 + i*0.5), xytext=(2.6, y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['arrow']))
    
    # AFA → Attention
    ax.annotate('', xy=(7, 4), xytext=(5.7, 4),
               arrowprops=dict(arrowstyle='->', lw=2, color='#4CAF50'))
    
    # Stage 4 → Attention
    y4 = 2.5 + 3 * 1.1 + stage_heights[3]/2
    ax.annotate('', xy=(7.5, 4.6), xytext=(2.6, y4),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['attention_border'],
                             connectionstyle="arc3,rad=0.3"))
    ax.text(5, 6.5, 'Semantic\nFeatures', fontsize=7, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE0B2', alpha=0.7))
    
    # Stage 4 → HyperNet
    ax.annotate('', xy=(10.5, 2.25), xytext=(2.6, y4),
               arrowprops=dict(arrowstyle='->', lw=2, color='#388E3C',
                             connectionstyle="arc3,rad=-0.3"))
    
    # Attention → Concat (加权箭头)
    ax.annotate('', xy=(9.8, 4), xytext=(9.5, 4),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['arrow_weight']))
    ax.text(9.65, 4.3, '×w₁,w₂,w₃', fontsize=7, ha='center', 
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFCDD2', alpha=0.8),
           weight='bold')
    
    # Concat → TargetNet
    ax.annotate('', xy=(10.5, 5.5), xytext=(10.2, 4.1),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['arrow']))
    
    # HyperNet → TargetNet (动态权重)
    for i in range(4):
        y_target = 5.9 - i * 0.4
        ax.annotate('', xy=(10.65, y_target), xytext=(11.7, 2.5),
                   arrowprops=dict(arrowstyle='->', lw=1, color='#388E3C',
                                 linestyle='--', alpha=0.6))
    
    ax.text(11.9, 4, 'Dynamic\nWeights', fontsize=6, ha='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#C8E6C9', alpha=0.7))
    
    # TargetNet → Output
    ax.annotate('', xy=(14.2, 4), xytext=(11.7, 5.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['arrow']))
    
    # 添加图例
    legend_y = 0.5
    ax.text(0.5, legend_y + 0.3, 'Legend:', fontsize=9, weight='bold')
    
    # 图例项
    legend_items = [
        (COLORS['swin'][2], 'Swin Stage'),
        (COLORS['afa'], 'AFA Module'),
        (COLORS['attention'], 'Attention ⭐'),
        (COLORS['hypernet'], 'HyperNet'),
        (COLORS['targetnet'], 'TargetNet'),
    ]
    
    for i, (color, label) in enumerate(legend_items):
        x = 1 + i * 1.8
        legend_box = FancyBboxPatch((x, legend_y-0.1), 0.3, 0.2,
                                   facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(legend_box)
        ax.text(x + 0.4, legend_y, label, fontsize=7, va='center')
    
    # 标题
    ax.text(8, 7.5, 'SMART-IQA Architecture', fontsize=14, weight='bold', ha='center')
    ax.text(8, 7.2, 'Swin Multi-scale Attention-guided Regression Transformer for Image Quality Assessment', 
           fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    return fig

def draw_afa_module():
    """绘制AFA模块详细图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 标题
    ax.text(5, 5.5, 'Adaptive Feature Aggregation (AFA) Module', 
           fontsize=12, weight='bold', ha='center')
    
    # 3个stage并排
    stage_info = [
        ('Stage 1', '96×56×56', '96×7×7', 2),
        ('Stage 2', '192×28×28', '192×7×7', 5),
        ('Stage 3', '512×14×14', '512×7×7', 8),
    ]
    
    colors = ['#E3F2FD', '#90CAF9', '#42A5F5']
    
    for i, (stage_name, input_dim, output_dim, x) in enumerate(stage_info):
        # 输入立方体
        input_box = FancyBboxPatch((x-0.3, 4.2), 0.6, 0.5,
                                  facecolor=colors[i],
                                  edgecolor='#1565C0', linewidth=1.5)
        ax.add_patch(input_box)
        ax.text(x, 4.45, stage_name, ha='center', fontsize=9, weight='bold')
        ax.text(x, 4.9, input_dim, ha='center', fontsize=7)
        
        # 箭头
        ax.annotate('', xy=(x, 3.9), xytext=(x, 4.2),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        # 处理模块
        module_y = [3.5, 3.0, 2.5]
        module_labels = ['Adaptive Pool\n(7×7)', 'Conv 1×1', 'BatchNorm\n+ ReLU']
        
        for j, (y, label) in enumerate(zip(module_y, module_labels)):
            module_box = FancyBboxPatch((x-0.35, y-0.15), 0.7, 0.3,
                                       boxstyle="round,pad=0.02",
                                       facecolor='#E8F5E9',
                                       edgecolor='#4CAF50', linewidth=1)
            ax.add_patch(module_box)
            ax.text(x, y, label, ha='center', va='center', fontsize=7)
            
            if j < len(module_y) - 1:
                ax.annotate('', xy=(x, module_y[j+1]+0.15), xytext=(x, y-0.15),
                           arrowprops=dict(arrowstyle='->', lw=1, color='black'))
        
        # 输出立方体
        output_box = FancyBboxPatch((x-0.3, 1.5), 0.6, 0.5,
                                   facecolor=colors[i],
                                   edgecolor='#1565C0', linewidth=1.5)
        ax.add_patch(output_box)
        ax.text(x, 1.75, f'f{i+1}', ha='center', fontsize=9, weight='bold')
        ax.text(x, 1.3, output_dim, ha='center', fontsize=7)
        
        # 最后的箭头
        ax.annotate('', xy=(x, 2), xytext=(x, 2.35),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # 底部说明
    ax.text(5, 0.7, 'All features unified to 7×7 spatial resolution', 
           ha='center', fontsize=8, style='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.7))
    
    ax.text(5, 0.3, 'Output: [f₁, f₂, f₃] → Input to Channel Attention', 
           ha='center', fontsize=8, weight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE0B2', alpha=0.7))
    
    plt.tight_layout()
    return fig

# 生成图片
print("正在生成主架构图...")
fig1 = draw_main_architecture()
fig1.savefig('paper_figures/main_architecture.pdf', dpi=300, bbox_inches='tight')
fig1.savefig('paper_figures/main_architecture.png', dpi=300, bbox_inches='tight')
print("✓ 主架构图已保存: paper_figures/main_architecture.pdf")

print("\n正在生成AFA模块详细图...")
fig2 = draw_afa_module()
fig2.savefig('paper_figures/afa_module_detail.pdf', dpi=300, bbox_inches='tight')
fig2.savefig('paper_figures/afa_module_detail.png', dpi=300, bbox_inches='tight')
print("✓ AFA模块详细图已保存: paper_figures/afa_module_detail.pdf")

print("\n✅ 所有架构图已生成完成！")
plt.close('all')

