#!/usr/bin/env python3
"""
生成SMART-IQA论文的所有可视化图表
包括: 注意力热力图、跨数据集对比、SOTA雷达图等
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import matplotlib.patches as mpatches
import os

# 设置中文字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建输出目录
output_dir = 'paper_figures'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("开始生成SMART-IQA论文可视化图表")
print("=" * 80)

# ============================================================================
# 图1: 跨数据集性能热力图 (Cross-Dataset Performance Heatmap)
# ============================================================================
def generate_cross_dataset_heatmap():
    print("\n[1/6] 生成跨数据集性能热力图...")
    
    # 数据：方法 × 数据集 (SRCC)
    methods = ['HyperIQA\n(ResNet50)', 'SMART-IQA\n(Ours)']
    datasets = ['KonIQ-10k\n(Train)', 'SPAQ\n(Phone)', 'KADID-10K\n(Synthetic)', 'AGIQA-3K\n(AI-Gen)']
    
    data = np.array([
        [0.9060, 0.8490, 0.4848, 0.6627],  # HyperIQA
        [0.9378, 0.8698, 0.5412, 0.6484],  # SMART-IQA (Ours)
    ])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # 创建热力图
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)
    
    # 设置刻度
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_yticklabels(methods, fontsize=12, weight='bold')
    
    # 在每个单元格中显示数值
    for i in range(len(methods)):
        for j in range(len(datasets)):
            # 选择文字颜色（深色背景用白色，浅色背景用黑色）
            text_color = 'white' if data[i, j] < 0.65 else 'black'
            weight = 'bold' if i == 1 else 'normal'  # Ours用粗体
            
            text = ax.text(j, i, f'{data[i, j]:.4f}',
                          ha="center", va="center", 
                          color=text_color, fontsize=12, weight=weight)
            
            # 如果是我们的结果且优于HyperIQA，添加向上箭头
            if i == 1 and data[i, j] > data[0, j]:
                ax.text(j, i - 0.35, '↑', ha="center", va="center", 
                       color='darkgreen', fontsize=16, weight='bold')
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('SRCC', rotation=270, labelpad=20, fontsize=12)
    
    # 标题
    ax.set_title('Cross-Dataset Generalization Performance (SRCC)', 
                 fontsize=14, weight='bold', pad=15)
    
    # 添加平均值列
    avg_train = [data[i, 0] for i in range(len(methods))]
    avg_cross = [np.mean(data[i, 1:]) for i in range(len(methods))]
    
    # 在图下方添加统计信息
    info_text = (
        f"In-domain: HyperIQA={avg_train[0]:.4f}, Ours={avg_train[1]:.4f} (+{(avg_train[1]-avg_train[0])*100:.2f}%)\n"
        f"Avg Cross-domain: HyperIQA={avg_cross[0]:.4f}, Ours={avg_cross[1]:.4f} (+{(avg_cross[1]-avg_cross[0])*100:.2f}%)"
    )
    plt.figtext(0.5, -0.05, info_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cross_dataset_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/cross_dataset_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {output_dir}/cross_dataset_heatmap.pdf/.png")

# ============================================================================
# 图2: SOTA方法雷达图 (SOTA Radar Chart)
# ============================================================================
def generate_sota_radar_chart():
    print("\n[2/6] 生成SOTA方法雷达图...")
    
    # 6个维度的数据
    categories = ['KonIQ-10k\nSRCC', 'Cross-domain\nAverage', 
                  'Parameter\nEfficiency', 'Inference\nSpeed',
                  'Training\nEfficiency', 'Robustness']
    
    # 归一化数据 (0-1)
    # Parameter Efficiency = 1 / (Params in millions / 100)
    # Inference Speed = FPS / 100
    # Training Efficiency = 1 / (hours / 10)
    methods = {
        'SMART-IQA\n(Ours)': [
            0.9378 / 0.95,          # KonIQ SRCC
            0.6865 / 0.75,          # Cross-domain avg
            1.0 - (88 / 150),       # Param efficiency (lower is better, invert)
            22 / 100,               # Inference speed (FPS)
            1.0 - (1.7 / 10),       # Training efficiency (lower hours is better)
            1.0 - (0.002 / 0.01)    # Robustness (lower std is better)
        ],
        'HyperIQA': [
            0.906 / 0.95,
            0.6655 / 0.75,
            1.0 - (25 / 150),
            100 / 100,
            1.0 - (2.0 / 10),
            1.0 - (0.003 / 0.01)
        ],
        'MUSIQ': [
            0.915 / 0.95,
            0.70 / 0.75,
            1.0 - (150 / 150),
            15 / 100,
            1.0 - (8.0 / 10),
            1.0 - (0.005 / 0.01)
        ],
        'MANIQA': [
            0.920 / 0.95,
            0.68 / 0.75,
            1.0 - (46 / 150),
            40 / 100,
            1.0 - (3.0 / 10),
            1.0 - (0.004 / 0.01)
        ],
    }
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 颜色
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']
    
    # 绘制每个方法
    for i, (method, color) in enumerate(zip(methods.keys(), colors)):
        values = methods[method] + [methods[method][0]]  # 闭合
        
        # 绘制线和填充
        ax.plot(angles, values, 'o-', linewidth=2.5, label=method, 
                color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # 设置刻度
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, weight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=11, 
              frameon=True, shadow=True)
    
    # 标题
    plt.title('Multi-dimensional Comparison with State-of-the-Art Methods', 
              size=14, weight='bold', y=1.12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sota_radar_chart.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/sota_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {output_dir}/sota_radar_chart.pdf/.png")

# ============================================================================
# 图3: 消融实验瀑布图 (Progressive Ablation Waterfall)
# ============================================================================
def generate_ablation_waterfall():
    print("\n[3/6] 生成消融实验瀑布图...")
    
    # 数据
    configs = ['ResNet50\nBaseline', 'Swin-Base\nOnly', 'Multi-Scale\nFusion', 
               'Attention\nMechanism', 'Full Model']
    srcc_values = [0.9070, 0.9338, 0.9353, 0.9378, 0.9378]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 计算增量
    baseline = srcc_values[0]
    cumulative = [0]
    gains = []
    
    for i in range(1, len(srcc_values)):
        gain = srcc_values[i] - (baseline if i == 1 else srcc_values[i-1])
        gains.append(gain)
        cumulative.append(srcc_values[i] - baseline)
    
    # 颜色方案
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#4CAF50']
    
    # 绘制柱状图
    bars = []
    x_pos = np.arange(len(configs))
    
    # Baseline
    bar = ax.bar(0, srcc_values[0], color=colors[0], alpha=0.8, 
                 edgecolor='black', linewidth=1.5)
    bars.append(bar)
    ax.text(0, srcc_values[0] + 0.002, f'{srcc_values[0]:.4f}', 
            ha='center', va='bottom', fontsize=11, weight='bold')
    
    # 增量柱
    for i in range(len(gains)):
        start = srcc_values[i]
        height = gains[i]
        bar = ax.bar(i + 1, height, bottom=start, color=colors[i + 1], 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        bars.append(bar)
        
        # 标注增量和累计值
        ax.text(i + 1, start + height + 0.002, f'{srcc_values[i+1]:.4f}', 
                ha='center', va='bottom', fontsize=11, weight='bold')
        ax.text(i + 1, start + height/2, f'+{height:.4f}\n({height/0.0308*100:.1f}%)', 
                ha='center', va='center', fontsize=9, color='white', weight='bold')
        
        # 连接线
        if i < len(gains) - 1:
            ax.plot([i + 0.4, i + 1 - 0.4], [srcc_values[i+1], srcc_values[i+1]], 
                    'k--', linewidth=1.5, alpha=0.5)
    
    # 设置
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, fontsize=11, weight='bold')
    ax.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
    ax.set_title('Progressive Ablation Study: Component Contributions', 
                 fontsize=14, weight='bold', pad=15)
    ax.set_ylim([0.90, 0.945])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加总提升标注
    ax.annotate('', xy=(4, 0.9378), xytext=(0, 0.9070),
                arrowprops=dict(arrowstyle='<->', lw=2.5, color='red'))
    ax.text(2, 0.920, f'Total Gain: +{0.0308:.4f} (+3.08%)', 
            ha='center', fontsize=12, weight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_waterfall.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/ablation_waterfall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {output_dir}/ablation_waterfall.pdf/.png")

# ============================================================================
# 图4: 模型大小对比散点图 (Model Size Comparison)
# ============================================================================
def generate_model_size_scatter():
    print("\n[4/6] 生成模型大小对比散点图...")
    
    # 数据: (参数量M, SRCC, 模型名称)
    models = [
        (25, 0.9060, 'HyperIQA\n(ResNet50)', 'red'),
        (28, 0.9249, 'SMART-IQA\nTiny', 'blue'),
        (50, 0.9338, 'SMART-IQA\nSmall', 'green'),
        (88, 0.9378, 'SMART-IQA\nBase', 'purple'),
        (46, 0.9200, 'MANIQA\n(ViT-S)', 'orange'),
        (150, 0.9150, 'MUSIQ\n(ViT)', 'brown'),
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制散点
    for params, srcc, name, color in models:
        # 特殊标记我们的方法
        if 'SMART-IQA' in name:
            marker = 'D'  # 菱形
            size = 300
            edgecolor = 'black'
            linewidth = 2.5
            alpha = 1.0
        else:
            marker = 'o'
            size = 200
            edgecolor = 'black'
            linewidth = 1.5
            alpha = 0.7
        
        ax.scatter(params, srcc, s=size, c=color, marker=marker, 
                   alpha=alpha, edgecolors=edgecolor, linewidths=linewidth,
                   label=name)
        
        # 标注
        offset_x = -8 if 'SMART-IQA' in name else 5
        offset_y = 0.0005 if params != 88 else -0.0015
        ax.annotate(name, (params, srcc), 
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=10, weight='bold' if 'SMART-IQA' in name else 'normal',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='yellow' if 'SMART-IQA' in name else 'white',
                             alpha=0.7, edgecolor='black'))
    
    # Pareto前沿线（高效模型）
    efficient_models = [(25, 0.9060), (28, 0.9249), (50, 0.9338), (88, 0.9378)]
    efficient_models_sorted = sorted(efficient_models, key=lambda x: x[0])
    ax.plot([m[0] for m in efficient_models_sorted], 
            [m[1] for m in efficient_models_sorted],
            'g--', linewidth=2, alpha=0.5, label='Efficiency Frontier')
    
    # 设置
    ax.set_xlabel('Parameters (Millions)', fontsize=13, weight='bold')
    ax.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
    ax.set_title('Performance-Efficiency Trade-off: Model Size vs Accuracy', 
                 fontsize=14, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([20, 160])
    ax.set_ylim([0.905, 0.940])
    
    # 添加注释区域
    ax.axhspan(0.935, 0.940, alpha=0.1, color='green', label='High Performance Zone')
    ax.axvspan(20, 50, alpha=0.05, color='blue', label='Efficient Zone')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_size_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/model_size_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {output_dir}/model_size_scatter.pdf/.png")

# ============================================================================
# 图5: 学习率敏感度曲线 (Learning Rate Sensitivity)
# ============================================================================
def generate_lr_sensitivity():
    print("\n[5/6] 生成学习率敏感度曲线...")
    
    # 数据
    lr_values = [1e-7, 5e-7, 1e-6, 3e-6, 5e-6]
    srcc_values = [0.9375, 0.9378, 0.9374, 0.9364, 0.9354]
    epochs = [14, 10, 10, 5, 5]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 图1: SRCC vs Learning Rate
    ax1.plot(lr_values, srcc_values, 'o-', linewidth=2.5, markersize=10, 
             color='#4ECDC4', markerfacecolor='#FF6B6B', 
             markeredgecolor='black', markeredgewidth=2)
    
    # 标记最优点
    best_idx = srcc_values.index(max(srcc_values))
    ax1.scatter([lr_values[best_idx]], [srcc_values[best_idx]], 
                s=500, c='gold', marker='*', edgecolors='black', 
                linewidths=2, zorder=5, label='Optimal LR')
    
    # 标注数值
    for i, (lr, srcc) in enumerate(zip(lr_values, srcc_values)):
        offset_y = 0.0002 if i != best_idx else 0.0005
        ax1.annotate(f'{srcc:.4f}', (lr, srcc), 
                     xytext=(0, offset_y), textcoords='offset points',
                     ha='center', fontsize=10, weight='bold')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate', fontsize=13, weight='bold')
    ax1.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
    ax1.set_title('Learning Rate Sensitivity Analysis', fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # 添加最优区间阴影
    ax1.axvspan(3e-7, 1e-6, alpha=0.1, color='green', label='Optimal Range')
    
    # 图2: Training Epochs vs Learning Rate
    ax2.bar(range(len(lr_values)), epochs, color='#95E1D3', 
            edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_xticks(range(len(lr_values)))
    ax2.set_xticklabels([f'{lr:.0e}' for lr in lr_values], fontsize=10)
    ax2.set_xlabel('Learning Rate', fontsize=13, weight='bold')
    ax2.set_ylabel('Epochs to Converge', fontsize=13, weight='bold')
    ax2.set_title('Training Efficiency vs Learning Rate', fontsize=13, weight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 标注数值
    for i, (lr, epoch) in enumerate(zip(lr_values, epochs)):
        ax2.text(i, epoch + 0.3, str(epoch), ha='center', 
                 fontsize=11, weight='bold')
    
    # 添加说明文字
    info_text = (
        "Key Finding: Swin Transformer requires 200× smaller LR than ResNet50\n"
        "Optimal LR: 5e-7 (ResNet50: 1e-4)"
    )
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lr_sensitivity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/lr_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {output_dir}/lr_sensitivity.pdf/.png")

# ============================================================================
# 图6: 消融实验饼图 (Component Contribution Pie)
# ============================================================================
def generate_contribution_pie():
    print("\n[6/6] 生成组件贡献饼图...")
    
    # 数据
    components = ['Swin Transformer', 'Attention Mechanism', 'Multi-Scale Fusion']
    contributions = [87, 8, 5]  # 百分比
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    explode = (0.1, 0, 0)  # 突出显示Swin
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(contributions, explode=explode, labels=components,
                                        colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 13, 'weight': 'bold'},
                                        pctdistance=0.85, labeldistance=1.15)
    
    # 美化百分比文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_weight('bold')
    
    # 标题
    ax.set_title('Component Contribution to Overall Improvement (+3.08% SRCC)', 
                 fontsize=14, weight='bold', pad=20)
    
    # 添加图例
    legend_labels = [f'{comp}: +{contributions[i]/100*0.0308:.4f} SRCC' 
                     for i, comp in enumerate(components)]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
    
    # 添加中心文字
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.text(0, 0, 'Total\n+3.08%', ha='center', va='center', 
            fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/contribution_pie.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/contribution_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {output_dir}/contribution_pie.pdf/.png")

# ============================================================================
# 主函数
# ============================================================================
def main():
    # 生成所有图表
    generate_cross_dataset_heatmap()
    generate_sota_radar_chart()
    generate_ablation_waterfall()
    generate_model_size_scatter()
    generate_lr_sensitivity()
    generate_contribution_pie()
    
    print("\n" + "=" * 80)
    print("✅ 所有图表生成完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}/")
    print("\n生成的文件:")
    print("  1. cross_dataset_heatmap.pdf/.png    - 跨数据集性能热力图")
    print("  2. sota_radar_chart.pdf/.png         - SOTA方法雷达图")
    print("  3. ablation_waterfall.pdf/.png       - 消融实验瀑布图")
    print("  4. model_size_scatter.pdf/.png       - 模型大小对比散点图")
    print("  5. lr_sensitivity.pdf/.png           - 学习率敏感度曲线")
    print("  6. contribution_pie.pdf/.png         - 组件贡献饼图")
    print("\n这些图表可以直接用于论文！")
    print("=" * 80)

if __name__ == '__main__':
    main()

