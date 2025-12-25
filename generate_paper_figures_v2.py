#!/usr/bin/env python3
"""
生成SMART-IQA论文图表 - 使用已知实验数据
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 使用Times字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

output_dir = 'paper_figures'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("生成SMART-IQA论文图表（使用已知数据）")
print("=" * 80)

# ============================================================================
# 图1: 学习率敏感度（扩大y轴范围）
# ============================================================================
print("\n[1/4] 生成学习率敏感度曲线...")

lr_data = {
    5e-6: {'srcc': 0.9354, 'plcc': 0.9448, 'epochs': 5},
    3e-6: {'srcc': 0.9364, 'plcc': 0.9464, 'epochs': 5},
    1e-6: {'srcc': 0.9374, 'plcc': 0.9485, 'epochs': 10},
    5e-7: {'srcc': 0.9378, 'plcc': 0.9485, 'epochs': 10},  # Best
    1e-7: {'srcc': 0.9375, 'plcc': 0.9488, 'epochs': 14},
}

lr_values = [5e-6, 3e-6, 1e-6, 5e-7, 1e-7]
srcc_values = [lr_data[lr]['srcc'] for lr in lr_values]
plcc_values = [lr_data[lr]['plcc'] for lr in lr_values]
epochs_values = [lr_data[lr]['epochs'] for lr in lr_values]

# 绘制单图，同时显示SRCC和PLCC，适中尺寸
fig, ax1 = plt.subplots(1, 1, figsize=(5.5, 3.8))

# SRCC曲线（左Y轴）
color1 = '#FF6B6B'
ax1.plot(lr_values, srcc_values, 'o-', linewidth=2.5, markersize=8, 
         color=color1, markerfacecolor=color1, 
         markeredgecolor='black', markeredgewidth=1.5, label='SRCC')

best_srcc_idx = srcc_values.index(max(srcc_values))
ax1.scatter([lr_values[best_srcc_idx]], [srcc_values[best_srcc_idx]], 
            s=300, c='gold', marker='*', edgecolors='black', 
            linewidths=2, zorder=5)

ax1.set_xscale('log')
ax1.set_xlabel('Learning Rate', fontsize=11, weight='bold')
ax1.set_ylabel('SRCC', fontsize=11, weight='bold', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim([0.930, 0.950])  # 统一范围
ax1.grid(True, alpha=0.3, linestyle='--', which='both')  # 添加网格线

# PLCC曲线（右Y轴）
ax2 = ax1.twinx()
color2 = '#4ECDC4'
ax2.plot(lr_values, plcc_values, 's-', linewidth=2.5, markersize=8,
         color=color2, markerfacecolor=color2,
         markeredgecolor='black', markeredgewidth=1.5, label='PLCC')

best_plcc_idx = plcc_values.index(max(plcc_values))
ax2.scatter([lr_values[best_plcc_idx]], [plcc_values[best_plcc_idx]], 
            s=300, c='gold', marker='*', edgecolors='black', 
            linewidths=2, zorder=5)

ax2.set_ylabel('PLCC', fontsize=11, weight='bold', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim([0.930, 0.950])  # 统一范围

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/lr_sensitivity_final.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/lr_sensitivity_final.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ 已保存: {output_dir}/lr_sensitivity_final.pdf/.png")

# ============================================================================
# 图2: 不同模型大小对比
# ============================================================================
print("\n[2/4] 生成模型大小对比图...")

models = ['HyperIQA\n(ResNet50)', 'SMART-IQA\nTiny', 'SMART-IQA\nSmall', 'SMART-IQA\nBase']
params = [25, 28, 50, 88]
srcc = [0.9070, 0.9249, 0.9338, 0.9378]
plcc = [0.9180, 0.9360, 0.9455, 0.9485]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 图1: SRCC柱状图
colors = ['red', 'skyblue', 'lightgreen', 'gold']
bars1 = ax1.bar(range(len(models)), srcc, color=colors, 
                edgecolor='black', linewidth=2, alpha=0.8)

ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, fontsize=11, weight='bold')
ax1.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
ax1.set_ylim([0.900, 0.945])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for i, (model, s) in enumerate(zip(models, srcc)):
    ax1.text(i, s + 0.002, f'{s:.4f}', ha='center', fontsize=11, weight='bold')
    if i > 0:
        improvement = (s - srcc[0]) / srcc[0] * 100
        ax1.text(i, s - 0.015, f'+{improvement:.2f}%', ha='center', 
                fontsize=9, color='green', weight='bold')

ax1.axhline(y=srcc[0], color='red', linestyle='--', linewidth=2, alpha=0.5)

# 图2: 散点图
for i, (param, s, model) in enumerate(zip(params, srcc, models)):
    if 'SMART-IQA' in model:
        marker = 'D'
        size = 300
        edgewidth = 2.5
    else:
        marker = 'o'
        size = 200
        edgewidth = 1.5
    
    ax2.scatter(param, s, s=size, c=colors[i], marker=marker, 
               alpha=0.8, edgecolors='black', linewidths=edgewidth)
    
    # 标注模型名称，避免与点重合
    if 'Tiny' in model:
        label = 'Swin-Tiny'
        offset_x = 50
        offset_y = -10
    elif 'Small' in model:
        label = 'Swin-Small'
        offset_x = 0
        offset_y = 25
    elif 'Base' in model:
        label = 'Swin-Base'
        offset_x = 8
        offset_y = -30
    else:
        label = 'HyperIQA\n(ResNet50)'
        offset_x = 50
        offset_y = -10
    
    ax2.annotate(label, (param, s), 
                xytext=(offset_x, offset_y), textcoords='offset points',
                fontsize=9, weight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='yellow' if 'Base' in model else 'white',
                         alpha=0.7, edgecolor='black'))

ax2.plot(params, srcc, 'g--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Parameters (Millions)', fontsize=13, weight='bold')
ax2.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim([20, 95])
ax2.set_ylim([0.900, 0.945])

plt.tight_layout()
plt.savefig(f'{output_dir}/model_size_final.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/model_size_final.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ 已保存: {output_dir}/model_size_final.pdf/.png")

# ============================================================================
# 图3: 消融实验对比（使用最终SRCC值）
# ============================================================================
print("\n[3/4] 生成消融实验对比图...")

ablation_data = {
    'HyperIQA\n(ResNet50)': {'srcc': 0.9070, 'color': '#FF6B6B'},
    'Swin Only': {'srcc': 0.9338, 'color': '#4ECDC4'},
    'Swin+\nMulti-Scale': {'srcc': 0.9353, 'color': '#95E1D3'},
    'Full Model\n(+Attention)': {'srcc': 0.9378, 'color': '#FFD93D'},
}

names = list(ablation_data.keys())
srccs = [ablation_data[name]['srcc'] for name in names]
colors_abl = [ablation_data[name]['color'] for name in names]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(range(len(names)), srccs, color=colors_abl,
              edgecolor='black', linewidth=2, alpha=0.8)

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=11, weight='bold')
ax.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
ax.set_title('Ablation Study: Component Contributions', fontsize=14, weight='bold')
ax.set_ylim([0.900, 0.945])
ax.grid(axis='y', alpha=0.3, linestyle='--')

for i, (name, srcc_val) in enumerate(zip(names, srccs)):
    ax.text(i, srcc_val + 0.002, f'{srcc_val:.4f}', ha='center', 
            fontsize=11, weight='bold')
    if i > 0:
        gain = srcc_val - srccs[0]
        total_gain = srccs[-1] - srccs[0]
        contribution = (gain / total_gain) * 100 if i == 1 else ((srcc_val - srccs[i-1]) / total_gain) * 100
        if i == 1:
            ax.text(i, srcc_val - 0.015, f'+{gain:.4f}\n({contribution:.0f}%)', 
                   ha='center', fontsize=9, color='darkgreen', weight='bold')
        else:
            delta = srcc_val - srccs[i-1]
            ax.text(i, srcc_val - 0.015, f'+{delta:.4f}\n({contribution:.0f}%)', 
                   ha='center', fontsize=9, color='darkgreen', weight='bold')

ax.axhline(y=srccs[0], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
ax.legend(fontsize=11)

total_improvement = srccs[-1] - srccs[0]
plt.figtext(0.5, 0.02, f'Total Improvement: +{total_improvement:.4f} (+{total_improvement/srccs[0]*100:.2f}%)', 
            ha='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{output_dir}/ablation_comparison_final.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/ablation_comparison_final.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ 已保存: {output_dir}/ablation_comparison_final.pdf/.png")

# ============================================================================
# 图4: 损失函数对比
# ============================================================================
print("\n[4/4] 生成损失函数对比图...")

loss_data = {
    'L1 (MAE)': {'srcc': 0.9375, 'plcc': 0.9488, 'rank': 1},
    'L2 (MSE)': {'srcc': 0.9373, 'plcc': 0.9469, 'rank': 2},
    'Pairwise\nFidelity': {'srcc': 0.9315, 'plcc': 0.9373, 'rank': 3},
    'SRCC Loss': {'srcc': 0.9313, 'plcc': 0.9416, 'rank': 4},
    'Pairwise\nRanking': {'srcc': 0.9292, 'plcc': 0.9249, 'rank': 5},
}

loss_names = list(loss_data.keys())
loss_srccs = [loss_data[name]['srcc'] for name in loss_names]
loss_plccs = [loss_data[name]['plcc'] for name in loss_names]
ranks = [loss_data[name]['rank'] for name in loss_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 图1: SRCC对比
colors_loss = ['gold', 'silver', '#CD7F32', 'skyblue', 'lightcoral']
bars1 = ax1.bar(range(len(loss_names)), loss_srccs, color=colors_loss,
               edgecolor='black', linewidth=2, alpha=0.8)

ax1.set_xticks(range(len(loss_names)))
ax1.set_xticklabels(loss_names, fontsize=10, weight='bold')
ax1.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
ax1.set_ylim([0.925, 0.940])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for i, (name, srcc_val, rank) in enumerate(zip(loss_names, loss_srccs, ranks)):
    ax1.text(i, srcc_val + 0.0008, f'{srcc_val:.4f}', ha='center', 
            fontsize=10, weight='bold')

# 图2: SRCC vs PLCC散点图
for i, (name, srcc_val, plcc_val, rank) in enumerate(zip(loss_names, loss_srccs, loss_plccs, ranks)):
    size = 400 if rank <= 3 else 250
    ax2.scatter(srcc_val, plcc_val, s=size, c=colors_loss[i], 
               marker='o' if rank <= 2 else 's',
               alpha=0.8, edgecolors='black', linewidths=2)
    
    if i == 0:  # L1 (MAE)
        offset_x = -70
        offset_y = 0
    elif i == 1:  # L2 (MSE)
        offset_x = -70
        offset_y = 0
    elif i == 2:  # Pairwise Fidelity
        offset_x = 20
        offset_y = 0
    elif i == 3:  # SRCC Loss
        offset_x = 20
        offset_y = 0
    else:  # Pairwise Ranking (i == 4)
        offset_x = 20
        offset_y = 0
    ax2.annotate(name.replace('\n', ' '), (srcc_val, plcc_val),
                xytext=(offset_x, offset_y), textcoords='offset points',
                fontsize=9, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor=colors_loss[i], alpha=0.7, edgecolor='black'))

ax2.set_xlabel('SRCC', fontsize=13, weight='bold')
ax2.set_ylabel('PLCC', fontsize=13, weight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim([0.928, 0.938])
ax2.set_ylim([0.920, 0.950])

# 对角线参考
ax2.plot([0.928, 0.938], [0.928, 0.938], 'k--', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig(f'{output_dir}/loss_function_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/loss_function_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ 已保存: {output_dir}/loss_function_comparison.pdf/.png")

print("\n" + "=" * 80)
print("✅ 所有图表生成完成！")
print("=" * 80)
print(f"\n输出目录: {output_dir}/")
print("\n生成的文件:")
print("  1. lr_sensitivity_final.pdf/.png      - 学习率敏感度")
print("  2. model_size_final.pdf/.png          - 模型大小对比")
print("  3. ablation_comparison_final.pdf/.png - 消融实验对比")
print("  4. loss_function_comparison.pdf/.png  - 损失函数对比")
print("=" * 80)

