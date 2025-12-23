#!/usr/bin/env python3
"""
生成论文所需的所有图表
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 创建输出文件夹
os.makedirs('IEEE-conference-template-062824/figures', exist_ok=True)

# 设置全局样式
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 14

# ============================================================================
# Figure 1: Ablation Study Bar Chart
# ============================================================================

def generate_ablation_chart():
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    components = ['Swin\nTransformer', 'Multi-Scale\nFusion', 'Attention\nFusion']
    contributions_pct = [87, 5, 8]
    improvements_srcc = [0.0268, 0.0015, 0.0025]
    
    # Bar chart for percentage contributions
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax1.bar(components, contributions_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Contribution (%)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, contributions_pct):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{pct}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Secondary y-axis for SRCC improvements
    ax2 = ax1.twinx()
    line = ax2.plot(components, [i*100 for i in improvements_srcc], 
                    'ro-', linewidth=2.5, markersize=10, label='SRCC Improvement')
    ax2.set_ylabel('SRCC Improvement (%)', fontsize=13, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 3.0)
    
    # Add SRCC improvement labels
    for i, (comp, imp) in enumerate(zip(components, improvements_srcc)):
        ax2.text(i, imp*100 + 0.15, f'+{imp:.4f}', ha='center', va='bottom', 
                fontsize=10, color='red', fontweight='bold')
    
    plt.title('Component Contribution Analysis', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('IEEE-conference-template-062824/figures/ablation_chart.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('IEEE-conference-template-062824/figures/ablation_chart.png', dpi=300, bbox_inches='tight')
    print("✅ Generated: ablation_chart.pdf & .png")
    plt.close()

# ============================================================================
# Figure 2: Learning Rate Sensitivity Curve
# ============================================================================

def generate_lr_curve():
    lrs = np.array([1e-7, 5e-7, 1e-6, 3e-6, 5e-6])
    srccs = np.array([0.9375, 0.9378, 0.9374, 0.9364, 0.9354])
    
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, srccs, 'o-', linewidth=2.5, markersize=12, color='#2E86AB', label='SRCC')
    
    # Mark optimal point
    optimal_idx = np.argmax(srccs)
    plt.plot(lrs[optimal_idx], srccs[optimal_idx], 'r*', markersize=20, 
             label=f'Optimal: 5e-7 (SRCC={srccs[optimal_idx]:.4f})')
    
    # Add vertical line at optimal
    plt.axvline(lrs[optimal_idx], color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add value labels
    for lr, srcc in zip(lrs, srccs):
        plt.text(lr, srcc + 0.0003, f'{srcc:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.xscale('log')
    plt.xlabel('Learning Rate', fontsize=13, fontweight='bold')
    plt.ylabel('SRCC', fontsize=13, fontweight='bold')
    plt.title('Learning Rate Sensitivity Analysis', fontsize=15, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(loc='lower left', fontsize=11)
    plt.ylim(0.935, 0.938)
    
    plt.tight_layout()
    plt.savefig('IEEE-conference-template-062824/figures/lr_sensitivity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('IEEE-conference-template-062824/figures/lr_sensitivity.png', dpi=300, bbox_inches='tight')
    print("✅ Generated: lr_sensitivity.pdf & .png")
    plt.close()

# ============================================================================
# Figure 3: Model Size vs Performance Scatter Plot
# ============================================================================

def generate_model_size_scatter():
    models = ['Tiny', 'Small', 'Base']
    params = np.array([28, 50, 88])  # Million
    srccs = np.array([0.9249, 0.9338, 0.9378])
    colors = ['#F18F01', '#A23B72', '#2E86AB']
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    for i, (model, param, srcc, color) in enumerate(zip(models, params, srccs, colors)):
        plt.scatter(param, srcc, s=400, color=color, alpha=0.7, 
                   edgecolors='black', linewidth=2, label=f'{model} ({param}M, {srcc:.4f})')
    
    # Connect with line
    plt.plot(params, srccs, '--', linewidth=2, color='gray', alpha=0.5)
    
    # Add annotations
    for model, param, srcc in zip(models, params, srccs):
        plt.annotate(f'{model}\n{param}M params\nSRCC: {srcc:.4f}',
                    xy=(param, srcc), xytext=(10, 10), textcoords='offset points',
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    plt.xlabel('Parameters (Million)', fontsize=13, fontweight='bold')
    plt.ylabel('SRCC', fontsize=13, fontweight='bold')
    plt.title('Model Size vs Performance Trade-off', fontsize=15, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(loc='lower right', fontsize=11)
    plt.ylim(0.920, 0.940)
    
    plt.tight_layout()
    plt.savefig('IEEE-conference-template-062824/figures/model_size_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('IEEE-conference-template-062824/figures/model_size_scatter.png', dpi=300, bbox_inches='tight')
    print("✅ Generated: model_size_scatter.pdf & .png")
    plt.close()

# ============================================================================
# Figure 4: Cross-Dataset Generalization Comparison
# ============================================================================

def generate_cross_dataset_comparison():
    datasets = ['KonIQ-10k\n(In-domain)', 'SPAQ\n(Smartphone)', 'KADID-10K\n(Synthetic)', 'AGIQA-3K\n(AI-gen)']
    srcc_values = [0.9378, 0.8698, 0.5412, 0.6484]
    colors = ['#2E86AB', '#06A77D', '#F18F01', '#D62246']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(datasets, srcc_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, srcc_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add horizontal line for average cross-domain
    avg_cross = np.mean(srcc_values[1:])
    ax.axhline(avg_cross, color='red', linestyle='--', linewidth=2, 
               label=f'Avg Cross-domain: {avg_cross:.4f}')
    
    ax.set_ylabel('SRCC', fontsize=13, fontweight='bold')
    ax.set_title('Cross-Dataset Generalization Performance', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('IEEE-conference-template-062824/figures/cross_dataset_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('IEEE-conference-template-062824/figures/cross_dataset_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Generated: cross_dataset_comparison.pdf & .png")
    plt.close()

# ============================================================================
# Figure 5: Progressive Ablation (Step-by-step improvement)
# ============================================================================

def generate_progressive_ablation():
    configs = ['ResNet50\n(Baseline)', 'Swin-Base\n(Single-scale)', 'Swin-Base\n+ Multi-scale', 'Swin-Base\n+ MS + Attn\n(Full)']
    srccs = [0.907, 0.9338, 0.9353, 0.9378]
    improvements = [0, 0.0268, 0.0015, 0.0025]
    colors = ['#CCCCCC', '#F18F01', '#A23B72', '#2E86AB']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(configs, srccs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add SRCC labels
    for bar, srcc in zip(bars, srccs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
               f'{srcc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement arrows
    for i in range(1, len(configs)):
        x1 = i - 0.5
        x2 = i - 0.5
        y1 = srccs[i-1] + 0.001
        y2 = srccs[i] - 0.001
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        # Add improvement value
        mid_y = (y1 + y2) / 2
        ax.text(x1 + 0.1, mid_y, f'+{improvements[i]:.4f}', 
               fontsize=10, color='green', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    
    ax.set_ylabel('SRCC', fontsize=13, fontweight='bold')
    ax.set_title('Progressive Ablation Study: Step-by-Step Improvement', fontsize=15, fontweight='bold')
    ax.set_ylim(0.90, 0.94)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total improvement annotation
    ax.text(0.5, 0.935, f'Total Improvement: +{sum(improvements[1:]):.4f} (+3.08%)', 
           fontsize=12, fontweight='bold', color='red',
           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('IEEE-conference-template-062824/figures/progressive_ablation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('IEEE-conference-template-062824/figures/progressive_ablation.png', dpi=300, bbox_inches='tight')
    print("✅ Generated: progressive_ablation.pdf & .png")
    plt.close()

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("📊 Generating Paper Figures")
    print("="*60 + "\n")
    
    generate_ablation_chart()
    generate_lr_curve()
    generate_model_size_scatter()
    generate_cross_dataset_comparison()
    generate_progressive_ablation()
    
    print("\n" + "="*60)
    print("✅ All figures generated successfully!")
    print("📁 Location: IEEE-conference-template-062824/figures/")
    print("="*60 + "\n")
    
    print("Generated files:")
    print("  - ablation_chart.pdf & .png")
    print("  - lr_sensitivity.pdf & .png")
    print("  - model_size_scatter.pdf & .png")
    print("  - cross_dataset_comparison.pdf & .png")
    print("  - progressive_ablation.pdf & .png")
    print("\n📝 Copy training_curves_best_model.png to figures folder:")
    print("     cp training_curves_best_model.png IEEE-conference-template-062824/figures/")

