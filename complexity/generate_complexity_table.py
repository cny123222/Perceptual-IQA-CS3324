#!/usr/bin/env python3
"""
生成计算复杂度对比表格和图表
用于论文展示
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

# 设置字体为Times
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

# 模型数据
models = {
    'HyperIQA\n(ResNet50)': {
        'params': 27.38,  # M
        'flops': 4.33,    # G
        'time': 3.12,     # ms
        'throughput': 329.73  # images/sec
    },
    'SMART-Tiny\n(Swin-T)': {
        'params': 29.52,
        'flops': 4.47,
        'time': 6.00,     # ms
        'throughput': 167.24  # images/sec
    },
    'SMART-Small\n(Swin-S)': {
        'params': 50.84,
        'flops': 8.65,
        'time': 10.62,    # ms
        'throughput': 92.73  # images/sec
    },
    'SMART-Base\n(Swin-B)': {
        'params': 89.11,
        'flops': 15.28,
        'time': 10.06,
        'throughput': 97.37
    }
}

def generate_latex_table():
    """生成LaTeX表格"""
    latex = r"""\begin{table}[!t]
\centering
\caption{Computational Complexity Comparison}
\label{tab:complexity}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Params (M)} & \textbf{FLOPs (G)} & \textbf{Time (ms)} & \textbf{FPS} \\
\midrule
HyperIQA (ResNet50) & 27.38 & 4.33 & 3.12 & 320.5 \\
SMART-Tiny (Swin-T) & 29.52 & 4.47 & 6.00 & 166.7 \\
SMART-Small (Swin-S) & 50.84 & 8.65 & 10.62 & 94.2 \\
SMART-Base (Swin-B) & 89.11 & 15.28 & 10.06 & 99.4 \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open('/root/Perceptual-IQA-CS3324/complexity/TABLE_COMPLEXITY.tex', 'w') as f:
        f.write(latex)
    
    print("✅ LaTeX表格已保存: complexity/TABLE_COMPLEXITY.tex")

def generate_comparison_plot():
    """生成对比图表"""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    
    model_names = list(models.keys())
    params = [models[m]['params'] for m in model_names]
    flops = [models[m]['flops'] for m in model_names]
    
    # 简化模型名称用于图表
    short_names = ['HyperIQA', 'SMART-T', 'SMART-S', 'SMART-B']
    
    # 子图1: 参数量对比
    ax1 = axes[0]
    bars1 = ax1.bar(short_names, params, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], 
                    alpha=0.8, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Parameters (M)', fontsize=11)
    ax1.set_ylim([0, max(params) * 1.15])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # 添加数值标注
    for bar, val in zip(bars1, params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(params)*0.02,
                f'{val:.1f}M', ha='center', va='bottom', fontsize=9)
    
    # 子图2: FLOPs对比
    ax2 = axes[1]
    bars2 = ax2.bar(short_names, flops, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], 
                    alpha=0.8, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('FLOPs (G)', fontsize=11)
    ax2.set_ylim([0, max(flops) * 1.15])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # 添加数值标注
    for bar, val in zip(bars2, flops):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(flops)*0.02,
                f'{val:.2f}G', ha='center', va='bottom', fontsize=9)
    
    # 旋转x轴标签
    for ax in axes:
        ax.tick_params(axis='both', labelsize=10)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    
    # 保存
    plt.savefig('/root/Perceptual-IQA-CS3324/paper_figures/complexity_comparison.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('/root/Perceptual-IQA-CS3324/paper_figures/complexity_comparison.png', 
                dpi=300, bbox_inches='tight')
    
    print("✅ 复杂度对比图已保存:")
    print("   - paper_figures/complexity_comparison.pdf")
    print("   - paper_figures/complexity_comparison.png")
    
    plt.close()

def generate_inference_time_plot():
    """生成推理时间对比图"""
    # 所有模型都有实际测量数据
    measured_models = list(models.keys())
    times = [models[m]['time'] for m in measured_models]
    throughputs = [models[m]['throughput'] for m in measured_models]
    
    short_names = ['HyperIQA', 'SMART-T', 'SMART-S', 'SMART-B']
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    
    # 子图1: 推理时间
    ax1 = axes[0]
    bars1 = ax1.bar(short_names, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], 
                    alpha=0.8, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Inference Time (ms)', fontsize=11)
    ax1.set_ylim([0, max(times) * 1.2])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # 添加数值标注
    for bar, val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.02,
                f'{val:.2f}ms', ha='center', va='bottom', fontsize=9)
    
    # 子图2: 吞吐量
    ax2 = axes[1]
    bars2 = ax2.bar(short_names, throughputs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], 
                    alpha=0.8, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Throughput (images/sec)', fontsize=11)
    ax2.set_ylim([0, max(throughputs) * 1.2])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # 添加数值标注
    for bar, val in zip(bars2, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(throughputs)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存
    plt.savefig('/root/Perceptual-IQA-CS3324/paper_figures/inference_time_comparison.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('/root/Perceptual-IQA-CS3324/paper_figures/inference_time_comparison.png', 
                dpi=300, bbox_inches='tight')
    
    print("✅ 推理时间对比图已保存:")
    print("   - paper_figures/inference_time_comparison.pdf")
    print("   - paper_figures/inference_time_comparison.png")
    
    plt.close()

def generate_markdown_summary():
    """生成Markdown汇总"""
    summary = """# 计算复杂度分析汇总

## 📊 所有模型对比

| 模型 | 参数量 (M) | FLOPs (G) | 推理时间 (ms) | 吞吐量 (images/sec) |
|------|-----------|-----------|--------------|---------------------|
| HyperIQA (ResNet50) | 27.38 | 4.33 | 3.12 | 329.73 |
| SMART-Tiny (Swin-T) | 29.52 | 4.47 | 6.00 | 167.24 |
| SMART-Small (Swin-S) | 50.84 | 8.65 | 10.62 | 92.73 |
| SMART-Base (Swin-B) | 89.11 | 15.28 | 10.06 | 97.37 |

## 🔍 关键观察

### 1. 参数量分析
- **HyperIQA (ResNet50)**: 27.38M - 最轻量
- **SMART-Tiny**: 29.52M - 与ResNet50相近 (+7.8%)
- **SMART-Small**: 50.84M - 中等规模 (+85.6% vs ResNet50)
- **SMART-Base**: 89.11M - 最大模型 (+225.5% vs ResNet50)

### 2. 计算复杂度分析
- **HyperIQA (ResNet50)**: 4.33G FLOPs - 最低计算量
- **SMART-Tiny**: 4.47G FLOPs - 与ResNet50相近 (+3.2%)
- **SMART-Small**: 8.65G FLOPs - 约2倍于ResNet50
- **SMART-Base**: 15.28G FLOPs - 约3.5倍于ResNet50

### 3. 推理速度分析（实测）
- **HyperIQA (ResNet50)**: 
  - 推理时间: 3.12ms
  - 吞吐量: 329.73 images/sec
  - **最快的模型**
  
- **SMART-Base**: 
  - 推理时间: 10.06ms (约3.2倍于ResNet50)
  - 吞吐量: 97.37 images/sec
  - 虽然较慢，但准确度显著提升（SRCC: 0.9378 vs ~0.89）

### 4. 准确度-效率权衡

从已有的实验结果来看：

| 模型 | SRCC | PLCC | 参数量 (M) | FLOPs (G) | 推理时间 (ms) |
|------|------|------|-----------|-----------|--------------|
| HyperIQA (ResNet50) | ~0.890 | ~0.910 | 27.38 | 4.33 | 3.12 |
| SMART-Base | 0.9378 | 0.9485 | 89.11 | 15.28 | 10.06 |

**准确度提升**: +5.4% SRCC, +4.2% PLCC  
**计算成本**: +3.5× FLOPs, +3.2× 推理时间

## 💡 结论

1. **SMART-Tiny** 与 HyperIQA (ResNet50) 复杂度相近，可作为直接替代方案
2. **SMART-Base** 虽然计算量较大，但仍在实时推理范围内（10ms < 100fps）
3. Swin Transformer backbone 相比ResNet50：
   - 以适度的计算开销（3-4倍）
   - 换取显著的准确度提升（5%+ SRCC）
   - 推理速度仍然实用（97 images/sec on RTX 5090）

## 📝 测试环境

- **GPU**: NVIDIA GeForce RTX 5090
- **输入尺寸**: 224×224×3
- **Batch Size**: 1 (用于推理时间测量)
- **精度**: FP32
- **Warmup**: 10次迭代
- **测量次数**: 100次迭代（取平均）

## 📚 详细报告

- [HyperIQA (ResNet50) 详细报告](complexity_results_resnet50.md)
- [SMART-Tiny 详细报告](complexity_results_swin_tiny.md)
- [SMART-Small 详细报告](complexity_results_swin_small.md)
- [SMART-Base 详细报告](complexity_results_swin_base.md)
"""
    
    with open('/root/Perceptual-IQA-CS3324/complexity/COMPLEXITY_SUMMARY.md', 'w') as f:
        f.write(summary)
    
    print("✅ Markdown汇总已保存: complexity/COMPLEXITY_SUMMARY.md")

def main():
    print("="*80)
    print("生成计算复杂度分析汇总")
    print("="*80)
    
    # 1. 生成LaTeX表格
    print("\n1. 生成LaTeX表格...")
    generate_latex_table()
    
    # 2. 生成对比图表
    print("\n2. 生成复杂度对比图...")
    generate_comparison_plot()
    
    # 3. 生成推理时间对比图
    print("\n3. 生成推理时间对比图...")
    generate_inference_time_plot()
    
    # 4. 生成Markdown汇总
    print("\n4. 生成Markdown汇总...")
    generate_markdown_summary()
    
    print("\n" + "="*80)
    print("✅ 所有汇总文件生成完成！")
    print("="*80)
    print("\n生成的文件:")
    print("  📊 LaTeX表格: complexity/TABLE_COMPLEXITY.tex")
    print("  📈 复杂度对比图: paper_figures/complexity_comparison.pdf/.png")
    print("  ⏱️  推理时间图: paper_figures/inference_time_comparison.pdf/.png")
    print("  📝 汇总报告: complexity/COMPLEXITY_SUMMARY.md")

if __name__ == '__main__':
    main()

