"""
Generate dual bar charts for ablation study
Left: SRCC comparison, Right: PLCC comparison
Using Times New Roman font with academic color scheme
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Use serif font (Liberation Serif is Times-compatible)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Data from ablation experiments
configurations = ['ResNet50\n(HyperIQA)', 'Swin-Base\n(Backbone)', '+ AFA', '+ Attention\n(Full)']
srcc_values = [0.9070, 0.9338, 0.9353, 0.9378]
plcc_values = [0.9180, 0.9437, 0.9469, 0.9485]

# Academic color scheme: professional colors inspired by scientific journals
# Muted blue, teal, amber, deep blue - colorful but professional
colors = ['#4472C4', '#70AD47', '#FFC000', '#5B9BD5']

# Create figure with two subplots (reduced size for better page fit)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

# Left subplot: SRCC
x_pos = np.arange(len(configurations))
bars1 = ax1.bar(x_pos, srcc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.0, width=0.7)

ax1.set_ylabel('SRCC', fontsize=11, fontweight='bold')
ax1.set_title('(a) SRCC Comparison', fontsize=11, fontweight='bold', pad=8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(configurations, fontsize=9)
ax1.set_ylim(0.88, 0.95)
ax1.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_axisbelow(True)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, srcc_values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='normal')

# Right subplot: PLCC
bars2 = ax2.bar(x_pos, plcc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.0, width=0.7)

ax2.set_ylabel('PLCC', fontsize=11, fontweight='bold')
ax2.set_title('(b) PLCC Comparison', fontsize=11, fontweight='bold', pad=8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(configurations, fontsize=9)
ax2.set_ylim(0.90, 0.96)
ax2.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_axisbelow(True)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, plcc_values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='normal')

# Tight layout (no overall title)
plt.tight_layout()

# Save figure
output_path = '/root/Perceptual-IQA-CS3324/paper_figures/ablation_dual_bars_times.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"✓ Saved dual bar chart to: {output_path}")

# Also save PNG version
png_path = output_path.replace('.pdf', '.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
print(f"✓ Saved PNG version to: {png_path}")

plt.close()

print("\n📊 Ablation Study Dual Bar Chart Summary:")
print(f"   Left panel (SRCC):")
print(f"   • ResNet50: {srcc_values[0]:.4f}")
print(f"   • Swin-Base: {srcc_values[1]:.4f}")
print(f"   • + AFA: {srcc_values[2]:.4f}")
print(f"   • + Attention: {srcc_values[3]:.4f}")
print(f"\n   Right panel (PLCC):")
print(f"   • ResNet50: {plcc_values[0]:.4f}")
print(f"   • Swin-Base: {plcc_values[1]:.4f}")
print(f"   • + AFA: {plcc_values[2]:.4f}")
print(f"   • + Attention: {plcc_values[3]:.4f}")

