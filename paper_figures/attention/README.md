# Attention Visualization

This directory contains the attention mechanism visualization used in the paper.

## Files

- `attention_comparison_combined.pdf` - Final figure used in paper (Figure 6)
  - Shows channel attention weight distribution for different quality levels
  - Demonstrates adaptive attention mechanism

## Generation

To regenerate this figure:

```bash
# 1. Run attention visualization to generate data
cd /root/Perceptual-IQA-CS3324
python tools/visualization/visualize_attention.py

# 2. Create the combined comparison figure
python tools/visualization/create_attention_comparison.py
```

The generated figure will be saved to this directory.

## Source Data

- Input data: `tools/visualization/attention_visualization_results.json`
- Generation script: `tools/visualization/create_attention_comparison.py`

