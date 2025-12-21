# Round 1 结果对比分析

## 目标
对比所有实验的**第一轮（Round 1）**结果，而不是看3轮平均或最佳轮次。

## 重要发现

### Swin-Base + w/o Attention

**实验日志**: `swin_multiscale_ranking_alpha0.5_20251221_003537.log`
- **Round 1 SRCC**: 0.9316
- **Round 2 SRCC**: 0.9305  
- **Round 3**: 未完成

**注意**: 最佳模型SRCC 0.9336来自**不同的实验**！

### Swin-Small + Attention

**实验日志**: `swin_multiscale_ranking_alpha0.5_20251220_001328.log`
- **Round 1 SRCC**: 0.9311
- **Round 2 SRCC**: 0.9293
- **Round 3**: 未完成

### 需要确认的问题

1. **0.9336是哪个实验的结果？**
   - Checkpoint目录: `koniq-10k-swin-ranking-alpha0.5_20251220_091014/`
   - 有checkpoint_epoch_1（0.9327）和checkpoint_epoch_2（0.9336）
   - 但找不到对应的训练日志！

2. **如果只看Round 1**:
   - Base w/o Attention: 0.9316
   - Small + Attention: 0.9311
   - 差距很小（0.0005）

## 下一步

需要找到0.9336对应的完整训练日志，确认：
- 它是Base还是Small？
- 它是第几轮的结果？
- 有没有Attention？
