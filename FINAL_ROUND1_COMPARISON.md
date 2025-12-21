# 所有实验Round 1结果对比

## 关键发现：0.9336来自Round 3！

根据record.md，最佳模型0.9336的完整轮次结果：
- **Round 1**: SRCC 0.9316, PLCC 0.9450
- **Round 2**: SRCC 0.9305, PLCC 0.9444  
- **Round 3**: SRCC 0.9336, PLCC 0.9464 ⭐ **最佳**

## Round 1 对比结果

| 配置 | Model Size | Attention | Alpha | Round 1 SRCC | 备注 |
|------|-----------|-----------|-------|--------------|------|
| **最佳Base** | Base | ❌ No | 0.5 | **0.9316** | 🥇 Round 1最佳 |
| Small+Att | Small | ✅ Yes | 0.5 | 0.9311 | 🥈 仅差0.0005 |
| Base最新 | Base | ❌ No | 0.5 | 0.9316 | 与最佳Base相同 |

## 关键结论

### 1. **如果只看Round 1**：

✅ **Base w/o Attention = 0.9316 是最好的**
- 但领先Small + Attention仅0.0005（0.05%）
- 差距极小，可以认为几乎相同

### 2. **多轮训练的重要性**：

**Base w/o Attention的3轮表现**：
- Round 1: 0.9316
- Round 2: 0.9305（下降）
- Round 3: 0.9336（提升）

**Small + Attention的3轮表现**：
- Round 1: 0.9311  
- Round 2: 0.9293（下降）
- Round 3: 未完成

📊 **观察**：
- Base的Round 3超过了Round 1
- Small的Round 2下降幅度更大
- **Base更稳定！**

### 3. **Attention的作用**：

在**相同轮次（Round 1）**对比：
- Small + Attention: 0.9311
- Base w/o Attention: 0.9316

**结论**：
- ❌ Attention对Small模型帮助不明显（甚至略低）
- ✅ Base模型即使不用Attention也更强

### 4. **模型大小的影响**：

- **Base (88M)** > **Small (50M)**
- 即使Base不用Attention，仍然优于Small + Attention
- **模型容量比Attention机制更重要**

## 最终推荐

### 如果只做1轮训练：
✅ **推荐：Swin-Base w/o Attention**
- Round 1 SRCC: 0.9316
- 最稳定，最简单

### 如果做3轮取最佳：
✅ **推荐：Swin-Base w/o Attention**  
- Round 3 SRCC: 0.9336
- 3轮中有提升空间

### Attention的建议：
❌ **不推荐使用Attention**
- 对Small模型无明显帮助
- Base模型不需要Attention已经很强
- 增加复杂度但收益不明显

## 回答你的问题

**"只对比一轮的话，base+w/o attention还是最好的吗？"**

✅ **是的！** Base w/o Attention的Round 1 (0.9316) 仍然是最好的。

但需要注意：
1. 领先Small + Attention仅0.0005
2. Round 3的0.9336比Round 1更好
3. 建议做3轮取最佳，而不是只做1轮

