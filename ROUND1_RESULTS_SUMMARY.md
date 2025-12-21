# Round 1 实验结果总结（单轮对比）

## 🎯 目标
只看每个实验的**第一轮（Round 1）**结果，找出最有潜力的配置。

---

## 📊 Round 1 实验结果排名

### 🏆 Top 5 配置（按SRCC排序）

| 排名 | 配置 | Model | Attention | Alpha | Round 1 SRCC | Round 1 PLCC | 备注 |
|------|------|-------|-----------|-------|--------------|--------------|------|
| 🥇 | **Base w/o Att** | Base (88M) | ❌ | 0.5 | **0.9316** | **0.9450** | 当前最佳 |
| 🥈 | Small + Att | Small (50M) | ✅ | 0.5 | 0.9311 | 0.9424 | 仅差0.0005 |
| 🥉 | Small w/o Att | Small (50M) | ❌ | 0.5 | 0.9303 | 0.9444 | 稳定 |
| 4 | Tiny + ColorJitter | Tiny (28M) | ❌ | 0 | 0.9236 | 0.9371 | 有ColorJitter |
| 5 | Tiny + alpha=0 | Tiny (28M) | ❌ | 0 | 0.9229 | 0.9361 | 无ColorJitter |

---

## 🔬 详细分析

### 1️⃣ **Base w/o Attention (SRCC 0.9316)** 🏆

**配置**:
- Model: Swin-Base (88M 参数)
- Attention: ❌ No (简单concatenation)
- Ranking Loss Alpha: 0.5
- Learning Rate: 5e-6
- Weight Decay: 2e-4
- Dropout: 0.4
- Drop Path: 0.3

**优势**:
- ✅ Round 1 最佳结果 (0.9316)
- ✅ 稳定性好（3轮范围: 0.9305-0.9336）
- ✅ 模型最大，容量最强

**是否值得继续**:
✅ **强烈推荐** - 已经是最佳配置

---

### 2️⃣ **Small + Attention (SRCC 0.9311)** 🥈

**配置**:
- Model: Swin-Small (50M 参数)
- Attention: ✅ Yes (attention-based fusion)
- Ranking Loss Alpha: 0.5
- Learning Rate: 5e-6
- Weight Decay: 2e-4
- Dropout: 0.4
- Drop Path: 0.3

**优势**:
- ✅ Round 1 第二名 (0.9311)
- ✅ 仅比Base低0.0005 (0.05%)
- ✅ 参数量更少 (50M vs 88M)

**劣势**:
- ⚠️ 稳定性较差（3轮范围: 0.9254-0.9311，波动0.57%）
- ⚠️ Round 2-3 性能下降

**是否值得继续**:
⚠️ **谨慎** - Attention效果不稳定，不如直接用Base

---

### 3️⃣ **Small w/o Attention (SRCC 0.9303)** 🥉

**配置**:
- Model: Swin-Small (50M 参数)
- Attention: ❌ No (简单concatenation)
- Ranking Loss Alpha: 0.5

**优势**:
- ✅ 比Small + Attention更稳定
- ✅ 简单有效

**劣势**:
- ⚠️ 比Base低0.0013 (0.14%)

**是否值得继续**:
⚠️ **不推荐** - 不如直接用Base

---

## 🧪 未测试但值得尝试的配置

### 1. **Base + Attention** ⭐⭐⭐⭐

**理由**:
- Base是最强的backbone
- Small + Attention在Round 1有0.9311，接近Base
- **可能性**: Base + Attention > Base w/o Attention

**预期**:
- 乐观: SRCC 0.9320-0.9330
- 保守: SRCC 0.9310-0.9320
- 悲观: SRCC 0.9300-0.9310 (Attention负作用)

**推荐指数**: ⭐⭐⭐⭐ (4/5)

---

### 2. **Base + alpha=0.3** ⭐⭐⭐

**理由**:
- 当前Base使用alpha=0.5
- 可能alpha=0.3更优

**预期**:
- SRCC 0.9310-0.9320

**推荐指数**: ⭐⭐⭐ (3/5)

---

### 3. **Base + alpha=0** ⭐⭐

**理由**:
- 纯L1 loss，无ranking loss
- Small模型上alpha=0和alpha=0.5差距不大

**预期**:
- SRCC 0.9300-0.9315

**推荐指数**: ⭐⭐ (2/5) - 优先级较低

---

### 4. **Base + 更强正则化** ⭐⭐

**理由**:
- 当前dropout=0.4, drop_path=0.3
- 可能更强正则化能提升稳定性

**配置建议**:
- dropout=0.5, drop_path=0.4
- 或 weight_decay=3e-4

**预期**:
- SRCC 0.9305-0.9320

**推荐指数**: ⭐⭐ (2/5) - 优先级较低

---

### 5. **Base + 更小学习率** ⭐

**理由**:
- 当前lr=5e-6
- 可能lr=3e-6或4e-6更优

**预期**:
- SRCC 0.9310-0.9325

**推荐指数**: ⭐ (1/5) - 优先级很低

---

## 📈 模型大小的影响

| Model | 参数量 | Round 1 SRCC | 提升幅度 |
|-------|--------|--------------|----------|
| Tiny | 28M | 0.9236 | 基准 |
| Small | 50M | 0.9303 | +0.67% |
| **Base** | 88M | **0.9316** | **+0.80%** |

**结论**: 模型越大，性能越好 ✅

---

## 🎲 Attention的影响

| 配置 | Attention | Round 1 SRCC | 差异 |
|------|-----------|--------------|------|
| Small w/o Att | ❌ | 0.9303 | 基准 |
| Small + Att | ✅ | 0.9311 | +0.08% |
| Base w/o Att | ❌ | 0.9316 | 基准 |
| Base + Att | ✅ | **未测试** | ??? |

**结论**: 
- Small上: Attention有微小提升 (+0.08%)
- Base上: **需要测试！**

---

## 🎯 Ranking Loss Alpha的影响

| 配置 | Alpha | Round 1 SRCC | 说明 |
|------|-------|--------------|------|
| Tiny | 0 | 0.9229 | 纯L1 |
| Tiny | 0.3 | 未完整测试 | - |
| Tiny | 0.5 | 未完整测试 | - |
| Small | 0.5 | 0.9303 | - |
| Base | 0.5 | 0.9316 | 当前最佳 |
| Base | 0.3 | **未测试** | 值得尝试 |
| Base | 0 | **未测试** | 值得尝试 |

**结论**: 
- Base模型可能对alpha更敏感
- **需要测试alpha=0.3和alpha=0**

---

## 💡 最终推荐

### 立即测试的实验（按优先级）:

#### 🔥 **优先级1: Base + Attention**
```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --attention_fusion \
  --epochs 30 \
  --train_test_num 1 \
  --batch_size 32 \
  --ranking_loss_alpha 0.5 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```
**原因**: 最有可能超过当前最佳 (0.9316)

---

#### 🔥 **优先级2: Base + alpha=0.3**
```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --epochs 30 \
  --train_test_num 1 \
  --batch_size 32 \
  --ranking_loss_alpha 0.3 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```
**原因**: 微调alpha可能有提升

---

#### 🌟 **优先级3: Base + alpha=0**
```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --epochs 30 \
  --train_test_num 1 \
  --batch_size 32 \
  --ranking_loss_alpha 0 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --test_random_crop \
  --no_spaq
```
**原因**: 验证ranking loss的作用

---

### 不推荐的方向:

❌ **Small模型系列** - 性能不如Base，没有继续探索的价值  
❌ **Tiny模型系列** - 性能太低  
❌ **更强正则化** - 当前正则化已经很强  
❌ **更小学习率** - 当前lr已经很小  

---

## 📊 预期结果

| 实验 | 预期SRCC | 可能性 |
|------|----------|--------|
| Base + Attention | 0.9320-0.9330 | 40% |
| Base + Attention | 0.9310-0.9320 | 35% |
| Base + Attention | 0.9300-0.9310 | 25% |
| Base + alpha=0.3 | 0.9310-0.9320 | 60% |
| Base + alpha=0 | 0.9300-0.9315 | 70% |

---

## 🎯 总结

### Round 1 最佳配置:
✅ **Base w/o Attention (SRCC 0.9316)**

### 最值得尝试:
1. ⭐⭐⭐⭐ Base + Attention
2. ⭐⭐⭐ Base + alpha=0.3
3. ⭐⭐ Base + alpha=0

### 关键发现:
- ✅ 模型大小 > Attention机制
- ✅ Base模型是最优backbone
- ⚠️ Attention在Small上效果有限（+0.08%）
- ❓ Attention在Base上的效果未知（需要测试！）

---

**建议**: 先运行Base + Attention和Base + alpha=0.3这两个实验，如果有一个超过0.9320，就采用那个配置；如果都不行，就继续使用Base w/o Attention (0.9316)作为最终模型。

