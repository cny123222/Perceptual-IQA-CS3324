# **Hyper-IQA 模型增强指南**

## **1. 项目目标**

通过实施一项或多项高级架构或训练策略的改进，对基线 Hyper-IQA 模型进行优化。核心目标是在关键的图像质量评价（IQA）指标（主要是 SRCC 和 PLCC）上取得更优异的性能，并对所实施的改进提供全面、深入的分析。

## **2. 改进方案模块**

以下是三个结构化的改进方案。建议首先实施 **方案一** 作为一个高价值的基线增强，然后可以根据情况选择性地集成 **方案二** 或探索 **方案三**。

---

### **方案一：【高性价比】升级特征提取骨干网络**

**核心思想**: 原始模型使用的 ResNet-50 (2015年) 已相对陈旧。通过更换一个更强大、更现代的骨干网络，可以为超网络（Hyper Network）和目标网络（Target Network）提供信息更丰富、表达能力更强的特征，从而从根本上提升整个模型的性能上限。

**行动步骤**:

1.  **选择新骨干网络**: 从当前先进的高性能架构中进行选择。**Swin Transformer** 是一个绝佳的选择，因为它在捕捉图像局部和全局依赖关系方面的能力已得到广泛验证。
    *   **首选**: `swin_tiny_patch4_window7_224`
    *   **备选**: `convnext_tiny`, `efficientnetv2_s`

2.  **通过 `timm` 库实现**: `timm` 库使得更换骨干网络变得异常简单。实例化模型时，需设置 `pretrained=True` 以利用 ImageNet 预训练权重，并设置 `features_only=True` 以方便地从中间层提取特征图。

    ```python
    import timm
    import torch
    import torch.nn as nn

    class EnhancedHyperIQA(nn.Module):
        def __init__(self, backbone_name='swin_tiny_patch4_window7_224'):
            super().__init__()
            # 步骤1: 实例化新的骨干网络
            # 'features_only=True' 参数会让模型返回一个包含不同阶段特征图的列表
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
            )
            
            # 步骤2: 获取特征维度，以便适配下游网络（这是确保兼容性的关键一步）
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            
            # 示例：打印维度以理解输出结构
            # 对于Swin-T，可能会输出4个特征图，通道数分别为 [96, 192, 384, 768]
            self.feature_dims = [f.shape[1] for f in dummy_features]
            print(f"骨干网络特征维度: {self.feature_dims}")

            # 步骤3: 适配超网络和目标网络
            # 这两个网络的输入通道数必须被更新，以匹配 self.feature_dims
            # ... (你的超网络和目标网络的定义将在这里)
            # 例如，用于超网络输入的全局语义特征维度：
            # 通常使用最后一个特征维度
            self.hyper_net_in_dim = self.feature_dims[-1]
            # ...
            # 例如，用于目标网络输入的多尺度特征维度：
            # 所有特征维度的总和
            self.target_net_in_dim = sum(self.feature_dims)
            # ...
    ```

3.  **适配下游网络**:
    *   **全局语义特征**: 将新骨干网络最后一个阶段输出的特征图（例如，Swin-T 的 768 通道）进行全局平均池化，然后送入超网络。相应地，调整超网络输入层的维度。
    *   **多尺度特征**: 从 `features_only` 输出的所有中间阶段提取特征。这些特征将分别被处理，然后拼接（concatenate）起来，作为目标网络的输入。调整目标网络的输入层以匹配拼接后的总特征维度。

**报告中预期的分析要点**:
- **量化对比**: 创建一个表格，清晰对比 `ResNet-50` 与你的新骨干网络（如 `Swin-T`）在以下方面的差异：
    - 模型参数量 (Params, M)
    - 计算量 (FLOPs, G)
    - 推理延迟 (ms/image)
    - 在所有测试集上的 SRCC 和 PLCC 分数。
- **定性分析**: 深入探讨为什么新骨干网络性能更好。可以提出假设，例如 Transformer 的自注意力机制能更有效地权衡不同图像区域的失真对整体质量的重要性。

---

### **方案二：【高效率】优化学习目标——引入排序损失函数**

**核心思想**: 核心评价指标 SRCC 本质上衡量的是**排序的一致性**。标准的 L1/MAE 损失函数优化的是预测值的绝对误差，但并未直接优化排序能力。通过引入一个成对排序损失（Pairwise Ranking Loss），我们可以训练模型对图片间的相对质量更敏感，从而直接提升 SRCC 指标。

**行动步骤**:

1.  **定义排序损失函数**: 实现一个成对的 Hinge Loss。
    ```python
    import torch
    import torch.nn.functional as F

    def pairwise_ranking_loss(preds, labels, margin=0.1):
        """
        为一个批次（batch）计算成对排序损失。
        Args:
            preds (torch.Tensor): 模型对批次的预测分数。
            labels (torch.Tensor): 批次的真实标签分数。
            margin (float): Hinge Loss 的边界值。
        Returns:
            torch.Tensor: 排序损失值。
        """
        # 在批次内创建预测值和标签值的成对差异矩阵
        # 这会创建一个 N*N 的矩阵，N是批次大小
        pred_diffs = preds.unsqueeze(1) - preds.unsqueeze(0)
        label_diffs = labels.unsqueeze(1) - labels.unsqueeze(0)
        
        # 我们只关心标签值有差异的图像对
        # sign() 的结果会是 -1, 0, 或 1
        label_signs = torch.sign(label_diffs)
        
        # 当预测的顺序与标签的顺序相反时，产生损失
        # 我们使用 Hinge Loss 的形式
        loss = F.relu(-pred_diffs * label_signs + margin)
        
        # 只计算有效的图像对（即标签不同的对）的损失
        mask = (label_signs != 0).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1e-6)
        
        return loss
    ```

2.  **集成到训练循环中**: 修改主训练循环以计算一个复合损失。
    ```python
    # 在你的训练循环内部...
    # predictions = model(images)
    # labels = ...

    # 原始的 L1 Loss
    l1_loss = F.l1_loss(predictions, labels)
    
    # 新的排序损失
    rank_loss = pairwise_ranking_loss(predictions, labels, margin=0.1)
    
    # 复合损失
    alpha = 0.5 # 这是一个需要调试的超参数
    total_loss = l1_loss + alpha * rank_loss
    
    # 反向传播
    # total_loss.backward()
    # optimizer.step()
    ```

3.  **超参数调试**: 权重 `alpha` 至关重要。在验证集上实验不同的值，如 `[0.1, 0.5, 1.0, 2.0]`，以找到平衡绝对误差和排序能力的最优值。

**报告中预期的分析要点**:
- **指标针对性提升**: 绘制仅使用 L1 Loss 和使用复合 Loss 两种情况下，模型在验证集上的 SRCC 和 PLCC 随训练轮数变化的曲线图。预期会观察到 SRCC 曲线有更显著的提升。
- **散点图可视化**: 为两种模型分别生成“预测分数 vs. 真实分数”的散点图。通过对比分析，论证复合损失模型的预测点分布虽然不一定更靠近 y=x 对角线 (PLCC)，但展现出更紧密、更符合单调递增的趋势 (SRCC)。

---

### **方案三：【前沿探索】强化超网络——为其提供多尺度语义信息**

**核心思想**: 原始模型中，用于生成“评价规则”的超网络仅接收来自骨干网络**最后一层**的全局语义特征，这可能构成**信息瓶颈**。通过为超网络提供更丰富的多尺度语义信息，它可能生成更精细、更准确的“定制评价规则”。

**行动步骤**:

1.  **构建“超级语义向量”**: 不再只使用最后一层特征，而是聚合来自多个阶段的特征。

    ```python
    # 在你的模型 forward 函数内部...
    
    # features 是一个特征图列表，例如来自 Swin Transformer
    # features = self.backbone(image_input) -> [f_stage1, f_stage2, f_stage3, f_stage4]
    
    # 原始的用于超网络的语义特征
    # semantic_feature_original = F.adaptive_avg_pool2d(features[-1], 1).squeeze()

    # --- 新的实现 ---
    # 1. 对来自多个阶段的特征图进行全局平均池化
    pooled_features = [F.adaptive_avg_pool2d(f, 1).squeeze(dim=-1).squeeze(dim=-1) for f in features]
    
    # 2. 将它们拼接起来，形成新的“超级语义向量”
    # 需要处理批次大小为1时，squeeze可能移除batch维度的情况
    if pooled_features[0].dim() == 1:
        pooled_features = [pf.unsqueeze(0) for pf in pooled_features]
        
    hyper_feature_vector = torch.cat(pooled_features, dim=1)
    # --- 新实现结束 ---
    
    # 3. 将这个新的向量送入超网络
    # target_net_params = self.hyper_net(hyper_feature_vector)
    ```

2.  **适配超网络输入层**: 必须修改超网络的第一个层（如卷积层或全连接层），使其输入维度能够接受新的、维度更大的 `hyper_feature_vector`。例如，如果你拼接了4个阶段的特征，其维度分别为 `[96, 192, 384, 768]`，那么新的输入维度将是 `96 + 192 + 384 + 768 = 1440`。

**报告中预期的分析要点**:
- **架构图**: 绘制一张“改进前”和“改进后”的数据流图，清晰地展示数据如何流入超网络。这能直观地凸显你的创新点。
- **性能分析**: 如果实现成功，理想情况下模型的 SRCC 和 PLCC 都应有所提升，因为动态生成的“规则”（即目标网络的权重）本身的质量得到了根本性的改善。
- **权重空间可视化 (高级)**: 复现原始论文中的 PCA 可视化（Figure 5）。通过对比论证，你的增强模型为不同内容的图像在生成的权重空间中创建了更清晰、分离更远的簇，从而为你的“规则生成器变得更智能”这一论点提供证据。