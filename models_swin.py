import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import timm


class MultiScaleAttention(nn.Module):
    """
    Lightweight channel attention for multi-scale feature fusion.
    
    Uses the highest-level feature (feat3) to generate attention weights
    for all 4 scale features, enabling dynamic and adaptive fusion.
    
    Args:
        in_channels_list: List of channel dimensions for each scale
                         Tiny/Small: [96, 192, 384, 768]
                         Base: [128, 256, 512, 1024]
    """
    def __init__(self, in_channels_list):
        super(MultiScaleAttention, self).__init__()
        self.in_channels = in_channels_list
        self.num_scales = len(in_channels_list)
        
        # Attention generation network (lightweight)
        # Uses the highest-level feature to generate weights for all scales
        self.attention_net = nn.Sequential(
            nn.Linear(in_channels_list[-1], 256),  # Last channel (768 or 1024) -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Strong regularization to prevent overfitting
            nn.Linear(256, self.num_scales),  # -> 4 weights
            nn.Softmax(dim=1)  # Normalize to sum to 1
        )
        
        # Initialize weights
        for m in self.attention_net.modules():
            if isinstance(m, nn.Linear):
                # Use smaller initialization for more balanced initial weights
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    # Initialize bias to encourage uniform attention at start
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, feat_list):
        """
        Args:
            feat_list: List of 4 feature maps [feat0, feat1, feat2, feat3]
                       Tiny/Small:
                         feat0: (B, 96, H0, W0)
                         feat1: (B, 192, H1, W1)
                         feat2: (B, 384, H2, W2)
                         feat3: (B, 768, 7, 7)
                       Base:
                         feat0: (B, 128, H0, W0)
                         feat1: (B, 256, H1, W1)
                         feat2: (B, 512, H2, W2)
                         feat3: (B, 1024, 7, 7)
        
        Returns:
            fused_feat: (B, sum(channels), 7, 7) - weighted concatenated features
            attention_weights: (B, 4) - attention weights for visualization
        """
        B = feat_list[0].size(0)
        
        # 1. Unify spatial dimensions to 7x7
        feats_pooled = []
        for feat in feat_list:
            feat_pooled = F.adaptive_avg_pool2d(feat, (7, 7))
            feats_pooled.append(feat_pooled)
        
        # 2. Extract global representation from highest-level feature
        feat3_global = F.adaptive_avg_pool2d(feat_list[-1], (1, 1)).squeeze(-1).squeeze(-1)  # [B, last_channel]
        
        # 3. Generate attention weights
        attention_weights = self.attention_net(feat3_global)  # [B, 4]
        
        # 4. Apply attention weights to each scale
        weighted_feats = []
        for i, feat in enumerate(feats_pooled):
            # Broadcast attention weight: [B] -> [B, 1, 1, 1]
            weight = attention_weights[:, i].view(B, 1, 1, 1)
            weighted_feat = feat * weight
            weighted_feats.append(weighted_feat)
        
        # 5. Concatenate along channel dimension
        fused_feat = torch.cat(weighted_feats, dim=1)  # [B, sum(channels), 7, 7]
        
        return fused_feat, attention_weights


class HyperNet(nn.Module):
    """
    Hyper network for learning perceptual rules with Swin Transformer backbone.

    Args:
        lda_out_channels: local distortion aware module output size.
        hyper_in_channels: input feature channels for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        feature_size: input feature map width/height for hyper network.

    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.

    """
    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size, target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size, feature_size, use_multiscale=False, use_attention=False, drop_path_rate=0.2, dropout_rate=0.3, model_size='tiny'):
        super(HyperNet, self).__init__()

        self.hyperInChn = hyper_in_channels
        self.use_multiscale = use_multiscale  # 多尺度融合标志
        self.use_attention = use_attention  # 注意力机制标志
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size
        self.dropout_rate = dropout_rate  # Dropout rate for regularization
        self.model_size = model_size  # Swin model size

        self.swin = swin_backbone(lda_out_channels, target_in_size, pretrained=True, drop_path_rate=drop_path_rate, model_size=model_size)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Multi-scale attention module (if enabled)
        if use_multiscale and use_attention:
            if model_size == 'base':
                channels_list = [128, 256, 512, 1024]
            else:  # tiny or small
                channels_list = [96, 192, 384, 768]
            self.multiscale_attention = MultiScaleAttention(channels_list)
            print(f'Using attention-based multi-scale feature fusion for {model_size.upper()}')

        # Conv layers for swin output features
        # Determine input channels based on model size and whether multi-scale is used
        if model_size == 'base':
            # Swin-Base: [128, 256, 512, 1024]
            single_scale_channels = 1024
            multi_scale_channels = 128 + 256 + 512 + 1024  # 1920
        else:
            # Swin-Tiny/Small: [96, 192, 384, 768]
            single_scale_channels = 768
            multi_scale_channels = 96 + 192 + 384 + 768  # 1440
        
        input_channels = multi_scale_channels if use_multiscale else single_scale_channels
        print(f'HyperNet input channels: {input_channels} (multi_scale={use_multiscale}, model={model_size})')
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        # Add Dropout for regularization to combat overfitting
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,  padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

        # initialize (skip modules without weights like Dropout)
        for i, m_name in enumerate(self._modules):
            if i > 2 and hasattr(self._modules[m_name], 'weight'):
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, img):
        feature_size = self.feature_size

        swin_out = self.swin(img)

        # input vector for target net
        target_in_vec = swin_out['target_in_vec'].view(-1, self.target_in_size, 1, 1)

        # input features for hyper net
        if self.use_multiscale and 'hyper_in_feat_multi' in swin_out:
            # 多尺度融合模式
            feat0, feat1, feat2, feat3 = swin_out['hyper_in_feat_multi']
            
            if self.use_attention:
                # 使用注意力机制进行动态加权融合
                hyper_in_feat_raw, attention_weights = self.multiscale_attention([feat0, feat1, feat2, feat3])
                # 保存注意力权重用于可视化（可选）
                self.last_attention_weights = attention_weights.detach()
            else:
                # 简单拼接融合（原始方法）
                # 将所有阶段特征统一到 7x7 空间尺寸
                feat0_pooled = F.adaptive_avg_pool2d(feat0, (feature_size, feature_size))  # [B, C0, 7, 7]
                feat1_pooled = F.adaptive_avg_pool2d(feat1, (feature_size, feature_size))  # [B, C1, 7, 7]
                feat2_pooled = F.adaptive_avg_pool2d(feat2, (feature_size, feature_size))  # [B, C2, 7, 7]
                feat3_pooled = feat3  # [B, C3, 7, 7] 已经是目标尺寸
                
                # 在通道维度拼接：Tiny/Small: 96+192+384+768=1440, Base: 128+256+512+1024=1920
                hyper_in_feat_raw = torch.cat([feat0_pooled, feat1_pooled, feat2_pooled, feat3_pooled], dim=1)
            
            hyper_in_feat = self.conv1(hyper_in_feat_raw).view(-1, self.hyperInChn, feature_size, feature_size)
        else:
            # 单尺度模式（向后兼容）
            hyper_in_feat = self.conv1(swin_out['hyper_in_feat']).view(-1, self.hyperInChn, feature_size, feature_size)

        # Apply dropout for regularization (only during training)
        hyper_in_feat = self.dropout(hyper_in_feat)

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        return out


class TargetNet(nn.Module):
    """
    Target network for quality prediction.
    """
    def __init__(self, paras, dropout_rate=0.3):
        super(TargetNet, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            TargetFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        self.l4 = nn.Sequential(
            TargetFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            TargetFC(paras['target_fc5w'], paras['target_fc5b']),
        )

    def forward(self, x):
        q = self.l1(x)
        q = self.dropout(q)  # Dropout after l1
        q = self.l2(q)
        q = self.dropout(q)  # Dropout after l2
        q = self.l3(q)
        q = self.dropout(q)  # Dropout after l3
        q = self.l4(q).squeeze()
        return q


class TargetFC(nn.Module):
    """
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):

        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])


class SwinBackbone(nn.Module):
    """
    Swin Transformer backbone with multi-scale feature extraction and LDA modules.
    """
    def __init__(self, lda_out_channels, in_chn, drop_path_rate=0.2, model_size='tiny'):
        super(SwinBackbone, self).__init__()
        
        # Model selection
        model_configs = {
            'tiny': {
                'name': 'swin_tiny_patch4_window7_224',
                'channels': [96, 192, 384, 768],
                'params': '~28M'
            },
            'small': {
                'name': 'swin_small_patch4_window7_224',
                'channels': [96, 192, 384, 768],
                'params': '~50M'
            },
            'base': {
                'name': 'swin_base_patch4_window7_224',
                'channels': [128, 256, 512, 1024],
                'params': '~88M'
            }
        }
        
        if model_size not in model_configs:
            raise ValueError(f"model_size must be one of {list(model_configs.keys())}")
        
        config = model_configs[model_size]
        self.channels = config['channels']
        print(f"Loading Swin Transformer {model_size.upper()} ({config['params']} parameters)")
        
        # Load Swin Transformer with features_only mode
        # drop_path_rate: Stochastic depth for regularization (0.2 recommended)
        self.backbone = timm.create_model(
            config['name'],
            pretrained=True,
            features_only=True,
            drop_path_rate=drop_path_rate,  # Enable stochastic depth
            out_indices=(0, 1, 2, 3)  # Extract all 4 stages
        )
        
        # Get feature dimensions and sizes
        # Stage 1: channels[0], 56x56
        # Stage 2: channels[1], 28x28
        # Stage 3: channels[2], 14x14
        # Stage 4: channels[3], 7x7
        
        # Local distortion aware modules for each stage (dynamic based on model size)
        # Stage 1: channels[0], 56x56 -> pool to 8x8
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(self.channels[0], 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),  # 56x56 -> 8x8
        )
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)
        
        # Stage 2: channels[1], 28x28 -> pool to 4x4
        self.lda2_pool = nn.Sequential(
            nn.Conv2d(self.channels[1], 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),  # 28x28 -> 4x4
        )
        self.lda2_fc = nn.Linear(32 * 16, lda_out_channels)
        
        # Stage 3: channels[2], 14x14 -> pool to 2x2
        self.lda3_pool = nn.Sequential(
            nn.Conv2d(self.channels[2], 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),  # 14x14 -> 2x2
        )
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)
        
        # Stage 4: channels[3], 7x7 -> pool to 1x1
        self.lda4_pool = nn.AvgPool2d(7, stride=7)  # 7x7 -> 1x1
        self.lda4_fc = nn.Linear(self.channels[3], in_chn - lda_out_channels * 3)
        
        # Initialize LDA modules
        nn.init.kaiming_normal_(self.lda1_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda2_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda3_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda1_fc.weight.data)
        nn.init.kaiming_normal_(self.lda2_fc.weight.data)
        nn.init.kaiming_normal_(self.lda3_fc.weight.data)
        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

    def forward(self, x):
        # Extract multi-scale features from Swin Transformer
        features = self.backbone(x)  # Returns list of 4 feature maps
        
        # Convert features to correct format [batch, channels, height, width]
        # Some versions/timm configurations may return [B, H, W, C], need [B, C, H, W]
        feat0 = features[0]
        feat1 = features[1]
        feat2 = features[2]
        feat3 = features[3]
        
        # Check if features are in [B, H, W, C] format and convert to [B, C, H, W]
        # Use self.channels which is dynamically set based on model_size
        # Tiny/Small: [96, 192, 384, 768], Base: [128, 256, 512, 1024]
        feats = [feat0, feat1, feat2, feat3]
        
        for i, feat in enumerate(feats):
            if len(feat.shape) == 4:
                # Check if last dimension matches expected channel count
                if feat.shape[-1] == self.channels[i] and feat.shape[1] != self.channels[i]:
                    # Convert from [B, H, W, C] to [B, C, H, W]
                    feats[i] = feat.permute(0, 3, 1, 2).contiguous()
        
        feat0, feat1, feat2, feat3 = feats
        
        # Apply LDA modules to each scale
        # Stage 1: [B, C0, 56, 56] -> pool to [B, 16, 8, 8] -> flatten -> [B, 1024] -> [B, lda_out]
        lda_1 = self.lda1_fc(self.lda1_pool(feat0).view(x.size(0), -1))
        # Stage 2: [B, C1, 28, 28] -> pool to [B, 32, 4, 4] -> flatten -> [B, 512] -> [B, lda_out]
        lda_2 = self.lda2_fc(self.lda2_pool(feat1).view(x.size(0), -1))
        # Stage 3: [B, C2, 14, 14] -> pool to [B, 64, 2, 2] -> flatten -> [B, 256] -> [B, lda_out]
        lda_3 = self.lda3_fc(self.lda3_pool(feat2).view(x.size(0), -1))
        # Stage 4: [B, C3, 7, 7] -> pool to [B, C3, 1, 1] -> flatten -> [B, C3] -> [B, in_chn - 3*lda_out]
        # (C0,C1,C2,C3 = Tiny/Small:[96,192,384,768], Base:[128,256,512,1024])
        lda_4 = self.lda4_fc(self.lda4_pool(feat3).view(x.size(0), -1))
        
        # Concatenate all LDA features to form target_in_vec
        vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)
        
        out = {}
        out['hyper_in_feat'] = feat3  # 保持向后兼容：使用最终阶段特征 [B, C3, 7, 7]
        out['hyper_in_feat_multi'] = [feat0, feat1, feat2, feat3]  # 新增：返回所有阶段特征用于多尺度融合
        out['target_in_vec'] = vec
        
        return out


def swin_backbone(lda_out_channels, in_chn, pretrained=True, drop_path_rate=0.2, model_size='tiny', **kwargs):
    """Constructs a Swin Transformer backbone.

    Args:
        lda_out_channels: output channels for each LDA module
        in_chn: total input channels for target network (sum of all LDA outputs)
        model_size: 'tiny' (28M), 'small' (50M), or 'base' (88M)
        pretrained (bool): If True, uses pretrained weights from ImageNet
        drop_path_rate (float): Stochastic depth rate for regularization
    """
    model = SwinBackbone(lda_out_channels, in_chn, drop_path_rate=drop_path_rate, model_size=model_size)
    return model


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

