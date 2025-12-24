"""
ResNet50 + Multi-scale + Attention
验证我们的改进（多尺度融合+通道注意力）是否对CNN backbone也有效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TargetFC(nn.Module):
    """
    Fully connection operations for target net (from original HyperIQA)
    Uses group convolution for batch-wise dynamic weights
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


class MultiScaleAttention(nn.Module):
    """
    Channel Attention for multi-scale feature fusion (same as Swin version)
    """
    def __init__(self, in_channels_list):
        super(MultiScaleAttention, self).__init__()
        self.in_channels = in_channels_list
        self.num_scales = len(in_channels_list)
        
        # Attention generation network
        self.attention_net = nn.Sequential(
            nn.Linear(in_channels_list[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_scales),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights
        for m in self.attention_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, feat_list):
        """
        Args:
            feat_list: List of 4 feature maps [feat1, feat2, feat3, feat4]
                feat1: (B, 256, 7, 7)
                feat2: (B, 512, 7, 7)
                feat3: (B, 1024, 7, 7)
                feat4: (B, 2048, 7, 7)
        
        Returns:
            fused_feat: (B, sum(channels), 7, 7) - weighted concatenated features
            attention_weights: (B, 4) - attention weights
        """
        B = feat_list[0].size(0)
        
        # Extract global representation from highest-level feature
        feat4_global = F.adaptive_avg_pool2d(feat_list[-1], (1, 1)).squeeze(-1).squeeze(-1)
        
        # Generate attention weights
        attention_weights = self.attention_net(feat4_global)  # [B, 4]
        
        # Apply attention weights to each scale
        weighted_feats = []
        for i, feat in enumerate(feat_list):
            weight = attention_weights[:, i].view(B, 1, 1, 1)
            weighted_feat = feat * weight
            weighted_feats.append(weighted_feat)
        
        # Concatenate along channel dimension
        fused_feat = torch.cat(weighted_feats, dim=1)  # [B, sum(channels), 7, 7]
        
        return fused_feat, attention_weights


class ResNetImproved(nn.Module):
    """
    ResNet50 + Multi-scale Feature Fusion + Channel Attention
    
    用来验证我们的改进是否具有普适性（不依赖于Swin Transformer）
    """
    def __init__(self, 
                 lda_out_channels=16,
                 hyper_in_channels=2048,
                 target_in_size=112,
                 target_fc1_size=224,
                 target_fc2_size=112,
                 target_fc3_size=56,
                 target_fc4_size=28,
                 feature_size=7,
                 use_multiscale=True,
                 use_attention=True,
                 dropout_rate=0.3):
        super(ResNetImproved, self).__init__()
        
        self.hyperInChn = hyper_in_channels
        self.use_multiscale = use_multiscale
        self.use_attention = use_attention
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size
        self.dropout_rate = dropout_rate
        
        # Load pretrained ResNet50
        print("Loading pretrained ResNet50...")
        resnet = models.resnet50(pretrained=True)
        
        # Extract stages
        self.stage0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.stage1 = resnet.layer1  # Output: (B, 256, 56, 56)
        self.stage2 = resnet.layer2  # Output: (B, 512, 28, 28)
        self.stage3 = resnet.layer3  # Output: (B, 1024, 14, 14)
        self.stage4 = resnet.layer4  # Output: (B, 2048, 7, 7)
        
        # Multi-scale feature processing
        if use_multiscale:
            print("Using multi-scale feature fusion")
            # Adaptive pooling to unify spatial dimensions to 7x7
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
            
            # Conv 1x1 + BN + ReLU for each stage
            self.conv1 = nn.Sequential(
                nn.Conv2d(256, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(1024, 1024, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            )
            
            if use_attention:
                print("Using channel attention mechanism")
                # Channel attention for multi-scale fusion
                self.multiscale_attention = MultiScaleAttention([256, 512, 1024, 2048])
                input_channels = 256 + 512 + 1024 + 2048  # 3840
            else:
                print("Using simple concatenation (no attention)")
                input_channels = 256 + 512 + 1024 + 2048  # 3840
        else:
            print("Using single-scale (Stage 4 only)")
            # Only use stage 4
            input_channels = 2048
        
        print(f"HyperNet input channels: {input_channels}")
        
        # Projection layer to match target_in_size
        self.projection = nn.Conv2d(input_channels, target_in_size, 1, padding=(0, 0))
        
        # HyperNet - generates weights for TargetNet
        self.conv1_hyper = nn.Sequential(
            nn.Conv2d(input_channels, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.fc1w_conv = nn.Conv2d(512, int(target_in_size * self.f1 / feature_size ** 2), 1, padding=(0, 0))
        self.fc1b_fc = nn.Linear(512, self.f1)
        
        self.fc2w_conv = nn.Conv2d(512, int(self.f1 * self.f2 / feature_size ** 2), 1, padding=(0, 0))
        self.fc2b_fc = nn.Linear(512, self.f2)
        
        self.fc3w_conv = nn.Conv2d(512, int(self.f2 * self.f3 / feature_size ** 2), 1, padding=(0, 0))
        self.fc3b_fc = nn.Linear(512, self.f3)
        
        self.fc4w_conv = nn.Conv2d(512, int(self.f3 * self.f4 / feature_size ** 2), 1, padding=(0, 0))
        self.fc4b_fc = nn.Linear(512, self.f4)
        
        # Final prediction layer
        self.fc5w_fc = nn.Linear(512, self.f4)
        self.fc5b_fc = nn.Linear(512, 1)
        
        # Global average pooling for HyperNet
        self.pool_hyper = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            out: Dictionary containing 'score' and optionally 'attention_weights'
        """
        batch_size = x.size(0)
        
        # Extract multi-scale features from ResNet50
        x = self.stage0(x)   # (B, 64, 56, 56)
        f1 = self.stage1(x)  # (B, 256, 56, 56)
        f2 = self.stage2(f1) # (B, 512, 28, 28)
        f3 = self.stage3(f2) # (B, 1024, 14, 14)
        f4 = self.stage4(f3) # (B, 2048, 7, 7)
        
        # Multi-scale fusion
        if self.use_multiscale:
            # Pool all features to 7x7
            f1_pooled = self.conv1(self.pool(f1))  # (B, 256, 7, 7)
            f2_pooled = self.conv2(self.pool(f2))  # (B, 512, 7, 7)
            f3_pooled = self.conv3(self.pool(f3))  # (B, 1024, 7, 7)
            # f4 is already 7x7
            
            if self.use_attention:
                # Apply channel attention
                feat_fused, attention_weights = self.multiscale_attention([f1_pooled, f2_pooled, f3_pooled, f4])
            else:
                # Simple concatenation
                feat_fused = torch.cat([f1_pooled, f2_pooled, f3_pooled, f4], dim=1)
                attention_weights = None
        else:
            # Single-scale (Stage 4 only)
            feat_fused = f4
            attention_weights = None
        
        # HyperNet - generate dynamic weights
        hyper_feat = self.conv1_hyper(feat_fused)  # (B, 512, 7, 7)
        
        # Generate weights and biases for TargetNet
        fc1w = self.fc1w_conv(hyper_feat).view(batch_size, self.f1, self.target_in_size, 1, 1)
        fc1b = self.fc1b_fc(self.pool_hyper(hyper_feat).squeeze(-1).squeeze(-1))
        
        fc2w = self.fc2w_conv(hyper_feat).view(batch_size, self.f2, self.f1, 1, 1)
        fc2b = self.fc2b_fc(self.pool_hyper(hyper_feat).squeeze(-1).squeeze(-1))
        
        fc3w = self.fc3w_conv(hyper_feat).view(batch_size, self.f3, self.f2, 1, 1)
        fc3b = self.fc3b_fc(self.pool_hyper(hyper_feat).squeeze(-1).squeeze(-1))
        
        fc4w = self.fc4w_conv(hyper_feat).view(batch_size, self.f4, self.f3, 1, 1)
        fc4b = self.fc4b_fc(self.pool_hyper(hyper_feat).squeeze(-1).squeeze(-1))
        
        fc5w = self.fc5w_fc(self.pool_hyper(hyper_feat).squeeze(-1).squeeze(-1))
        fc5b = self.fc5b_fc(self.pool_hyper(hyper_feat).squeeze(-1).squeeze(-1))
        
        # Prepare input for TargetNet (project to target_in_size)
        target_in = self.projection(feat_fused)  # (B, target_in_size, 7, 7)
        target_in = F.adaptive_avg_pool2d(target_in, (1, 1))  # (B, target_in_size, 1, 1)
        
        # TargetNet forward pass with dynamic weights (using grouped convolution)
        target_fc1 = TargetFC(fc1w, fc1b)(target_in)
        target_fc1 = F.relu(target_fc1)
        
        target_fc2 = TargetFC(fc2w, fc2b)(target_fc1)
        target_fc2 = F.relu(target_fc2)
        
        target_fc3 = TargetFC(fc3w, fc3b)(target_fc2)
        target_fc3 = F.relu(target_fc3)
        
        target_fc4 = TargetFC(fc4w, fc4b)(target_fc3)
        target_fc4 = F.relu(target_fc4)
        
        # Final prediction
        score = TargetFC(fc5w.view(batch_size, 1, self.f4, 1, 1), fc5b)(target_fc4).squeeze()
        
        # Return output
        out = {'score': score}
        if attention_weights is not None:
            out['attention_weights'] = attention_weights
        
        return out


def test_model():
    """Test the ResNetImproved model"""
    print("=" * 80)
    print("Testing ResNetImproved Models")
    print("=" * 80)
    
    # Test different configurations
    configs = [
        {"name": "ResNet50 Baseline (Single-scale, No attention)", 
         "use_multiscale": False, "use_attention": False},
        {"name": "ResNet50 + Multi-scale", 
         "use_multiscale": True, "use_attention": False},
        {"name": "ResNet50 + Multi-scale + Attention", 
         "use_multiscale": True, "use_attention": True},
    ]
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']}")
        print(f"{'='*80}")
        
        model = ResNetImproved(
            use_multiscale=config['use_multiscale'],
            use_attention=config['use_attention']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output score shape: {output['score'].shape}")
        if 'attention_weights' in output:
            print(f"Attention weights shape: {output['attention_weights'].shape}")
            print(f"Attention weights (sample): {output['attention_weights'][0]}")
        
        print(f"✓ Forward pass successful!")


if __name__ == '__main__':
    test_model()

