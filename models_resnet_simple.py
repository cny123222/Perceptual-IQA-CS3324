"""
Simplified ResNet + AFA + Attention model
Avoiding complex HyperNet dynamic weight generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimplifiedResNetIQA(nn.Module):
    """Simplified ResNet-based IQA model with optional AFA and attention"""
    
    def __init__(self, use_multiscale=False, use_attention=False, dropout_rate=0.3):
        super(SimplifiedResNetIQA, self).__init__()
        
        self.use_multiscale = use_multiscale
        self.use_attention = use_attention
        
        # Load pretrained ResNet50
        print("Loading pretrained ResNet50...")
        resnet = models.resnet50(pretrained=True)
        
        # Extract stages
        self.stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage1 = resnet.layer1  # 256 channels
        self.stage2 = resnet.layer2  # 512 channels
        self.stage3 = resnet.layer3  # 1024 channels
        self.stage4 = resnet.layer4  # 2048 channels
        
        if use_multiscale:
            print("Using multi-scale feature fusion")
            # Adaptive pooling to 7x7
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
            
            if use_attention:
                print("Using channel attention mechanism")
                # Channel attention for multi-scale features
                total_channels = 256 + 512 + 1024 + 2048
                self.attention_fc1 = nn.Linear(total_channels, total_channels // 4)
                self.attention_fc2 = nn.Linear(total_channels // 4, 4)  # 4 stages
            
            # Feature dimension after concatenation
            feat_dim = 256 + 512 + 1024 + 2048  # 3840
        else:
            print("Using single-scale (Stage 4 only)")
            feat_dim = 2048
        
        # Quality prediction head
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features
        x = self.stage0(x)
        f1 = self.stage1(x)  # (B, 256, 56, 56)
        f2 = self.stage2(f1) # (B, 512, 28, 28)
        f3 = self.stage3(f2) # (B, 1024, 14, 14)
        f4 = self.stage4(f3) # (B, 2048, 7, 7)
        
        if self.use_multiscale:
            # Pool all to 7x7
            f1_pooled = self.pool(f1)  # (B, 256, 7, 7)
            f2_pooled = self.pool(f2)  # (B, 512, 7, 7)
            f3_pooled = self.pool(f3)  # (B, 1024, 7, 7)
            # f4 is already 7x7
            
            # Global average pooling per scale
            f1_gap = F.adaptive_avg_pool2d(f1_pooled, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 256)
            f2_gap = F.adaptive_avg_pool2d(f2_pooled, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 512)
            f3_gap = F.adaptive_avg_pool2d(f3_pooled, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 1024)
            f4_gap = F.adaptive_avg_pool2d(f4, (1, 1)).squeeze(-1).squeeze(-1)        # (B, 2048)
            
            # Always concatenate features
            feat_concat = torch.cat([f1_gap, f2_gap, f3_gap, f4_gap], dim=1)  # (B, 3840)
            
            if self.use_attention:
                # Use attention as feature reweighting (applied to concatenated features)
                # Compute attention weights
                attn = self.attention_fc1(feat_concat)
                attn = F.relu(attn)
                attn_weights = self.attention_fc2(attn)  # (B, 4)
                attn_weights = F.softmax(attn_weights, dim=1)
                
                # Apply attention as a gating mechanism on the full feature
                # Expand attention to match feature dimensions
                # Create channel-wise attention mask
                attn_mask = torch.cat([
                    attn_weights[:, 0:1].repeat(1, 256),
                    attn_weights[:, 1:2].repeat(1, 512),
                    attn_weights[:, 2:3].repeat(1, 1024),
                    attn_weights[:, 3:4].repeat(1, 2048)
                ], dim=1)  # (B, 3840)
                
                feat = feat_concat * attn_mask  # Element-wise multiplication
            else:
                # Simple concatenation without attention
                feat = feat_concat
        else:
            # Single scale - only use stage 4
            feat = F.adaptive_avg_pool2d(f4, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 2048)
        
        # Predict quality score
        score = self.fc(feat)  # (B, 1)
        
        return {'score': score, 'attention_weights': None}


def create_model(use_multiscale=False, use_attention=False, dropout_rate=0.3):
    """Factory function to create model"""
    return SimplifiedResNetIQA(use_multiscale, use_attention, dropout_rate)

