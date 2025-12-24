"""
Models for ablation study: ResNet50 baseline, +AFA, +Attention
Based on original HyperIQA architecture
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, lda_out_channels, in_chn, target_in_size, use_afa=False, use_attention=False):
        self.inplanes = 64
        self.use_afa = use_afa
        self.use_attention = use_attention
        
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_chn, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])   # 256 channels
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 512 channels
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 1024 channels
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 2048 channels
        
        # Local distortion aware (LDA) module
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7)
        )
        self.lda1_fc = nn.Linear(16 * 8 * 8, lda_out_channels)
        
        self.lda2_pool = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 4 * 4, lda_out_channels)
        
        self.lda3_pool = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 2 * 2, lda_out_channels)
        
        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(2048, lda_out_channels)
        
        # Multi-scale feature aggregation
        if use_afa:
            # Pool all to 7x7
            self.afa_pool = nn.AdaptiveAvgPool2d((7, 7))
            
            # Project to same channels for fusion
            self.afa_conv1 = nn.Conv2d(256, 256, 1)
            self.afa_conv2 = nn.Conv2d(512, 256, 1)
            self.afa_conv3 = nn.Conv2d(1024, 256, 1)
            self.afa_conv4 = nn.Conv2d(2048, 256, 1)
            
            if use_attention:
                # Channel attention for multi-scale features
                total_channels = 256 * 4  # 1024
                self.channel_attn = nn.Sequential(
                    nn.Linear(total_channels, total_channels // 4),
                    nn.ReLU(),
                    nn.Linear(total_channels // 4, total_channels),
                    nn.Sigmoid()
                )
            
            # Final projection after fusion
            self.afa_final = nn.Conv2d(1024, 2048, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages
        x1 = self.layer1(x)   # (B, 256, H/4, W/4)
        x2 = self.layer2(x1)  # (B, 512, H/8, W/8)
        x3 = self.layer3(x2)  # (B, 1024, H/16, W/16)
        x4 = self.layer4(x3)  # (B, 2048, H/32, W/32)

        # LDA (Local Distortion Aware) - from original HyperIQA
        lda_1 = self.lda1_pool(x1)
        lda_1 = lda_1.view(lda_1.size(0), -1)
        lda_1 = self.lda1_fc(lda_1)

        lda_2 = self.lda2_pool(x2)
        lda_2 = lda_2.view(lda_2.size(0), -1)
        lda_2 = self.lda2_fc(lda_2)

        lda_3 = self.lda3_pool(x3)
        lda_3 = lda_3.view(lda_3.size(0), -1)
        lda_3 = self.lda3_fc(lda_3)

        lda_4 = self.lda4_pool(x4)
        lda_4 = lda_4.view(lda_4.size(0), -1)
        lda_4 = self.lda4_fc(lda_4)

        # Concatenate LDA features
        target_in_vec = torch.cat((lda_1, lda_2, lda_3, lda_4), dim=1)

        # Multi-scale feature aggregation (AFA)
        if self.use_afa:
            # Pool all to 7x7
            f1 = self.afa_pool(x1)  # (B, 256, 7, 7)
            f2 = self.afa_pool(x2)  # (B, 512, 7, 7)
            f3 = self.afa_pool(x3)  # (B, 1024, 7, 7)
            f4 = self.afa_pool(x4)  # (B, 2048, 7, 7)
            
            # Project to same channels
            f1 = self.afa_conv1(f1)  # (B, 256, 7, 7)
            f2 = self.afa_conv2(f2)  # (B, 256, 7, 7)
            f3 = self.afa_conv3(f3)  # (B, 256, 7, 7)
            f4 = self.afa_conv4(f4)  # (B, 256, 7, 7)
            
            # Concatenate
            feat_fused = torch.cat([f1, f2, f3, f4], dim=1)  # (B, 1024, 7, 7)
            
            if self.use_attention:
                # Apply channel attention
                feat_gap = feat_fused.mean(dim=[2, 3])  # (B, 1024)
                attn = self.channel_attn(feat_gap)  # (B, 1024)
                attn = attn.unsqueeze(2).unsqueeze(3)  # (B, 1024, 1, 1)
                feat_fused = feat_fused * attn  # Element-wise multiplication
            
            # Project back to 2048 channels
            hyper_in_feat = self.afa_final(feat_fused)  # (B, 2048, 7, 7)
        else:
            # Baseline: use only layer4 output
            hyper_in_feat = x4

        out = {}
        out['target_in_vec'] = target_in_vec
        out['hyper_in_feat'] = hyper_in_feat

        return out


def resnet50_backbone(lda_out_channels, target_in_size, use_afa=False, use_attention=False, pretrained=True):
    """Construct a ResNet-50 model with optional AFA and attention"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], lda_out_channels, 3, target_in_size, 
                   use_afa=use_afa, use_attention=use_attention)
    
    if pretrained:
        # Load pretrained weights for backbone only
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        
        # Filter out keys that don't match (LDA, AFA modules are new)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    return model


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


class HyperNet(nn.Module):
    """
    Hyper network for learning perceptual rules
    Modified to support AFA and channel attention
    """
    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size, 
                 target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size, 
                 feature_size, use_afa=False, use_attention=False):
        super(HyperNet, self).__init__()

        self.hyperInChn = hyper_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size

        self.res = resnet50_backbone(lda_out_channels, target_in_size, 
                                     use_afa=use_afa, use_attention=use_attention,
                                     pretrained=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Conv layers for resnet output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )

        # Hyper network part
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

        # Initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                if hasattr(self._modules[m_name], 'weight'):
                    nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, img):
        feature_size = self.feature_size

        res_out = self.res(img)

        # Input vector for target net
        target_in_vec = res_out['target_in_vec'].view(-1, self.target_in_size, 1, 1)

        # Input features for hyper net
        hyper_in_feat = self.conv1(res_out['hyper_in_feat']).view(-1, self.hyperInChn, feature_size, feature_size)

        # Generating target net weights & biases
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
    """Target network for quality prediction"""
    def __init__(self, paras):
        super(TargetNet, self).__init__()
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
        q = self.l2(q)
        q = self.l3(q)
        q = self.l4(q)

        return q

