import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DynamicFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet18'):
        super().__init__()
        # 细节路径
        self.detail_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 语义路径
        if backbone == 'resnet18':
            self.semantic_path = models.resnet18(pretrained=True)
            self.semantic_path = nn.Sequential(*list(self.semantic_path.children())[:-2])
            self.semantic_out_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # 可变形注意力门控
        self.dag = DeformableAttentionGate(64, self.semantic_out_channels)
        
    def forward(self, img):
        # 细节路径
        F_d = self.detail_path(img)  # B x 64 x H x W
        
        # 语义路径
        F_s = self.semantic_path(img)  # B x 512 x H/8 x W/8
        
        # 上采样语义特征
        F_s_up = F.interpolate(F_s, size=F_d.shape[2:], mode='bilinear', align_corners=False)
        
        # 可变形注意力融合
        attention = self.dag(F_d, F_s_up)
        F_fused = F_s_up * attention + F_d * (1 - attention)
        
        return F_fused

class DeformableAttentionGate(nn.Module):
    def __init__(self, detail_channels, semantic_channels):
        super().__init__()
        self.conv_offset = nn.Conv2d(detail_channels + semantic_channels, 18, kernel_size=3, padding=1)
        self.conv_attention = nn.Conv2d(detail_channels + semantic_channels, 1, kernel_size=3, padding=1)
        
    def forward(self, F_d, F_s_up):
        # 拼接特征
        x = torch.cat([F_d, F_s_up], dim=1)
        
        # 预测偏移量
        offset = self.conv_offset(x)
        
        # 可变形卷积
        attention = self.conv_attention(x)
        attention = torch.sigmoid(attention)  # 0-1之间的注意力权重
        
        return attention