import torch
import torch.nn as nn
from torchvision import models

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, anchor, positive, negative):
        pos_sim = self.cos(anchor, positive)
        neg_sim = self.cos(anchor, negative)
        losses = torch.relu(neg_sim - pos_sim + self.margin)
        return losses.mean()
class ContrastiveHash(nn.Module):
    def __init__(self, backbone='resnet50', hash_dim=64):
        super().__init__()
        # 骨干网络
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # 移除最后的全连接层
        
        # 哈希生成层
        self.hash_layer = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, hash_dim),
            nn.Tanh()  # 输出范围[-1,1]便于二值化
        )
        
        # 特征投影头（对比学习）
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x, return_hash=True):
        features = self.backbone(x)
        
        # 返回哈希码
        if return_hash:
            hash_code = self.hash_layer(features)
            return torch.sign(hash_code)  # 二值化
        
        # 返回对比特征
        return self.projection(features)