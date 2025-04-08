import torch
import torch.nn as nn
from saliency_detector import SaliencyRegionDetector  # 修改为绝对导入
from feature_extractor import DynamicFeatureExtractor  # 修改为绝对导入
from hash_layer import CPHashLayer  # 修改为绝对导入

class ScreenHashModel(nn.Module):
    def __init__(self, hash_length=128):
        super().__init__()
        self.srd = SaliencyRegionDetector()
        self.dfe = DynamicFeatureExtractor()
        self.cphash = CPHashLayer(hash_length=hash_length)
        
    def forward(self, img):
        # 显著性检测
        saliency_map = self.srd(img)
        
        # 显著性加权输入
        weighted_img = img * saliency_map
        
        # 特征提取
        features = self.dfe(weighted_img)
        
        # 哈希码生成
        binary_hash, hash_logits = self.cphash(features)
        
        return binary_hash, hash_logits, saliency_map

class ScreenHashLoss(nn.Module):
    def __init__(self, lambda_sim=1.0, lambda_quant=0.5, lambda_balance=0.1, margin=1.0):
        super().__init__()
        self.lambda_sim = lambda_sim
        self.lambda_quant = lambda_quant
        self.lambda_balance = lambda_balance
        self.margin = margin
        
    def forward(self, hash_logits, binary_hash, labels):
        """
        输入:
            hash_logits: MLP输出 (B x L)
            binary_hash: 二值哈希码 (B x L)
            labels: 相似性矩阵 (B x B), 1表示相似, -1表示不相似
        """
        B = binary_hash.size(0)
        
        # 相似性保持损失
        sim_loss = 0
        for i in range(B):
            for j in range(B):
                if i != j:
                    dot_product = torch.dot(binary_hash[i], binary_hash[j])
                    sim_loss += torch.clamp(self.margin - labels[i,j] * dot_product, min=0)**2
        sim_loss /= B * (B - 1)  # 归一化
        
        # 二值量化损失
        quant_loss = torch.mean(torch.abs(torch.abs(hash_logits) - 1))
        
        # 比特平衡损失
        balance_loss = torch.mean(binary_hash, dim=0).pow(2).sum()
        
        # 总损失
        total_loss = (self.lambda_sim * sim_loss + 
                     self.lambda_quant * quant_loss + 
                     self.lambda_balance * balance_loss)
        
        return total_loss, {
            'sim_loss': sim_loss.item(),
            'quant_loss': quant_loss.item(),
            'balance_loss': balance_loss.item()
        }