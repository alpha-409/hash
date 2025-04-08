import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac

class CPHashLayer(nn.Module):
    def __init__(self, input_channels=64, hash_length=128, rank=32):
        super().__init__()
        self.hash_length = hash_length
        self.rank = rank
        
        # CP分解后的MLP
        self.mlp = nn.Sequential(
            nn.Linear(rank * 3, 256),
            nn.ReLU(),
            nn.Linear(256, hash_length)
        )
        
    def forward(self, x):
        """输入: x (Tensor) - B x C x H x W"""
        B, C, H, W = x.shape
        
        # 重塑为三阶张量 (这里简化处理，实际可以更复杂)
        X = x.reshape(B, C, H*W)  # B x C x (H*W)
        
        # CP分解 (批处理)
        factors = []
        for i in range(B):
            # 对每个样本单独进行CP分解
            weights, factors_i = parafac(X[i], rank=self.rank, init='random')
            factors.append(torch.cat([f.flatten() for f in factors_i]))
        
        factors = torch.stack(factors)  # B x (rank*3)
        
        # 通过MLP生成哈希码
        hash_logits = self.mlp(factors)
        binary_hash = torch.sign(hash_logits)  # B x hash_length
        
        return binary_hash, hash_logits  # 返回二值哈希和logits(用于训练)