import torch
import tensorly as tl
from tensorly.decomposition import parafac

def tensor_decomposition_hash(features, rank=16):
    """
    基于CP分解的哈希生成
    参数:
        features: 形状为[N, D]的特征矩阵
        rank: 分解秩
    返回:
        hash_codes: 二值哈希码[N, rank]
    """
    # 构建3阶张量（添加虚拟维度）
    tensor = features.unsqueeze(2)  # [N, D, 1]
    
    # CP分解
    factors = parafac(tensor, rank=rank)
    
    # 取第一个因子矩阵作为哈希
    hash_continuous = factors[0]
    return torch.sign(hash_continuous)