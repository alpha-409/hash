import numpy as np
import torch
from torch.nn import functional as F
from resnet_detection import ResNetFeatureExtractor
from PIL import Image

# 全局的多尺度特征提取器，避免重复初始化
_extractor = None

def get_extractor():
    global _extractor
    if _extractor is None:
        # 使用 ResNet50 的 avgpool 层作为特征提取层
        _extractor = ResNetFeatureExtractor(layer='avgpool')
    return _extractor

def extract_multiscale_features(img, scales):
    """
    对输入图像在不同尺度下进行特征提取，并返回特征矩阵
    参数:
        img: 输入图像，可以是 PIL.Image 或 torch.Tensor（形状为 (3, H, W)）
        scales (list): 尺度因子列表，例如 [1.0, 0.75, 0.5]
    返回:
        numpy.ndarray: 特征矩阵，形状 (n_scales, feature_dim)
    """
    extractor = get_extractor()
    features_list = []
    for scale in scales:
        # 如果输入是张量格式
        if isinstance(img, torch.Tensor):
            # 确保输入为 (3, H, W)
            if img.dim() == 3:
                img_tensor = img.unsqueeze(0)  # 变为 (1, 3, H, W)
            else:
                img_tensor = img
            # 获取原始尺寸
            _, C, H, W = img_tensor.shape
            new_H = max(1, int(H * scale))
            new_W = max(1, int(W * scale))
            # 使用双线性插值缩放图像
            img_scaled = F.interpolate(img_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
            # 移除 batch 维度
            img_scaled = img_scaled.squeeze(0)
            # 提取特征（extract_features 内部会处理 tensor 格式）
            feature = extractor.extract_features(img_scaled)
        # 如果输入是 PIL.Image 格式
        elif isinstance(img, Image.Image):
            w, h = img.size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img_scaled = img.resize(new_size, Image.ANTIALIAS)
            feature = extractor.extract_features(img_scaled)
        else:
            raise ValueError("输入图像类型不受支持，仅支持 PIL.Image 或 torch.Tensor")
        
        # 确保特征展平为一维向量
        if feature.ndim > 1:
            feature = feature.flatten()
        features_list.append(feature)
    return np.stack(features_list, axis=0)

def fuse_features(features_matrix, hash_length):
    """
    融合多尺度特征矩阵，通过 SVD 分解得到融合表示，
    并根据目标维度（hash_length^2）截断/降维
    参数:
        features_matrix (numpy.ndarray): 多尺度特征矩阵，形状 (n_scales, d)
        hash_length (int): 指定哈希的边长，最终哈希维度为 hash_length * hash_length
    返回:
        numpy.ndarray: 融合后的特征向量
    """
    # 通过 SVD 分解得到特征分解
    U, S, Vt = np.linalg.svd(features_matrix, full_matrices=False)
    # 取第一主成分作为融合特征
    fused_feature = Vt[0]
    target_dim = hash_length * hash_length
    # 如果融合特征维度大于目标维度，则截断；否则直接返回
    if fused_feature.shape[0] > target_dim:
        fused_feature = fused_feature[:target_dim]
    return fused_feature

def multiscale_hash(img, hash_size=8, scales=[1.0, 0.75, 0.5]):
    """
    基于多尺度 ResNet 特征和矩阵分解生成二值哈希
    参数:
        img: 输入图像，支持 PIL.Image 或 torch.Tensor（形状为 (3, H, W)）
        hash_size (int): 哈希边长，生成的哈希维度为 hash_size^2
        scales (list): 多尺度因子列表
    返回:
        numpy.ndarray: 二值哈希向量
    """
    # 提取多尺度特征矩阵
    features_matrix = extract_multiscale_features(img, scales)
    # 融合多尺度特征，并降维到目标哈希维度
    fused_feature = fuse_features(features_matrix, hash_size)
    # 采用融合特征的中值作为二值化阈值
    median_val = np.median(fused_feature)
    binary_hash = fused_feature > median_val
    return binary_hash

def multiscale_deep(img, scales=[1.0, 0.75, 0.5]):
    """
    基于多尺度 ResNet 特征融合生成深度特征向量（归一化后用于相似度计算）
    参数:
        img: 输入图像，支持 PIL.Image 或 torch.Tensor（形状为 (3, H, W)）
        scales (list): 多尺度因子列表
    返回:
        numpy.ndarray: 归一化的深度融合特征向量
    """
    features_matrix = extract_multiscale_features(img, scales)
    # 融合特征时保留全部维度（这里直接使用第一主成分）
    fused_feature = fuse_features(features_matrix, hash_length=int(np.sqrt(features_matrix.shape[1])))
    # 归一化特征向量
    norm = np.linalg.norm(fused_feature)
    if norm > 0:
        fused_feature = fused_feature / norm
    return fused_feature

def compute_multiscale_distance(feature1, feature2):
    """
    计算两个多尺度深度特征之间的余弦距离
    参数:
        feature1, feature2 (numpy.ndarray): 深度特征向量
    返回:
        float: 余弦距离（值越小表示越相似）
    """
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    distance = 1.0 - similarity
    return distance
