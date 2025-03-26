import numpy as np
import torch
from torch.nn import functional as F
from resnet_detection import ResNetFeatureExtractor
from PIL import Image

# 全局变量，避免重复初始化
_extractor = None
_global_projection_matrix = None

def get_extractor():
    global _extractor
    if _extractor is None:
        # 使用 ResNet50 的 avgpool 层作为特征提取层
        _extractor = ResNetFeatureExtractor(layer='avgpool')
    return _extractor

def get_projection_matrix(d, target_dim):
    """
    获取固定的随机投影矩阵，将原始特征维度 d 投影到目标维度 target_dim
    """
    global _global_projection_matrix
    if _global_projection_matrix is None or _global_projection_matrix.shape[0] != d or _global_projection_matrix.shape[1] != target_dim:
        np.random.seed(42)
        _global_projection_matrix = np.random.randn(d, target_dim)
    return _global_projection_matrix

def extract_multiscale_features(img, scales, hash_size):
    """
    对输入图像在不同尺度下提取 ResNet 特征，返回形状为 (n_scales, d) 的特征矩阵
    参数:
        img: 输入图像，支持 PIL.Image 或 torch.Tensor（形状为 (3, H, W)）
        scales (list): 尺度因子列表，例如 [1.0, 0.75, 0.5]
        hash_size (int): 哈希边长，用于后续降维目标维度计算（目标维度 = hash_size^2）
    返回:
        numpy.ndarray: 多尺度特征矩阵，形状 (n_scales, d)
    """
    extractor = get_extractor()
    features_list = []
    for scale in scales:
        if isinstance(img, torch.Tensor):
            # 保证输入为 (1, 3, H, W)
            img_tensor = img.unsqueeze(0) if img.dim() == 3 else img
            _, C, H, W = img_tensor.shape
            new_H = max(1, int(H * scale))
            new_W = max(1, int(W * scale))
            # 使用双线性插值缩放
            img_scaled = F.interpolate(img_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
            img_scaled = img_scaled.squeeze(0)
            feature = extractor.extract_features(img_scaled)
        elif isinstance(img, Image.Image):
            w, h = img.size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img_scaled = img.resize(new_size, Image.ANTIALIAS)
            feature = extractor.extract_features(img_scaled)
        else:
            raise ValueError("仅支持 PIL.Image 或 torch.Tensor 类型的输入")
        # 将特征展平为一维向量
        if feature.ndim > 1:
            feature = feature.flatten()
        features_list.append(feature)
    return np.stack(features_list, axis=0)

def project_features(features_matrix, hash_size):
    """
    利用固定随机投影将每个尺度的高维特征降维到目标维度（hash_size^2），并重构为张量格式
    参数:
        features_matrix: 多尺度特征矩阵，形状 (n_scales, d)
        hash_size (int): 指定哈希边长，目标降维后向量长度为 hash_size^2
    返回:
        numpy.ndarray: 降维后并重构为张量的特征，形状 (n_scales, hash_size, hash_size)
    """
    n_scales, d = features_matrix.shape
    target_dim = hash_size * hash_size
    R = get_projection_matrix(d, target_dim)  # shape: (d, target_dim)
    # 对每个尺度特征进行线性投影
    projected = np.dot(features_matrix, R)  # shape: (n_scales, target_dim)
    # 将投影后的向量重塑为 (hash_size, hash_size)
    tensor_features = projected.reshape(n_scales, hash_size, hash_size)
    return tensor_features

def fuse_tensor_features(tensor_features):
    """
    对多尺度特征张量进行张量分解，采用对模式0（尺度维度）展开后利用 SVD 得到的第一左奇异向量作为融合权重，
    对各尺度信息加权融合得到最终特征向量
    参数:
        tensor_features: 形状 (n_scales, hash_size, hash_size)
    返回:
        numpy.ndarray: 融合后的特征向量，长度为 hash_size^2
    """
    n_scales, h, w = tensor_features.shape
    # 将张量在尺度维度上展开为矩阵 A，形状 (n_scales, h*w)
    A = tensor_features.reshape(n_scales, -1)
    # 对 A 做 SVD 分解：A = U S Vt
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # 利用第一左奇异向量对各尺度进行加权融合
    fused_vector = np.dot(U[:, 0], A)  # 结果形状 (h*w,)
    return fused_vector

def multiscale_hash(img, hash_size=8, scales=[1.0, 0.75, 0.5]):
    """
    基于多尺度 ResNet 特征构造张量，再通过张量分解融合信息生成二值哈希码
    参数:
        img: 输入图像，支持 PIL.Image 或 torch.Tensor（形状为 (3, H, W)）
        hash_size (int): 哈希边长，生成的哈希码长度为 hash_size^2
        scales (list): 多尺度因子列表
    返回:
        numpy.ndarray: 二值哈希向量
    """
    # 1. 提取多尺度特征（原始高维特征）
    features_matrix = extract_multiscale_features(img, scales, hash_size)
    # 2. 利用固定随机投影降维，并构造形状为 (n_scales, hash_size, hash_size) 的张量
    tensor_features = project_features(features_matrix, hash_size)
    # 3. 对构造的张量进行分解，融合多尺度信息
    fused_feature = fuse_tensor_features(tensor_features)
    # 4. 以融合特征的中值作为阈值生成二值哈希码
    median_val = np.median(fused_feature)
    binary_hash = fused_feature > median_val
    return binary_hash

def multiscale_deep(img, scales=[1.0, 0.75, 0.5], hash_size=8):
    """
    基于多尺度 ResNet 特征融合生成归一化的深度特征向量（用于相似度计算）
    参数:
        img: 输入图像，支持 PIL.Image 或 torch.Tensor（形状为 (3, H, W)）
        scales (list): 多尺度因子列表
        hash_size (int): 用于深度特征降维时的哈希边长（主要用于维度统一）
    返回:
        numpy.ndarray: 归一化的融合深度特征向量，长度为 hash_size^2
    """
    features_matrix = extract_multiscale_features(img, scales, hash_size)
    tensor_features = project_features(features_matrix, hash_size)
    fused_feature = fuse_tensor_features(tensor_features)
    norm = np.linalg.norm(fused_feature)
    if norm > 0:
        fused_feature = fused_feature / norm
    return fused_feature

def compute_multiscale_distance(feature1, feature2):
    """
    计算两个多尺度深度特征向量之间的余弦距离
    参数:
        feature1, feature2: 深度特征向量
    返回:
        float: 余弦距离（数值越小表示越相似）
    """
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    distance = 1.0 - similarity
    return distance
