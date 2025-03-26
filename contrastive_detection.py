import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image

# ---------------------------
# 对比学习特征提取器定义
# ---------------------------
class ContrastiveFeatureExtractor(nn.Module):
    """
    基于对比学习的特征提取器：
    使用预训练ResNet50去掉最后全连接层作为骨干，并增加投影头，
    得到归一化的对比学习嵌入向量。
    """
    def __init__(self, base_model_name='resnet50', projection_dim=128):
        super(ContrastiveFeatureExtractor, self).__init__()
        # 加载预训练的ResNet50模型
        backbone = models.__dict__[base_model_name](weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 移除最后全连接层（保留avgpool后的特征）
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features  # 2048
        # 构造投影头：两层MLP
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, projection_dim)
        )
    
    def forward(self, x):
        """
        参数:
            x: 输入张量，形状 (batch_size, 3, H, W)
        返回:
            projections: 归一化后的投影向量，形状 (batch_size, projection_dim)
            features: 基础特征向量，形状 (batch_size, feature_dim)
        """
        features = self.feature_extractor(x)   # (batch_size, feature_dim, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, feature_dim)
        projections = self.projection_head(features)    # (batch_size, projection_dim)
        projections = F.normalize(projections, dim=1)
        return projections, features

# 全局变量保存对比学习模型，避免重复初始化
_contrastive_extractor = None

def get_contrastive_extractor():
    global _contrastive_extractor
    if _contrastive_extractor is None:
        # 这里默认构造一个对比学习特征提取器，投影维度设为128
        _contrastive_extractor = ContrastiveFeatureExtractor(projection_dim=128)
        _contrastive_extractor.eval()
        # 若有GPU则移动到GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _contrastive_extractor.to(device)
    return _contrastive_extractor

# ---------------------------
# 随机投影矩阵（固定全局）定义
# ---------------------------
_global_projection_matrix = None
def get_projection_matrix(d, target_dim):
    """
    获取固定随机投影矩阵，将原始维度 d 投影到目标维度 target_dim
    """
    global _global_projection_matrix
    if _global_projection_matrix is None or _global_projection_matrix.shape != (d, target_dim):
        np.random.seed(42)
        _global_projection_matrix = np.random.randn(d, target_dim)
    return _global_projection_matrix

# ---------------------------
# 多尺度对比特征提取与张量构造
# ---------------------------
def extract_multiscale_contrastive_features(img, scales, hash_size):
    """
    对输入图像在不同尺度下使用对比学习模型提取特征，
    返回形状为 (n_scales, d) 的特征矩阵，其中 d为对比嵌入维度（例如128）
    
    参数:
        img: 输入图像，支持 PIL.Image 或 torch.Tensor（形状 (3, H, W)）
        scales: 尺度因子列表，例如 [1.0, 0.75, 0.5]
        hash_size: 哈希边长，用于后续降维（目标维度 = hash_size^2）
    返回:
        numpy.ndarray: 多尺度特征矩阵，形状 (n_scales, d)
    """
    extractor = get_contrastive_extractor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_list = []
    for scale in scales:
        # 对于张量输入：利用 F.interpolate 进行缩放
        if isinstance(img, torch.Tensor):
            img_tensor = img.unsqueeze(0) if img.dim() == 3 else img
            _, C, H, W = img_tensor.shape
            new_H = max(1, int(H * scale))
            new_W = max(1, int(W * scale))
            img_scaled = F.interpolate(img_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
            img_scaled = img_scaled.squeeze(0)
            # 需添加 batch 维度
            img_scaled = img_scaled.unsqueeze(0).to(device)
        # 对于 PIL.Image 输入，转换为张量并resize
        elif isinstance(img, Image.Image):
            new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
            img_scaled = img.resize(new_size, Image.ANTIALIAS)
            # 使用与ResNet预处理一致的方式，这里简单转为tensor（注意需归一化可根据实际训练时设置）
            transform = models.ResNet50_Weights.IMAGENET1K_V1.transforms()
            img_scaled = transform(img_scaled).unsqueeze(0).to(device)
        else:
            raise ValueError("输入类型不支持，仅支持 PIL.Image 或 torch.Tensor")
        
        with torch.no_grad():
            projections, _ = extractor(img_scaled)  # 使用投影向量作为对比特征
        # 得到 shape (1, projection_dim)，取第一个样本并转换为 numpy 数组
        feat = projections[0].cpu().numpy()
        features_list.append(feat)
    return np.stack(features_list, axis=0)

def project_features(features_matrix, hash_size):
    """
    利用固定随机投影将每个尺度的对比特征降维到目标维度（hash_size^2），
    并重塑为二维形式构成张量
    
    参数:
        features_matrix: 形状 (n_scales, d)
        hash_size: 指定哈希边长，目标降维后向量长度为 hash_size^2
    返回:
        numpy.ndarray: 张量特征，形状 (n_scales, hash_size, hash_size)
    """
    n_scales, d = features_matrix.shape
    target_dim = hash_size * hash_size
    R = get_projection_matrix(d, target_dim)  # (d, target_dim)
    projected = np.dot(features_matrix, R)  # (n_scales, target_dim)
    tensor_features = projected.reshape(n_scales, hash_size, hash_size)
    return tensor_features

def fuse_tensor_features(tensor_features):
    """
    对多尺度张量特征进行张量分解：
    将张量在尺度维度上展开为矩阵后，利用 SVD 得到第一左奇异向量作为各尺度加权，
    融合得到最终特征向量
    
    参数:
        tensor_features: 形状 (n_scales, hash_size, hash_size)
    返回:
        numpy.ndarray: 融合后的特征向量，长度为 hash_size^2
    """
    n_scales, h, w = tensor_features.shape
    A = tensor_features.reshape(n_scales, -1)  # (n_scales, h*w)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    fused_vector = np.dot(U[:, 0], A)  # (h*w,)
    return fused_vector

# ---------------------------
# 基于对比学习的哈希与深度特征函数
# ---------------------------
def contrastive_hash(img, hash_size=8, scales=[1.0, 0.75, 0.5]):
    """
    基于对比学习特征提取、张量构造与分解生成二值哈希码
    
    参数:
        img: 输入图像（PIL.Image 或 torch.Tensor）
        hash_size: 哈希边长（最终哈希长度 = hash_size^2）
        scales: 多尺度缩放因子列表
    返回:
        numpy.ndarray: 二值哈希向量
    """
    # 1. 多尺度提取对比特征，得到 shape (n_scales, projection_dim)
    features_matrix = extract_multiscale_contrastive_features(img, scales, hash_size)
    # 2. 固定随机投影降维，并构造张量 (n_scales, hash_size, hash_size)
    tensor_features = project_features(features_matrix, hash_size)
    # 3. 张量分解融合多尺度信息，得到最终融合向量（长度 = hash_size^2）
    fused_feature = fuse_tensor_features(tensor_features)
    # 4. 以融合特征中值作为阈值二值化生成哈希码
    median_val = np.median(fused_feature)
    binary_hash = fused_feature > median_val
    return binary_hash

def contrastive_deep(img, hash_size=8, scales=[1.0, 0.75, 0.5]):
    """
    基于对比学习特征融合生成归一化的深度特征向量，用于相似度计算
    
    参数:
        img: 输入图像（PIL.Image 或 torch.Tensor）
        hash_size: 用于降维时的哈希边长（输出长度 = hash_size^2）
        scales: 多尺度缩放因子列表
    返回:
        numpy.ndarray: 归一化的深度特征向量
    """
    features_matrix = extract_multiscale_contrastive_features(img, scales, hash_size)
    tensor_features = project_features(features_matrix, hash_size)
    fused_feature = fuse_tensor_features(tensor_features)
    norm = np.linalg.norm(fused_feature)
    if norm > 0:
        fused_feature = fused_feature / norm
    return fused_feature

def compute_contrastive_distance(feature1, feature2):
    """
    计算两个深度特征向量的余弦距离（1 - 余弦相似度）
    
    参数:
        feature1, feature2: 深度特征向量
    返回:
        float: 余弦距离（值越小表示越相似）
    """
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    distance = 1.0 - similarity
    return distance

# ---------------------------
# 示例（可用于调试）
# ---------------------------
if __name__ == '__main__':
    # 构造一个示例输入（随机图像张量，形状 (3, 224, 224)）
    dummy_img = torch.randn(3, 224, 224)
    
    # 生成对比学习哈希码
    hash_code = contrastive_hash(dummy_img, hash_size=8, scales=[1.0, 0.75, 0.5])
    print("Contrastive Hash Code:", hash_code)
    
    # 提取深度特征
    deep_feature = contrastive_deep(dummy_img, hash_size=8, scales=[1.0, 0.75, 0.5])
    print("Contrastive Deep Feature (normalized):", deep_feature)
    
    # 计算距离（示例：自己与自己距离应为0）
    distance = compute_contrastive_distance(deep_feature, deep_feature)
    print("Distance (self):", distance)
