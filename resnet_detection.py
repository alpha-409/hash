import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

class ResNetFeatureExtractor:
    def __init__(self, layer='avgpool'):
        """
        初始化ResNet50特征提取器
        
        参数:
            layer (str): 提取特征的层，默认为'avgpool'
        """
        # 加载预训练的ResNet50模型
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        
        # 移动到GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # 设置提取特征的层
        self.layer = layer
        self.features = None
        
        # 注册钩子函数来获取指定层的输出
        if layer == 'avgpool':
            self.model.avgpool.register_forward_hook(self._get_features_hook)
        elif layer == 'fc':
            self.model.fc.register_forward_hook(self._get_features_hook)
        else:
            raise ValueError(f"不支持的层: {layer}")
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def _get_features_hook(self, module, input, output):
        """
        钩子函数，用于获取中间层的输出
        """
        self.features = output
    
    def extract_features(self, img):
        """
        从图像中提取特征
        
        参数:
            img: PIL图像或张量
            
        返回:
            特征向量
        """
        # 如果输入是PIL图像，进行预处理
        if isinstance(img, Image.Image):
            img = self.preprocess(img)
        
        # 如果输入是张量但不是批次形式，添加批次维度
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            img = img.unsqueeze(0)
        
        # 移动到相应设备
        img = img.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(img)
        
        # 获取特征
        features = self.features.cpu().numpy()
        
        # 如果是批次，只返回第一个样本的特征
        if features.shape[0] == 1:
            features = features.squeeze(0)
        
        return features

# 方法1: ResNet-Hash (基于哈希的方法)
def resnet_hash(img, hash_size=8):
    """
    使用ResNet50提取特征并生成哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小（这里表示最终特征的维度）
        
    返回:
        二进制哈希值
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(resnet_hash, 'extractor'):
        resnet_hash.extractor = ResNetFeatureExtractor(layer='avgpool')
    
    # 提取特征
    features = resnet_hash.extractor.extract_features(img)
    
    # 将特征展平
    features = features.flatten()
    
    # 如果需要，可以使用PCA或其他方法降维到指定的hash_size
    # 这里简单地取前hash_size*hash_size个元素
    if hash_size * hash_size < len(features):
        features = features[:hash_size * hash_size]
    
    # 计算特征的中值
    median_value = np.median(features)
    
    # 生成二进制哈希
    hash_value = features > median_value
    
    return hash_value

# 方法2: ResNet-Deep (基于特征相似度的方法)
def resnet_deep(img, feature_dim=2048):
    """
    使用ResNet50提取深度特征用于相似度计算
    
    参数:
        img: 输入图像
        feature_dim: 特征维度，默认为2048（ResNet50的avgpool输出维度）
        
    返回:
        特征向量
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(resnet_deep, 'extractor'):
        resnet_deep.extractor = ResNetFeatureExtractor(layer='avgpool')
    
    # 提取特征
    features = resnet_deep.extractor.extract_features(img)
    
    # 将特征展平并归一化
    features = features.flatten()
    features = features / np.linalg.norm(features)
    
    return features

# ResNet-Deep的距离计算函数
def compute_resnet_deep_distance(feature1, feature2):
    """
    计算两个ResNet-Deep特征向量之间的距离
    
    参数:
        feature1, feature2: 特征向量
        
    返回:
        距离值（越小表示越相似）
    """
    # 使用余弦距离（1 - 余弦相似度）
    similarity = np.dot(feature1, feature2)
    # 转换为距离（值越小表示越相似）
    distance = 1.0 - similarity
    return distance

# 方法3: 基于张量分解的深度特征哈希
def tensor_decomposition_hash(img, hash_size=8):
    """
    使用张量分解技术从深度特征生成哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小
        
    返回:
        二进制哈希值
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(tensor_decomposition_hash, 'extractor'):
        tensor_decomposition_hash.extractor = ResNetFeatureExtractor(layer='avgpool')
    
    # 提取特征
    features = tensor_decomposition_hash.extractor.extract_features(img)
    
    # 将特征重塑为张量形式 (假设特征维度为2048，重塑为16x16x8)
    tensor_shape = (16, 16, 8)
    if np.prod(tensor_shape) > features.size:
        # 如果特征不够，通过复制填充
        repeats = int(np.ceil(np.prod(tensor_shape) / features.size))
        features = np.tile(features.flatten(), repeats)[:np.prod(tensor_shape)]
    else:
        features = features.flatten()[:np.prod(tensor_shape)]
    
    tensor_features = features.reshape(tensor_shape)
    
    # 使用高阶SVD (Tucker分解)进行张量分解
    from tensorly.decomposition import tucker
    import tensorly as tl
    
    # 设置张量库后端为numpy
    tl.set_backend('numpy')
    
    # Tucker分解
    ranks = [hash_size, hash_size, min(8, hash_size)]
    core, factors = tucker(tensor_features, rank=ranks)
    
    # 使用核心张量和因子矩阵的组合生成哈希
    # 方法1: 使用核心张量的符号
    core_flattened = core.flatten()
    hash_from_core = core_flattened > np.median(core_flattened)
    
    # 方法2: 使用因子矩阵的组合
    combined_factors = np.concatenate([f.flatten() for f in factors])
    if len(combined_factors) > hash_size * hash_size:
        combined_factors = combined_factors[:hash_size * hash_size]
    elif len(combined_factors) < hash_size * hash_size:
        # 填充到所需大小
        repeats = int(np.ceil((hash_size * hash_size) / len(combined_factors)))
        combined_factors = np.tile(combined_factors, repeats)[:hash_size * hash_size]
    
    hash_from_factors = combined_factors > np.median(combined_factors)
    
    # 将两种哈希结合
    if len(hash_from_core) >= hash_size * hash_size:
        final_hash = hash_from_core[:hash_size * hash_size]
    else:
        # 组合两种哈希
        combined_hash = np.concatenate([hash_from_core, hash_from_factors])
        final_hash = combined_hash[:hash_size * hash_size]
    
    return final_hash

# 方法4: 基于自编码器的深度特征哈希
def autoencoder_hash(img, hash_size=8):
    """
    使用自编码器从深度特征生成哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小
        
    返回:
        二进制哈希值
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(autoencoder_hash, 'extractor'):
        autoencoder_hash.extractor = ResNetFeatureExtractor(layer='avgpool')
    
    # 提取特征
    features = autoencoder_hash.extractor.extract_features(img)
    features = features.flatten()
    
    # 使用PCA进行降维（模拟自编码器的编码过程）
    if not hasattr(autoencoder_hash, 'pca'):
        from sklearn.decomposition import PCA
        autoencoder_hash.pca = PCA(n_components=hash_size * hash_size)
        # 注意：实际应用中应该在大量样本上预先训练PCA
        # 这里仅用当前样本初始化，实际使用时应该替换为预训练的PCA
        autoencoder_hash.pca.fit(features.reshape(1, -1))
    
    # 编码
    encoded_features = autoencoder_hash.pca.transform(features.reshape(1, -1)).flatten()
    
    # 生成二进制哈希
    # 使用零均值二值化而非中值阈值，这对于PCA编码的特征更合适
    encoded_features = encoded_features - np.mean(encoded_features)
    hash_value = encoded_features > 0
    
    return hash_value

# 方法5: 基于局部敏感哈希(LSH)的深度特征哈希
def lsh_hash(img, hash_size=8):
    """
    使用局部敏感哈希(LSH)从深度特征生成哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小
        
    返回:
        二进制哈希值
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(lsh_hash, 'extractor'):
        lsh_hash.extractor = ResNetFeatureExtractor(layer='avgpool')
    
    # 提取特征
    features = lsh_hash.extractor.extract_features(img)
    features = features.flatten()
    
    # 归一化特征
    features = features / (np.linalg.norm(features) + 1e-8)
    
    # 如果随机投影矩阵尚未创建，则创建
    if not hasattr(lsh_hash, 'projection_matrix'):
        # 创建随机投影矩阵
        # 使用正态分布生成随机投影向量
        np.random.seed(42)  # 固定随机种子以确保一致性
        feature_dim = len(features)
        lsh_hash.projection_matrix = np.random.randn(hash_size * hash_size, feature_dim)
        
        # 归一化投影向量
        for i in range(lsh_hash.projection_matrix.shape[0]):
            lsh_hash.projection_matrix[i] = lsh_hash.projection_matrix[i] / np.linalg.norm(lsh_hash.projection_matrix[i])
    
    # 应用随机投影
    projections = np.dot(lsh_hash.projection_matrix, features)
    
    # 生成二进制哈希
    hash_value = projections > 0
    
    return hash_value

# 方法6: 基于迭代量化(ITQ)的深度特征哈希
def itq_hash(img, hash_size=8, n_iterations=50):
    """
    使用迭代量化(ITQ)从深度特征生成哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小
        n_iterations: 迭代次数
        
    返回:
        二进制哈希值
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(itq_hash, 'extractor'):
        itq_hash.extractor = ResNetFeatureExtractor(layer='avgpool')
    
    # 提取特征
    features = itq_hash.extractor.extract_features(img)
    features = features.flatten()
    
    # 使用PCA降维
    if not hasattr(itq_hash, 'pca'):
        from sklearn.decomposition import PCA
        itq_hash.pca = PCA(n_components=hash_size * hash_size)
        # 注意：实际应用中应该在大量样本上预训练PCA
        itq_hash.pca.fit(features.reshape(1, -1))
        
        # 初始化随机旋转矩阵
        np.random.seed(42)
        k = hash_size * hash_size
        itq_hash.R = np.random.randn(k, k)
        # 正交化旋转矩阵
        q, _ = np.linalg.qr(itq_hash.R)
        itq_hash.R = q
    
    # 应用PCA
    V = itq_hash.pca.transform(features.reshape(1, -1)).flatten()
    
    # 应用学习到的旋转
    if hasattr(itq_hash, 'R_optimal'):
        # 如果已经有最优旋转矩阵，直接应用
        V_rotated = np.dot(V, itq_hash.R_optimal)
    else:
        # 否则使用初始旋转矩阵
        # 注意：在实际应用中，应该通过迭代过程学习最优旋转矩阵
        # 这里简化处理，直接使用初始矩阵
        V_rotated = np.dot(V, itq_hash.R)
    
    # 生成二进制哈希
    hash_value = V_rotated > 0
    
    return hash_value