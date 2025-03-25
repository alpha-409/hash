import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

class SingleScaleFeatureExtractor:
    """单尺度特征提取器，与ResNetFeatureExtractor类似但更简洁"""
    def __init__(self, base_model="resnet50"):
        """
        初始化单尺度特征提取器
        
        参数:
            base_model (str): 基础模型，支持"resnet50"和"vit"
        """
        self.base_model = base_model
        
        # 加载预训练模型
        if base_model == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 2048
        elif base_model == "vit":
            base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            self.feature_extractor = base_model
            self.feature_dim = base_model.hidden_dim
            # 移除分类头
            self.feature_extractor.heads = nn.Identity()
        else:
            raise ValueError(f"不支持的基础模型: {base_model}")
        
        # 设置为评估模式
        self.feature_extractor.eval()
        
        # 移动到GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
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
            features = self.feature_extractor(img)
            
            # 处理不同模型的输出格式
            if isinstance(features, torch.Tensor):
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
        
        # 转换为NumPy数组
        features = features.cpu().numpy()
        
        # 如果是批次，只返回第一个样本的特征
        if features.shape[0] == 1:
            features = features.squeeze(0)
        
        # 归一化特征
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features

class DualScaleFeatureExtractor:
    """双尺度特征提取器，结合原始尺度和一个额外尺度"""
    def __init__(self, scale=0.75, base_model="resnet50"):
        """
        初始化双尺度特征提取器
        
        参数:
            scale (float): 第二个尺度的缩放比例
            base_model (str): 基础模型，支持"resnet50"和"vit"
        """
        self.scale = scale
        self.base_extractor = SingleScaleFeatureExtractor(base_model=base_model)
        
        # 添加PCA降维
        self.use_pca = True
        self.pca_fitted = False
        self.pca_dim = 512  # 降维后的维度
    
    def extract_features(self, img):
        """
        从图像中提取双尺度特征
        
        参数:
            img: PIL图像
            
        返回:
            双尺度特征向量
        """
        # 确保输入是PIL图像
        if not isinstance(img, Image.Image):
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    img = img[0]
                img = transforms.ToPILImage()(img.cpu())
        
        # 提取原始尺度特征
        original_features = self.base_extractor.extract_features(img)
        
        # 调整图像大小并提取第二个尺度的特征
        w, h = img.size
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        scaled_img = img.resize((new_w, new_h), Image.LANCZOS)
        scaled_features = self.base_extractor.extract_features(scaled_img)
        
        # 特征级联而非加权融合
        concatenated_features = np.concatenate([original_features, scaled_features])
        
        # 使用PCA降维（如果启用）
        if self.use_pca:
            if not self.pca_fitted:
                # 第一次使用时初始化PCA
                from sklearn.decomposition import PCA
                self.pca = PCA(n_components=self.pca_dim)
                self.pca.fit(concatenated_features.reshape(1, -1))
                self.pca_fitted = True
            
            # 应用PCA降维
            concatenated_features = self.pca.transform(concatenated_features.reshape(1, -1)).flatten()
        
        # 归一化
        concatenated_features = concatenated_features / (np.linalg.norm(concatenated_features) + 1e-8)
        
        return concatenated_features

# 单尺度哈希函数
def singlescale_hash(img, hash_size=8):
    """
    使用单尺度特征提取并生成哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小
        
    返回:
        二进制哈希值
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(singlescale_hash, 'extractor'):
        singlescale_hash.extractor = SingleScaleFeatureExtractor(base_model="resnet50")
    
    # 提取特征
    features = singlescale_hash.extractor.extract_features(img)
    
    # 如果需要，可以使用PCA或其他方法降维到指定的hash_size
    # 这里简单地取前hash_size*hash_size个元素
    if hash_size * hash_size < len(features):
        features = features[:hash_size * hash_size]
    elif hash_size * hash_size > len(features):
        # 如果特征维度小于所需哈希位数，通过重复填充
        repeats = int(np.ceil((hash_size * hash_size) / len(features)))
        features = np.tile(features, repeats)[:hash_size * hash_size]
    
    # 使用自适应阈值生成二进制哈希
    threshold = np.median(features)
    hash_value = features > threshold
    
    return hash_value

# 单尺度深度特征函数
def singlescale_deep(img, feature_dim=None):
    """
    使用单尺度特征提取用于相似度计算
    
    参数:
        img: 输入图像
        feature_dim: 特征维度，默认为None（使用完整特征）
        
    返回:
        特征向量
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(singlescale_deep, 'extractor'):
        singlescale_deep.extractor = SingleScaleFeatureExtractor(base_model="resnet50")
    
    # 提取特征
    features = singlescale_deep.extractor.extract_features(img)
    
    # 如果指定了特征维度，截取前feature_dim个元素
    if feature_dim is not None and feature_dim < len(features):
        features = features[:feature_dim]
    
    return features

# 双尺度哈希函数
def dualscale_hash(img, hash_size=8):
    """
    使用双尺度特征提取并生成哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小
        
    返回:
        二进制哈希值
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(dualscale_hash, 'extractor'):
        dualscale_hash.extractor = DualScaleFeatureExtractor(scale=0.75, base_model="resnet50")
    
    # 提取双尺度特征
    features = dualscale_hash.extractor.extract_features(img)
    
    # 如果需要，可以使用PCA或其他方法降维到指定的hash_size
    # 这里简单地取前hash_size*hash_size个元素
    if hash_size * hash_size < len(features):
        features = features[:hash_size * hash_size]
    elif hash_size * hash_size > len(features):
        # 如果特征维度小于所需哈希位数，通过重复填充
        repeats = int(np.ceil((hash_size * hash_size) / len(features)))
        features = np.tile(features, repeats)[:hash_size * hash_size]
    
    # 使用自适应阈值生成二进制哈希
    threshold = np.median(features)
    hash_value = features > threshold
    
    return hash_value

# 双尺度深度特征函数
def dualscale_deep(img, feature_dim=None):
    """
    使用双尺度特征提取用于相似度计算
    
    参数:
        img: 输入图像
        feature_dim: 特征维度，默认为None（使用完整特征）
        
    返回:
        特征向量
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(dualscale_deep, 'extractor'):
        dualscale_deep.extractor = DualScaleFeatureExtractor(scale=0.75, base_model="resnet50")
    
    # 提取双尺度特征
    features = dualscale_deep.extractor.extract_features(img)
    
    # 如果指定了特征维度，截取前feature_dim个元素
    if feature_dim is not None and feature_dim < len(features):
        features = features[:feature_dim]
    
    return features

# 多尺度特征提取器（使用注意力机制融合特征）
class MultiscaleFeatureExtractor:
    def __init__(self, scales=[0.75, 1.0, 1.25], base_model="resnet50"):
        """
        初始化多尺度特征提取器
        
        参数:
            scales (list): 图像缩放比例列表
            base_model (str): 基础模型，支持"resnet50"和"vit"
        """
        self.scales = scales
        self.base_extractor = SingleScaleFeatureExtractor(base_model=base_model)
        
        # 特征维度
        self.feature_dim = self.base_extractor.feature_dim
        
        # 注意力机制参数
        self.attention_weights = None
        
        # 是否使用PCA降维
        self.use_pca = False
        self.pca_fitted = False
        self.pca_dim = 512  # 降维后的维度
    
    def attention_fusion(self, features_list):
        """
        使用注意力机制融合多个特征向量
        
        参数:
            features_list: 特征向量列表
            
        返回:
            融合后的特征向量
        """
        # 特征数量
        n_features = len(features_list)
        
        # 计算每个特征向量的注意力分数
        # 这里使用特征向量的L2范数作为注意力分数的一部分
        attention_scores = np.zeros(n_features)
        for i, features in enumerate(features_list):
            # 计算特征向量的L2范数
            norm = np.linalg.norm(features)
            # 使用softplus激活函数确保分数为正
            attention_scores[i] = np.log(1 + np.exp(norm))
        
        # 使用softmax将分数转换为权重
        exp_scores = np.exp(attention_scores)
        self.attention_weights = exp_scores / np.sum(exp_scores)
        
        # 加权融合特征
        fused_features = np.zeros_like(features_list[0])
        for i, features in enumerate(features_list):
            fused_features += self.attention_weights[i] * features
        
        # 归一化融合后的特征
        fused_features = fused_features / (np.linalg.norm(fused_features) + 1e-8)
        
        return fused_features
    
    def extract_features(self, img):
        """
        从图像中提取多尺度特征
        
        参数:
            img: PIL图像
            
        返回:
            多尺度特征向量
        """
        # 确保输入是PIL图像
        if not isinstance(img, Image.Image):
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    img = img[0]
                img = transforms.ToPILImage()(img.cpu())
        
        # 对每个尺度提取特征
        all_features = []
        for scale in self.scales:
            # 调整图像大小
            if scale != 1.0:
                w, h = img.size
                new_w, new_h = int(w * scale), int(h * scale)
                scaled_img = img.resize((new_w, new_h), Image.LANCZOS)
            else:
                scaled_img = img
            
            # 提取特征
            features = self.base_extractor.extract_features(scaled_img)
            all_features.append(features)
        
        # 使用注意力机制融合特征
        fused_features = self.attention_fusion(all_features)
        
        # 如果需要，使用PCA降维
        if self.use_pca:
            if not self.pca_fitted:
                # 第一次使用时初始化PCA
                from sklearn.decomposition import PCA
                self.pca = PCA(n_components=self.pca_dim)
                self.pca.fit(fused_features.reshape(1, -1))
                self.pca_fitted = True
            
            # 应用PCA降维
            fused_features = self.pca.transform(fused_features.reshape(1, -1)).flatten()
        
        # 归一化
        fused_features = fused_features / (np.linalg.norm(fused_features) + 1e-8)
        
        return fused_features

# 多尺度哈希函数（保留原有实现，但使用新的多尺度提取器）
def multiscale_hash(img, hash_size=8):
    """
    使用多尺度特征提取并生成哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小
        
    返回:
        二进制哈希值
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(multiscale_hash, 'extractor'):
        multiscale_hash.extractor = MultiscaleFeatureExtractor(
            scales=[0.75, 1.0, 1.25], 
            base_model="resnet50"
        )
    
    # 提取多尺度特征
    features = multiscale_hash.extractor.extract_features(img)
    
    # 如果需要，可以使用PCA或其他方法降维到指定的hash_size
    # 这里简单地取前hash_size*hash_size个元素
    if hash_size * hash_size < len(features):
        features = features[:hash_size * hash_size]
    elif hash_size * hash_size > len(features):
        # 如果特征维度小于所需哈希位数，通过重复填充
        repeats = int(np.ceil((hash_size * hash_size) / len(features)))
        features = np.tile(features, repeats)[:hash_size * hash_size]
    
    # 使用自适应阈值生成二进制哈希
    threshold = np.median(features)
    hash_value = features > threshold
    
    return hash_value

# 多尺度深度特征函数（保留原有实现，但使用新的多尺度提取器）
def multiscale_deep(img, feature_dim=None):
    """
    使用多尺度特征提取用于相似度计算
    
    参数:
        img: 输入图像
        feature_dim: 特征维度，默认为None（使用完整特征）
        
    返回:
        特征向量
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(multiscale_deep, 'extractor'):
        multiscale_deep.extractor = MultiscaleFeatureExtractor(
            scales=[0.75, 1.0, 1.25], 
            base_model="resnet50"
        )
    
    # 提取多尺度特征
    features = multiscale_deep.extractor.extract_features(img)
    
    # 如果指定了特征维度，截取前feature_dim个元素
    if feature_dim is not None and feature_dim < len(features):
        features = features[:feature_dim]
    
    return features

# 计算特征距离的函数（适用于所有尺度）
def compute_multiscale_distance(feature1, feature2):
    """
    计算两个特征向量之间的距离
    
    参数:
        feature1, feature2: 特征向量
        
    返回:
        距离值（越小表示越相似）
    """
    # 确保特征已归一化
    feature1 = feature1 / (np.linalg.norm(feature1) + 1e-8)
    feature2 = feature2 / (np.linalg.norm(feature2) + 1e-8)
    
    # 使用余弦距离（1 - 余弦相似度）
    similarity = np.dot(feature1, feature2)
    # 转换为距离（值越小表示越相似）
    distance = 1.0 - similarity
    return distance

# 为了兼容性保留原函数，但内部使用新的实现
def extract_multiscale_features(img, scales=[0.75, 1.0, 1.25]):
    """
    提取多尺度特征
    
    参数:
        img: 输入图像
        scales: 尺度列表
        
    返回:
        多尺度特征向量
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(extract_multiscale_features, 'extractor'):
        extract_multiscale_features.extractor = MultiscaleFeatureExtractor(scales=scales)
    
    # 使用新的实现提取多尺度特征
    return extract_multiscale_features.extractor.extract_features(img)