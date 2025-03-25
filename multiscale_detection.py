import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

class MultiscaleFeatureExtractor:
    def __init__(self, scales=[0.75, 1.0, 1.25], base_model="resnet50"):
        """
        初始化多尺度特征提取器
        
        参数:
            scales (list): 图像缩放比例列表
            base_model (str): 基础模型，支持"resnet50"和"vit"
        """
        self.scales = scales
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
        
        return features
    
    def extract_multiscale_features(self, img):
        """
        从图像中提取多尺度特征
        
        参数:
            img: PIL图像或张量
            
        返回:
            多尺度特征向量
        """
        all_features = []
        
        # 如果输入是张量，转换为PIL图像以便于缩放
        if isinstance(img, torch.Tensor):
            if img.dim() == 4:  # 批次形式
                img = img[0]  # 取第一个样本
            # 转换为PIL图像
            img = transforms.ToPILImage()(img.cpu())
        
        # 对每个尺度提取特征
        for i, scale in enumerate(self.scales):
            # 调整图像大小
            if scale != 1.0:
                w, h = img.size
                new_w, new_h = int(w * scale), int(h * scale)
                scaled_img = img.resize((new_w, new_h), Image.LANCZOS)
            else:
                scaled_img = img
            
            # 提取特征
            features = self.extract_features(scaled_img)
            
            # 归一化特征
            features = features / (np.linalg.norm(features) + 1e-8)
            
            all_features.append(features)
        
        # 加权融合特征 - 给中心尺度(1.0)更高的权重
        center_idx = self.scales.index(1.0) if 1.0 in self.scales else len(self.scales) // 2
        weights = np.ones(len(self.scales)) * 0.15
        weights[center_idx] = 0.7  # 给中心尺度更高的权重
        weights = weights / weights.sum()  # 归一化权重
        
        # 应用权重
        combined_features = np.zeros_like(all_features[0])
        for i, feat in enumerate(all_features):
            combined_features += weights[i] * feat
        
        # 再次归一化
        combined_features = combined_features / (np.linalg.norm(combined_features) + 1e-8)
        
        return combined_features

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
        # 使用ViT作为基础模型，性能更好
        multiscale_hash.extractor = MultiscaleFeatureExtractor(
            scales=[0.75, 1.0, 1.25], 
            base_model="vit"
        )
    
    # 提取多尺度特征
    features = multiscale_hash.extractor.extract_multiscale_features(img)
    
    # 如果需要，可以使用PCA或其他方法降维到指定的hash_size
    # 这里简单地取前hash_size*hash_size个元素
    if hash_size * hash_size < len(features):
        features = features[:hash_size * hash_size]
    elif hash_size * hash_size > len(features):
        # 如果特征维度小于所需哈希位数，通过重复填充
        repeats = int(np.ceil((hash_size * hash_size) / len(features)))
        features = np.tile(features, repeats)[:hash_size * hash_size]
    
    # 使用自适应阈值生成二进制哈希
    # 计算特征的中值作为阈值
    threshold = np.median(features)
    hash_value = features > threshold
    
    return hash_value

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
        # 使用ViT作为基础模型，性能更好
        multiscale_deep.extractor = MultiscaleFeatureExtractor(
            scales=[0.75, 1.0, 1.25], 
            base_model="vit"
        )
    
    # 提取多尺度特征
    features = multiscale_deep.extractor.extract_multiscale_features(img)
    
    # 如果指定了特征维度，截取前feature_dim个元素
    if feature_dim is not None and feature_dim < len(features):
        features = features[:feature_dim]
    
    return features

def compute_multiscale_distance(feature1, feature2):
    """
    计算两个多尺度特征向量之间的距离
    
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
    return extract_multiscale_features.extractor.extract_multiscale_features(img)