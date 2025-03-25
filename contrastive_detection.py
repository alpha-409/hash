import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image

class ContrastiveFeatureExtractor:
    def __init__(self, model_name="resnet50", pretrained=True):
        """
        初始化基于对比学习的特征提取器
        
        参数:
            model_name (str): 基础模型名称
            pretrained (bool): 是否使用预训练权重
        """
        # 加载基础模型
        if model_name == "resnet50":
            # 使用weights参数替代pretrained参数
            if pretrained:
                base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                base_model = models.resnet50(weights=None)
            # 移除最后的全连接层
            self.encoder = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == "resnet18":
            # 使用weights参数替代pretrained参数
            if pretrained:
                base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                base_model = models.resnet18(weights=None)
            self.encoder = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 512
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 添加投影头 (用于对比学习)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # 设置为评估模式
        self.encoder.eval()
        self.projection.eval()
        
        # 移动到GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = self.encoder.to(self.device)
        self.projection = self.projection.to(self.device)
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        # 数据增强（用于对比学习）
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, img, use_projection=True):
        """
        从图像中提取对比学习特征
        
        参数:
            img: PIL图像或张量
            use_projection: 是否使用投影头
            
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
            features = self.encoder(img)
            features = features.view(features.size(0), -1)
            
            if use_projection:
                features = self.projection(features)
                # L2归一化
                features = F.normalize(features, dim=1)
        
        # 转换为NumPy数组
        features = features.cpu().numpy()
        
        # 如果是批次，只返回第一个样本的特征
        if features.shape[0] == 1:
            features = features.squeeze(0)
        
        return features
    
    def generate_augmented_pairs(self, img, num_pairs=5):
        """
        生成增强的图像对，用于对比学习
        
        参数:
            img: PIL图像
            num_pairs: 生成的图像对数量
            
        返回:
            增强图像对的特征
        """
        if not isinstance(img, Image.Image):
            raise TypeError("输入必须是PIL图像")
        
        anchor_feature = self.extract_features(img)
        augmented_features = []
        
        for _ in range(num_pairs):
            # 应用随机增强
            augmented_img = self.augment(img)
            # 提取特征
            aug_feature = self.extract_features(augmented_img.unsqueeze(0))
            augmented_features.append(aug_feature.squeeze(0))
        
        return anchor_feature, augmented_features

def contrastive_hash(img, hash_size=8):
    """
    使用对比学习提取特征并生成哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小
        
    返回:
        二进制哈希值
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(contrastive_hash, 'extractor'):
        contrastive_hash.extractor = ContrastiveFeatureExtractor()
    
    # 提取特征
    features = contrastive_hash.extractor.extract_features(img)
    
    # 如果需要，可以使用PCA或其他方法降维到指定的hash_size
    # 这里简单地取前hash_size*hash_size个元素
    if hash_size * hash_size < len(features):
        features = features[:hash_size * hash_size]
    
    # 计算特征的中值
    median_value = np.median(features)
    
    # 生成二进制哈希
    hash_value = features > median_value
    
    return hash_value

def contrastive_deep(img, feature_dim=None):
    """
    使用对比学习提取深度特征用于相似度计算
    
    参数:
        img: 输入图像
        feature_dim: 特征维度，默认为None（使用完整特征）
        
    返回:
        特征向量
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(contrastive_deep, 'extractor'):
        contrastive_deep.extractor = ContrastiveFeatureExtractor()
    
    # 提取特征
    features = contrastive_deep.extractor.extract_features(img)
    
    # 如果指定了特征维度，截取前feature_dim个元素
    if feature_dim is not None and feature_dim < len(features):
        features = features[:feature_dim]
    
    return features

def compute_contrastive_distance(feature1, feature2):
    """
    计算两个对比学习特征向量之间的距离
    
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