import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

class ContrastiveFeatureExtractor:
    def __init__(self, model_name="resnet50"):
        """
        初始化对比学习特征提取器
        
        参数:
            model_name (str): 基础模型名称，支持'resnet50'和'vit'
        """
        self.model_name = model_name
        
        # 加载预训练模型
        if model_name == "resnet50":
            # 使用SimCLR或MoCo预训练的ResNet50更好，这里使用ImageNet预训练作为替代
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.model = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == "vit":
            base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            self.model = base_model
            self.feature_dim = base_model.hidden_dim
            # 移除分类头
            self.model.heads = nn.Identity()
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 设置为评估模式
        self.model.eval()
        
        # 移动到GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        # 对比学习增强
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, img, augment=False):
        """
        从图像中提取特征
        
        参数:
            img: PIL图像或张量
            augment: 是否应用数据增强
            
        返回:
            特征向量
        """
        # 如果输入是PIL图像，进行预处理
        if isinstance(img, Image.Image):
            if augment:
                img = self.augment(img)
            else:
                img = self.preprocess(img)
        
        # 如果输入是张量但不是批次形式，添加批次维度
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            img = img.unsqueeze(0)
        
        # 移动到相应设备
        img = img.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            features = self.model(img)
            
            # 处理不同模型的输出格式
            if isinstance(features, torch.Tensor):
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
        
        # 转换为NumPy数组
        features = features.cpu().numpy()
        
        # 如果是批次，只返回第一个样本的特征
        if features.shape[0] == 1:
            features = features.squeeze(0)
        
        # L2归一化
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def extract_contrastive_features(self, img, n_views=4):
        """
        提取对比学习特征
        
        参数:
            img: 输入图像
            n_views: 增强视图数量
            
        返回:
            对比学习特征
        """
        if not isinstance(img, Image.Image):
            # 如果是张量，转换为PIL图像
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    img = img[0]
                img = transforms.ToPILImage()(img.cpu())
        
        # 提取原始特征
        orig_features = self.extract_features(img, augment=False)
        
        # 提取多个增强视图的特征
        aug_features_list = []
        for _ in range(n_views):
            aug_features = self.extract_features(img, augment=True)
            aug_features_list.append(aug_features)
        
        # 计算增强视图特征的平均值
        aug_features_mean = np.mean(aug_features_list, axis=0)
        aug_features_mean = aug_features_mean / (np.linalg.norm(aug_features_mean) + 1e-8)
        
        # 融合原始特征和增强特征
        combined_features = 0.7 * orig_features + 0.3 * aug_features_mean
        combined_features = combined_features / (np.linalg.norm(combined_features) + 1e-8)
        
        return combined_features

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
        # 使用ViT作为基础模型，性能更好
        contrastive_hash.extractor = ContrastiveFeatureExtractor(model_name="vit")
    
    # 提取对比学习特征
    features = contrastive_hash.extractor.extract_contrastive_features(img, n_views=2)
    
    # 如果需要，可以使用PCA或其他方法降维到指定的hash_size
    # 这里简单地取前hash_size*hash_size个元素
    if hash_size * hash_size < len(features):
        features = features[:hash_size * hash_size]
    elif hash_size * hash_size > len(features):
        # 如果特征维度小于所需哈希位数，通过重复填充
        repeats = int(np.ceil((hash_size * hash_size) / len(features)))
        features = np.tile(features, repeats)[:hash_size * hash_size]
    
    # 使用迭代量化方法生成更好的二进制码
    # 简单实现：计算特征的中值作为阈值
    median_value = np.median(features)
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
        # 使用ViT作为基础模型，性能更好
        contrastive_deep.extractor = ContrastiveFeatureExtractor(model_name="vit")
    
    # 提取对比学习特征
    features = contrastive_deep.extractor.extract_contrastive_features(img, n_views=2)
    
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
    # 确保特征已归一化
    feature1 = feature1 / (np.linalg.norm(feature1) + 1e-8)
    feature2 = feature2 / (np.linalg.norm(feature2) + 1e-8)
    
    # 使用余弦距离（1 - 余弦相似度）
    similarity = np.dot(feature1, feature2)
    # 转换为距离（值越小表示越相似）
    distance = 1.0 - similarity
    return distance