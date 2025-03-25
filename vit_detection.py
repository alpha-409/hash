import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image

class ViTFeatureExtractor:
    def __init__(self, model_name="vit_b_16"):
        """
        初始化ViT特征提取器，使用torchvision模型
        
        参数:
            model_name (str): 模型名称，可选: vit_b_16, vit_b_32, vit_l_16, vit_l_32
        """
        # 加载预训练的ViT模型
        if model_name == "vit_b_16":
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif model_name == "vit_b_32":
            self.model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        elif model_name == "vit_l_16":
            self.model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        elif model_name == "vit_l_32":
            self.model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")
        
        # 移除分类头，只保留特征提取部分
        self.feature_dim = self.model.hidden_dim
        self.model.heads = nn.Identity()
        
        # 设置为评估模式
        self.model.eval()
        
        # 移动到GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, img):
        """
        从图像中提取ViT特征
        
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
            features = self.model(img)
        
        # 转换为NumPy数组
        features = features.cpu().numpy()
        
        # 如果是批次，只返回第一个样本的特征
        if features.shape[0] == 1:
            features = features.squeeze(0)
        
        # 归一化特征
        features = features / np.linalg.norm(features)
        
        return features

def vit_hash(img, hash_size=8):
    """
    使用ViT提取特征并生成哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小
        
    返回:
        二进制哈希值
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(vit_hash, 'extractor'):
        vit_hash.extractor = ViTFeatureExtractor()
    
    # 提取特征
    features = vit_hash.extractor.extract_features(img)
    
    # 如果需要，可以使用PCA或其他方法降维到指定的hash_size
    # 这里简单地取前hash_size*hash_size个元素
    if hash_size * hash_size < len(features):
        features = features[:hash_size * hash_size]
    
    # 计算特征的中值
    median_value = np.median(features)
    
    # 生成二进制哈希
    hash_value = features > median_value
    
    return hash_value

def vit_deep(img, feature_dim=None):
    """
    使用ViT提取深度特征用于相似度计算
    
    参数:
        img: 输入图像
        feature_dim: 特征维度，默认为None（使用完整特征）
        
    返回:
        特征向量
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(vit_deep, 'extractor'):
        vit_deep.extractor = ViTFeatureExtractor()
    
    # 提取特征
    features = vit_deep.extractor.extract_features(img)
    
    # 如果指定了特征维度，截取前feature_dim个元素
    if feature_dim is not None and feature_dim < len(features):
        features = features[:feature_dim]
    
    return features

def compute_vit_distance(feature1, feature2):
    """
    计算两个ViT特征向量之间的距离
    
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