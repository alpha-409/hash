import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

class MultiscaleFeatureExtractor:
    def __init__(self, scales=[1.0, 0.75, 0.5]):
        """
        初始化多尺度ResNet50特征提取器
        
        参数:
            scales (list): 图像缩放比例列表
        """
        # 加载预训练的ResNet50模型
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        
        # 移动到GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # 保存缩放比例
        self.scales = scales
        
        # 设置提取特征的层
        self.features = None
        self.model.avgpool.register_forward_hook(self._get_features_hook)
        
        # 基本图像预处理
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def _get_features_hook(self, module, input, output):
        """
        钩子函数，用于获取中间层的输出
        """
        self.features = output
    
    def extract_multiscale_features(self, img):
        """
        从图像中提取多尺度特征
        
        参数:
            img: PIL图像或张量
            
        返回:
            多尺度特征向量的连接
        """
        all_features = []
        
        # 如果输入是张量，转换为PIL图像以便于缩放
        if isinstance(img, torch.Tensor):
            if img.dim() == 4:  # 批次形式
                img = img[0]  # 取第一个样本
            # 转换为PIL图像
            img = transforms.ToPILImage()(img.cpu())
        
        # 获取原始图像尺寸
        width, height = img.size
        
        # 对每个尺度提取特征
        for scale in self.scales:
            # 计算新尺寸
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 调整图像大小
            scaled_img = img.resize((new_width, new_height), Image.BILINEAR)
            
            # 中心裁剪到224x224
            crop_transform = transforms.Compose([
                transforms.CenterCrop(min(new_width, new_height)),
                transforms.Resize(224)
            ])
            scaled_img = crop_transform(scaled_img)
            
            # 预处理
            input_tensor = self.preprocess(scaled_img).unsqueeze(0).to(self.device)
            
            # 前向传播
            with torch.no_grad():
                _ = self.model(input_tensor)
            
            # 获取特征并展平
            scale_features = self.features.cpu().numpy().flatten()
            all_features.append(scale_features)
        
        # 连接所有尺度的特征
        combined_features = np.concatenate(all_features)
        
        # 归一化特征向量
        combined_features = combined_features / np.linalg.norm(combined_features)
        
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
        multiscale_hash.extractor = MultiscaleFeatureExtractor()
    
    # 提取多尺度特征
    features = multiscale_hash.extractor.extract_multiscale_features(img)
    
    # 如果需要，可以使用PCA或其他方法降维到指定的hash_size
    # 这里简单地取前hash_size*hash_size个元素
    if hash_size * hash_size < len(features):
        features = features[:hash_size * hash_size]
    
    # 计算特征的中值
    median_value = np.median(features)
    
    # 生成二进制哈希
    hash_value = features > median_value
    
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
        multiscale_deep.extractor = MultiscaleFeatureExtractor()
    
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
    # 使用余弦距离（1 - 余弦相似度）
    similarity = np.dot(feature1, feature2)
    # 转换为距离（值越小表示越相似）
    distance = 1.0 - similarity
    return distance