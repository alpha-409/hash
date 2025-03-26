import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
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
    
    # # 提取特征
    # features = resnet_hash.extractor.extract_features(img)
    
    # # 将特征展平
    # features = features.flatten()
    
    # # 如果需要，可以使用PCA或其他方法降维到指定的hash_size
    # # 这里简单地取前hash_size*hash_size个元素
    # if hash_size * hash_size < len(features):
    #     features = features[:hash_size * hash_size]
    
    # # 计算特征的中值
    # median_value = np.median(features)
    
    # # 生成二进制哈希
    # hash_value = features > median_value
    if not hasattr(resnet_hash, 'pca'):
        resnet_hash.pca = PCA(n_components=hash_size**2)
        # 需要用样本数据预先训练PCA（这里需要训练逻辑）
    
    # 提取原始特征
    features = resnet_hash.extractor.extract_features(img).flatten()
    
    # PCA降维
    reduced_features = resnet_hash.pca.transform(features.reshape(1,-1))[0]
    
    # 自适应二值化（使用各维度均值）
    hash_value = (reduced_features > np.mean(reduced_features)).astype(int)
    
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