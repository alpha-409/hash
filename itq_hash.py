import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

class ITQHashGenerator:
    def __init__(self, n_components=128, n_bits=64, n_iterations=50, base_model="resnet50"):
        """
        初始化ITQ哈希生成器
        
        参数:
            n_components (int): PCA降维后的维度
            n_bits (int): 最终哈希码的位数
            n_iterations (int): ITQ迭代次数
            base_model (str): 基础特征提取模型
        """
        self.n_components = min(n_components, n_bits)  # 确保n_components不超过n_bits
        self.n_bits = n_bits
        self.n_iterations = n_iterations
        
        # 初始化PCA
        self.pca = None
        self.rotation = None  # ITQ旋转矩阵
        self.is_fitted = False
        
        # 初始化特征提取器
        if base_model == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 2048
        elif base_model == "vgg16":
            base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.feature_extractor = nn.Sequential(*list(base_model.features), nn.AdaptiveAvgPool2d((1, 1)))
            self.feature_dim = 512
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
            else:
                features = features.logits
        
        # 转换为NumPy数组
        features = features.cpu().numpy()
        
        # 如果是批次，只返回第一个样本的特征
        if features.shape[0] == 1:
            features = features.squeeze(0)
        
        return features
    
    def fit(self, features):
        """
        拟合PCA和ITQ旋转矩阵
        
        参数:
            features: 特征矩阵，形状为 (n_samples, feature_dim)
        """
        # 确保n_components不超过样本数量和特征维度
        n_samples, n_features = features.shape
        self.n_components = min(self.n_components, n_samples, n_features)
        
        # 拟合PCA
        self.pca = PCA(n_components=self.n_components)
        V = self.pca.fit_transform(features)
        
        # 初始化随机旋转矩阵
        self.rotation = np.random.randn(self.n_components, self.n_bits)
        self.rotation, _ = np.linalg.qr(self.rotation)  # 正交化
        
        # ITQ迭代优化
        for _ in range(self.n_iterations):
            # 计算量化误差
            Z = np.dot(V, self.rotation)
            B = np.sign(Z)
            
            # 更新旋转矩阵
            C = np.dot(V.T, B)
            UB, sigma, UA = np.linalg.svd(C)
            self.rotation = np.dot(UB, UA)
        
        self.is_fitted = True
    
    def generate_hash(self, features):
        """
        为特征生成哈希码
        
        参数:
            features: 特征向量或矩阵
            
        返回:
            二进制哈希码
        """
        if not self.is_fitted:
            raise ValueError("ITQ模型尚未拟合，请先调用fit方法")
        
        # 确保输入是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # PCA降维
        V = self.pca.transform(features)
        
        # 应用旋转矩阵
        Z = np.dot(V, self.rotation)
        
        # 二值化 - 使用0作为阈值而不是中值
        B = np.sign(Z)
        B[B <= 0] = 0  # 将-1转换为0
        B = B.astype(bool)
        
        # 如果只有一个样本，返回一维数组
        if B.shape[0] == 1:
            B = B.squeeze(0)
        
        return B

def itq_hash(img, hash_size=8, n_bits=None):
    """
    使用ITQ方法生成图像哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小（如果n_bits为None，则使用hash_size^2作为哈希位数）
        n_bits: 哈希位数，如果指定则优先使用
        
    返回:
        二进制哈希值
    """
    # 如果n_bits未指定，则使用hash_size的平方
    if n_bits is None:
        n_bits = hash_size * hash_size
    
    # 创建或获取哈希生成器
    if not hasattr(itq_hash, 'generator'):
        # 使用ViT作为基础模型
        n_components = min(128, n_bits)
        itq_hash.generator = ITQHashGenerator(n_components=n_components, n_bits=n_bits, 
                                             n_iterations=100, base_model="vit")
        itq_hash.is_fitted = False
    
    # 提取特征
    features = itq_hash.generator.extract_features(img)
    
    # 如果尚未拟合，收集样本
    if not itq_hash.is_fitted:
        if not hasattr(itq_hash, 'feature_samples'):
            itq_hash.feature_samples = [features]
        elif len(itq_hash.feature_samples) < 200:  # 增加样本数量到200
            itq_hash.feature_samples.append(features)
        else:
            # 有足够样本时拟合ITQ
            sample_matrix = np.vstack(itq_hash.feature_samples)
            itq_hash.generator.fit(sample_matrix)
            itq_hash.is_fitted = True
    
    # 如果已拟合，生成哈希码
    if itq_hash.is_fitted:
        hash_value = itq_hash.generator.generate_hash(features)
    else:
        # 未拟合时使用简单的二值化作为临时方案
        if hash_size * hash_size < len(features):
            features = features[:hash_size * hash_size]
        hash_value = features > np.median(features)
    
    return hash_value

def itq_deep(img, feature_dim=None):
    """
    使用ITQ方法提取深度特征
    
    参数:
        img: 输入图像
        feature_dim: 特征维度
        
    返回:
        特征向量
    """
    # 创建或获取哈希生成器
    if not hasattr(itq_deep, 'generator'):
        itq_deep.generator = ITQHashGenerator()
    
    # 提取特征
    features = itq_deep.generator.extract_features(img)
    
    # 如果指定了特征维度，截取前feature_dim个元素
    if feature_dim is not None and feature_dim < len(features):
        features = features[:feature_dim]
    
    # 归一化特征
    features = features / np.linalg.norm(features)
    
    return features

def compute_itq_distance(feature1, feature2):
    """
    计算两个特征向量之间的距离
    
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