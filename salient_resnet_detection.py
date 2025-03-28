import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# -------------------------- 视觉显著模型模块 --------------------------
class SaliencyModel(nn.Module):
    """轻量级显著性预测模块（可替换为预训练模型如DeepGaze）"""
    def __init__(self):
        super().__init__()
        self.saliency_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输入RGB图像
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),             # 输出单通道显著性图
            nn.Sigmoid()                                 # 归一化到[0,1]
        )
    
    def forward(self, x):
        return self.saliency_net(x)

# -------------------------- 改进的ResNet特征提取器 --------------------------
class SalientResNetFeatureExtractor:
    def __init__(self, layer='avgpool', use_attention=True):
        """
        参数:
            layer (str): 提取特征的层（'avgpool'或'fc'）
            use_attention (bool): 是否使用显著性注意力加权
        """
        # 加载ResNet50主干网络
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        
        # 显著性模型
        self.saliency_model = SaliencyModel()
        self.use_attention = use_attention
        
        # 注册钩子函数
        self.layer = layer
        self.features = None
        if layer == 'avgpool':
            self.model.avgpool.register_forward_hook(self._get_features_hook)
        elif layer == 'fc':
            self.model.fc.register_forward_hook(self._get_features_hook)
        else:
            raise ValueError(f"Unsupported layer: {layer}")
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.saliency_model = self.saliency_model.to(self.device)
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _get_features_hook(self, module, input, output):
        """钩子函数捕获特征"""
        self.features = output
    
    def _apply_saliency_attention(self, img_tensor):
        """应用显著性注意力加权"""
        # 生成显著性图 [B,1,H,W]
        saliency_map = self.saliency_model(img_tensor)
        # 调整尺寸匹配当前特征层（假设在avgpool层应用）
        if self.layer == 'avgpool':
            saliency_map = nn.functional.interpolate(saliency_map, size=(7,7))
        # 特征加权
        return self.features * saliency_map
    
    def extract_features(self, img):
        """提取带显著性加权的特征"""
        # 预处理
        if isinstance(img, Image.Image):
            img = self.preprocess(img)
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(img)
            if self.use_attention:
                # 应用显著性注意力
                weighted_features = self._apply_saliency_attention(img)
                self.features = weighted_features  # 覆盖原始特征
        
        # 获取特征
        features = self.features.cpu().numpy()
        if features.shape[0] == 1:
            features = features.squeeze(0)
        return features.flatten()

# -------------------------- 改进后的哈希生成方法 --------------------------
def salient_resnet_hash(img, hash_size=8):
    """显著性增强的哈希生成"""
    if not hasattr(salient_resnet_hash, 'extractor'):
        salient_resnet_hash.extractor = SalientResNetFeatureExtractor(layer='avgpool')
    
    features = salient_resnet_hash.extractor.extract_features(img)
    features = features[:hash_size*hash_size]  # 截断到指定维度
    
    # 基于显著区域生成哈希
    median = np.median(features)
    return (features > median).astype(int)

# -------------------------- 改进后的深度特征方法 --------------------------
def salient_resnet_deep(img):
    """显著性增强的深度特征"""
    if not hasattr(salient_resnet_deep, 'extractor'):
        salient_resnet_deep.extractor = SalientResNetFeatureExtractor(layer='avgpool')
    
    features = salient_resnet_deep.extractor.extract_features(img)
    return features / np.linalg.norm(features)  # L2归一化
def compute_salient_resnet_deep_distance(feature1, feature2):
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
# -------------------------- 示例用法 --------------------------
if __name__ == "__main__":
    # 加载测试图像
    img = Image.open("test_image.jpg")
    
    # 提取显著性加权特征
    salient_feature = salient_resnet_deep(img)
    print("显著性特征维度:", salient_feature.shape)
    
    # 生成哈希
    hash_code = salient_resnet_hash(img)
    print("显著性哈希:", hash_code[:16], "...")