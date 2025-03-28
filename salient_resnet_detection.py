import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2

# -------------------------- 传统显著性算法模块 --------------------------
class TraditionalSaliency:
    """封装两种经典的非深度显著性检测算法"""
    
    @staticmethod
    def SR_saliency(image, gaussian_kernel=3):
        """
        Spectral Residual 算法实现
        参数:
            image: PIL.Image 对象
            gaussian_kernel: 高斯滤波核大小
        返回:
            saliency_map: [H, W] numpy数组 范围[0,1]
        """
        gray = np.array(image.convert('L'))
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        phase = np.angle(fshift)
        log_amplitude = np.log(magnitude + 1e-5)
        avg_log_amp = cv2.blur(log_amplitude, (gaussian_kernel, gaussian_kernel))
        residual = log_amplitude - avg_log_amp
        inverse = np.exp(residual + 1j * phase)
        ishift = np.fft.ifftshift(inverse)
        img_back = np.fft.ifft2(ishift)
        saliency_map = np.abs(img_back)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        return saliency_map

    @staticmethod
    def FT_saliency(image, blur_kernel=5):
        """
        Frequency-Tuned 算法实现
        参数:
            image: PIL.Image 对象
            blur_kernel: 高斯模糊核大小
        返回:
            saliency_map: [H, W] numpy数组 范围[0,1]
        """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(img_cv, (blur_kernel, blur_kernel), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        mean_l = np.mean(l)
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        saliency = (l - mean_l)**2 + (a - mean_a)**2 + (b - mean_b)**2
        saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
        return saliency

# -------------------------- ResNet特征提取器（增强版） --------------------------
class ResNetFeatureExtractor:
    def __init__(self, layer='avgpool', use_saliency=True, saliency_algorithm='SR'):
        """
        初始化ResNet50特征提取器，并可选择使用非深度显著性方法对特征进行加权
        
        参数:
            layer (str): 提取特征的层，可选 'avgpool', 'fc' 或 'layer4'
            use_saliency (bool): 是否使用非深度显著性算法增强特征提取
            saliency_algorithm (str): 显著性算法选择，'SR' 或 'FT'
        """
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.layer = layer
        self.features = None

        if layer == 'avgpool':
            self.model.avgpool.register_forward_hook(self._get_features_hook)
        elif layer == 'fc':
            self.model.fc.register_forward_hook(self._get_features_hook)
        elif layer == 'layer4':
            self.model.layer4.register_forward_hook(self._get_features_hook)
        else:
            raise ValueError(f"不支持的层: {layer}")
        
        self.use_saliency = use_saliency
        self.saliency_algorithm = saliency_algorithm
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def _get_features_hook(self, module, input, output):
        self.features = output

    def extract_features(self, img):
        """
        从图像中提取特征，并可利用非深度显著方法加权
        
        参数:
            img: PIL图像或张量；若启用显著性加权，建议传入PIL.Image
        返回:
            特征向量（numpy数组）
        """
        original_img = img if isinstance(img, Image.Image) else None
        if isinstance(img, Image.Image):
            img = self.preprocess(img)
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device)
        
        with torch.no_grad():
            _ = self.model(img)
       
        if self.use_saliency and original_img is not None:
            if self.saliency_algorithm == 'SR':
                sal_map_np = TraditionalSaliency.SR_saliency(original_img)
            elif self.saliency_algorithm == 'FT':
                sal_map_np = TraditionalSaliency.FT_saliency(original_img)
            else:
                raise ValueError(f"不支持的显著性算法: {self.saliency_algorithm}")
            sal_map = torch.tensor(sal_map_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            if self.features.dim() == 4:
                _, _, H_feat, W_feat = self.features.shape
                sal_map_resized = nn.functional.interpolate(sal_map, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
                features = self.features * sal_map_resized
                
            else:
                weight = sal_map.mean()
                features = self.features * weight
                
        else:
            features = self.features
        
        features = features.cpu().numpy()
        if features.shape[0] == 1:
            features = features.squeeze(0)
        return features

# -------------------------- 方法1: salient_resnet_hash --------------------------
def salient_resnet_hash(img, hash_size=8, use_saliency=True, saliency_algorithm='SR'):
    """
    使用ResNet50提取特征并生成哈希
    参数:
        img: 输入图像（PIL.Image 或 Tensor）
        hash_size: 哈希大小（最终哈希向量的维度为 hash_size^2）
        use_saliency: 是否启用非深度显著性加权
        saliency_algorithm: 显著性算法，'SR' 或 'FT'
    返回:
        二值化哈希（布尔数组）
    """
    to_pil = transforms.ToPILImage()
    img = to_pil(img)
    if not hasattr(salient_resnet_hash, 'extractor'):
        # 对于哈希生成，采用具有空间结构的层，如 'layer4'
        salient_resnet_hash.extractor = ResNetFeatureExtractor(layer='layer4', use_saliency=use_saliency, saliency_algorithm=saliency_algorithm)
    
    features = salient_resnet_hash.extractor.extract_features(img)
    features = features.flatten()
    if hash_size * hash_size < len(features):
        features = features[:hash_size * hash_size]
    median_value = np.median(features)
    hash_value = features > median_value
    return hash_value

# -------------------------- 方法2: salient_resnet_deep --------------------------
def salient_resnet_deep(img, feature_dim=2048, use_saliency=True, saliency_algorithm='SR'):
    """
    使用ResNet50提取深度特征用于相似度计算
    参数:
        img: 输入图像（PIL.Image 或 Tensor）
        feature_dim: 特征维度（默认2048，对于avgpool层而言）
        use_saliency: 是否启用非深度显著性加权
        saliency_algorithm: 显著性算法，'SR' 或 'FT'
    返回:
        归一化的特征向量（numpy数组）
    """
    to_pil = transforms.ToPILImage()
    img = to_pil(img)
    if not hasattr(salient_resnet_deep, 'extractor'):
        salient_resnet_deep.extractor = ResNetFeatureExtractor(layer='avgpool', use_saliency=use_saliency, saliency_algorithm=saliency_algorithm)
    
    features = salient_resnet_deep.extractor.extract_features(img)
    features = features.flatten()
    features = features / np.linalg.norm(features)
    return features

# -------------------------- 方法3: compute_salient_resnet_deep_distance --------------------------
def compute_salient_resnet_deep_distance(feature1, feature2):
    """
    计算两个salient_resnet_deep特征向量之间的距离
    参数:
        feature1, feature2: 特征向量
    返回:
        距离值（余弦距离：1 - 余弦相似度）
    """
    similarity = np.dot(feature1, feature2)
    distance = 1.0 - similarity
    return distance

# -------------------------- 示例用法 --------------------------
if __name__ == "__main__":
    img = Image.open("test_image.jpg")
    
    # 示例1: 使用salient_resnet_hash生成哈希（不启用显著性加权）
    hash_normal = salient_resnet_hash(img, hash_size=8, use_saliency=False)
    print("salient_resnet_hash（无显著性加权）前16位:", hash_normal[:16], "...")
    
    # 示例2: 使用salient_resnet_hash生成哈希（启用非深度显著性加权，SR算法）
    hash_saliency = salient_resnet_hash(img, hash_size=8, use_saliency=True, saliency_algorithm='SR')
    print("salient_resnet_hash（显著性加权-SR）前16位:", hash_saliency[:16], "...")
    
    # 示例3: 使用salient_resnet_deep提取深度特征（启用显著性加权，可用于相似度计算）
    deep_feature = salient_resnet_deep(img, use_saliency=True, saliency_algorithm='SR')
    print("salient_resnet_deep特征维度:", deep_feature.shape)
    
    # 示例4: 计算两个深度特征向量之间的距离（自比较示例）
    distance = compute_salient_resnet_deep_distance(deep_feature, deep_feature)
    print("自比较距离:", distance)
