import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import tensorly as tl
from tensorly.decomposition import tucker
from skimage.transform import resize
import cv2

# 设置tensorly后端为PyTorch
tl.set_backend('pytorch')

class SaliencyModel:
    """视觉显著性模型，用于检测图像中的显著区域"""
    
    def __init__(self, model_type='u2net'):
        """
        初始化显著性模型
        
        参数:
            model_type: 显著性模型类型，支持'u2net'和'basnet'
        """
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        # 加载预训练模型
        if model_type == 'u2net':
            # 使用OpenCV的Saliency模块作为替代
            self.model = cv2.saliency.StaticSaliencySpectralResidual_create()
        else:
            # 使用OpenCV的另一种显著性检测方法
            self.model = cv2.saliency.StaticSaliencyFineGrained_create()
    
    def detect(self, img):
        """
        检测图像中的显著区域
        
        参数:
            img: PIL图像
            
        返回:
            显著性图，值范围[0,1]，尺寸与输入图像相同
        """
        # 转换为OpenCV格式
        if isinstance(img, Image.Image):
            img_cv = np.array(img)
            if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img
            
        # 确保图像是BGR格式
        if len(img_cv.shape) == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
            
        # 检测显著性
        success, saliency_map = self.model.computeSaliency(img_cv)
        
        if not success:
            # 如果检测失败，返回均匀权重
            return np.ones((img_cv.shape[0], img_cv.shape[1]), dtype=np.float32)
        
        # 归一化到[0,1]
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        # 调整大小以匹配原始图像
        if img_cv.shape[:2] != saliency_map.shape:
            saliency_map = resize(saliency_map, img_cv.shape[:2], anti_aliasing=True)
            
        return saliency_map


class LightweightResNet(nn.Module):
    """轻量级ResNet模型，用于提取图像特征"""
    
    def __init__(self):
        super(LightweightResNet, self).__init__()
        # 加载预训练的ResNet18模型
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 只使用前几层
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )
        
        # 冻结参数
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, x):
        return self.features(x)
    
    def extract_features(self, img):
        """
        提取图像特征
        
        参数:
            img: PIL图像
            
        返回:
            特征图，形状为[C, H, W]
        """
        if isinstance(img, Image.Image):
            x = self.transform(img).unsqueeze(0).to(self.device)
        else:
            x = img.to(self.device)
            
        with torch.no_grad():
            features = self(x)
            
        # 返回特征图，移除批次维度
        return features.squeeze(0).cpu()


class SaliencyTensorHash:
    """基于视觉显著性和张量分解的图像哈希算法"""
    
    def __init__(self, hash_size=16, tensor_rank=(10, 10, 10)):
        """
        初始化哈希算法
        
        参数:
            hash_size: 哈希大小
            tensor_rank: Tucker分解的秩，形式为(r1, r2, r3)
        """
        self.hash_size = hash_size
        self.tensor_rank = tensor_rank
        
        # 初始化显著性模型
        self.saliency_model = SaliencyModel()
        
        # 初始化特征提取模型
        self.feature_extractor = LightweightResNet()
        
    def compute_hash(self, img):
        """
        计算图像哈希
        
        参数:
            img: PIL图像
            
        返回:
            二进制哈希码
        """
        # 1. 提取显著性图
        saliency_map = self.saliency_model.detect(img)
        
        # 2. 提取深度特征图
        feature_maps = self.feature_extractor.extract_features(img)
        
        # 3. 构建加权特征张量
        C, H, W = feature_maps.shape
        saliency_resized = torch.from_numpy(
            resize(saliency_map, (H, W), anti_aliasing=True)
        ).float()
        
        # 应用显著性权重到特征图
        weighted_features = feature_maps * saliency_resized.unsqueeze(0)
        
        # 4. 执行Tucker分解
        core, factors = tucker(weighted_features, rank=self.tensor_rank)
        
        # 5. 使用核心张量生成哈希
        # 将核心张量展平
        core_flat = core.flatten()
        
        # 如果核心张量元素数量小于哈希大小的平方，则重复填充
        if len(core_flat) < self.hash_size * self.hash_size:
            repeats = int(np.ceil((self.hash_size * self.hash_size) / len(core_flat)))
            core_flat = core_flat.repeat(repeats)
        
        # 取前hash_size*hash_size个元素
        core_flat = core_flat[:self.hash_size * self.hash_size]
        
        # 计算中值作为阈值
        threshold = torch.median(core_flat)
        
        # 生成二进制哈希
        hash_code = (core_flat > threshold).numpy().astype(np.uint8)
        
        return hash_code
    
    def compute_distance(self, hash1, hash2):
        """
        计算两个哈希码之间的距离
        
        参数:
            hash1, hash2: 二进制哈希码
            
        返回:
            汉明距离
        """
        # 计算汉明距离
        return np.sum(hash1 != hash2)


# 对外接口函数
def saliency_tensor_hash(img, hash_size=16):
    """
    使用显著性和张量分解计算图像哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小
        
    返回:
        二进制哈希值
    """
    # 创建哈希算法实例（如果尚未创建）
    if not hasattr(saliency_tensor_hash, 'hasher'):
        tensor_rank = (min(10, hash_size), min(10, hash_size), min(10, hash_size))
        saliency_tensor_hash.hasher = SaliencyTensorHash(hash_size=hash_size, tensor_rank=tensor_rank)
    
    # 计算哈希
    return saliency_tensor_hash.hasher.compute_hash(img)

def compute_saliency_tensor_distance(hash1, hash2):
    """
    计算两个显著性张量哈希之间的距离
    
    参数:
        hash1, hash2: 哈希值
        
    返回:
        距离值（越小表示越相似）
    """
    # 计算汉明距离
    return np.sum(hash1 != hash2) / len(hash1)