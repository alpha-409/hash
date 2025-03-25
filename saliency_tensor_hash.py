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

# 修改SaliencyModel类
class SaliencyModel:
    """视觉显著性模型，用于检测图像中的显著区域"""
    
    def __init__(self, model_type='simple'):
        """
        初始化显著性模型
        
        参数:
            model_type: 显著性模型类型，支持'simple'、'frequency'和'gradient'
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
    
    def detect(self, img):
        """
        检测图像中的显著区域
        
        参数:
            img: PIL图像或PyTorch张量
            
        返回:
            显著性图，值范围[0,1]，尺寸与输入图像相同
        """
        # 转换为NumPy数组或处理PyTorch张量
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        elif torch.is_tensor(img):
            # 如果是张量，先检查是否需要转换为NumPy
            if self.model_type in ['frequency', 'gradient', 'simple']:
                # 这些方法内部会处理张量转换
                img_np = img
            else:
                # 其他方法可能需要直接转换为NumPy
                img_np = img.cpu().numpy()
        else:
            img_np = img
            
        # 确保图像是RGB格式（如果是灰度图）
        if not torch.is_tensor(img_np) and len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=2)
        
        # 根据选择的模型类型计算显著性
        if self.model_type == 'frequency':
            # 基于频域分析的显著性检测
            saliency_map = self._frequency_saliency(img_np)
        elif self.model_type == 'gradient':
            # 基于梯度的显著性检测
            saliency_map = self._gradient_saliency(img_np)
        else:
            # 默认使用简单的显著性检测（颜色对比度）
            saliency_map = self._simple_saliency(img_np)
        
        # 调整大小以匹配原始图像
        if isinstance(img, Image.Image):
            target_size = img.size[::-1]  # PIL的size是(width, height)，需要转为(height, width)
        elif torch.is_tensor(img):
            if len(img.shape) == 3:  # [C, H, W]
                target_size = (img.shape[1], img.shape[2])
            elif len(img.shape) == 4:  # [B, C, H, W]
                target_size = (img.shape[2], img.shape[3])
            else:
                target_size = saliency_map.shape
        else:
            target_size = img_np.shape[:2]
            
        if target_size != saliency_map.shape:
            saliency_map = resize(saliency_map, target_size, anti_aliasing=True)
            
        return saliency_map
    
    def _simple_saliency(self, img):
        """简单的基于颜色对比度的显著性检测"""
        # 检查输入类型并转换为NumPy数组
        if torch.is_tensor(img):
            img_np = img.cpu().numpy()
        else:
            img_np = img
            
        # 转换为Lab颜色空间
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        
        # 转换为灰度图
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        # 使用高斯模糊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 计算局部对比度
        local_contrast = cv2.Laplacian(blur, cv2.CV_64F)
        
        # 取绝对值并归一化
        saliency = np.abs(local_contrast)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def _gradient_saliency(self, img):
        """基于梯度的显著性检测"""
        # 检查输入类型并转换为NumPy数组
        if torch.is_tensor(img):
            img_np = img.cpu().numpy()
        else:
            img_np = img
            
        # 转换为灰度图
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.astype(np.uint8)
        
        # 计算x和y方向的梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 使用高斯模糊平滑显著性图
        saliency = cv2.GaussianBlur(grad_mag, (9, 9), 0)
        
        # 归一化
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def _frequency_saliency(self, img):
        """基于频域分析的显著性检测（光谱残差法）"""
        # 检查输入类型并转换为NumPy数组
        if torch.is_tensor(img):
            img_np = img.cpu().numpy()
        else:
            img_np = img
            
        # 转换为灰度图
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.astype(np.uint8)
        
        # 对图像进行傅里叶变换
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # 计算幅度谱
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-8)
        
        # 应用高斯滤波器平滑幅度谱
        magnitude_spectrum_blur = cv2.GaussianBlur(magnitude_spectrum, (5, 5), 0)
        
        # 计算光谱残差
        spectral_residual = magnitude_spectrum - magnitude_spectrum_blur
        
        # 重建显著性图
        dft_shift[:, :, 0] = np.cos(spectral_residual) * dft_shift[:, :, 0]
        dft_shift[:, :, 1] = np.sin(spectral_residual) * dft_shift[:, :, 1]
        
        # 逆傅里叶变换
        back_shift = np.fft.ifftshift(dft_shift)
        back = cv2.idft(back_shift)
        saliency = cv2.magnitude(back[:, :, 0], back[:, :, 1])
        
        # 归一化
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # 应用高斯模糊增强显著性图
        saliency = cv2.GaussianBlur(saliency, (9, 9), 0)
        
        return saliency

class ResNet50FeatureExtractor(nn.Module):
    """ResNet50特征提取器，用于提取图像特征"""
    
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        # 加载预训练的ResNet50模型
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 使用卷积层和部分残差块
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
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
    
    def __init__(self, hash_size=16, tensor_rank=(10, 10, 10), saliency_type='frequency'):
        """
        初始化哈希算法
        
        参数:
            hash_size: 哈希大小
            tensor_rank: Tucker分解的秩，形式为(r1, r2, r3)
            saliency_type: 显著性检测方法，可选'simple'、'gradient'或'frequency'
        """
        self.hash_size = hash_size
        self.tensor_rank = tensor_rank
        
        # 初始化显著性模型
        self.saliency_model = SaliencyModel(model_type=saliency_type)
        
        # 初始化特征提取模型
        self.feature_extractor = ResNet50FeatureExtractor()
        
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
    
    def detect_copy(self, img1, img2, threshold=0.25):
        """
        检测两张图像是否为拷贝关系
        
        参数:
            img1, img2: 两张输入图像
            threshold: 判定为拷贝的阈值，距离小于此值认为是拷贝
            
        返回:
            is_copy: 布尔值，表示是否为拷贝
            distance: 两张图像的距离值
            similarity: 相似度值 (1 - 归一化距离)
        """
        # 计算两张图像的哈希值
        hash1 = self.compute_hash(img1)
        hash2 = self.compute_hash(img2)
        
        # 计算距离
        distance = self.compute_distance(hash1, hash2)
        
        # 归一化距离 (0-1范围)
        normalized_distance = distance / len(hash1)
        
        # 计算相似度 (1表示完全相同，0表示完全不同)
        similarity = 1.0 - normalized_distance
        
        # 判断是否为拷贝
        is_copy = normalized_distance < threshold
        
        return is_copy, normalized_distance, similarity


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


def detect_image_copy(img1, img2, hash_size=16, threshold=0.25):
    """
    检测两张图像是否为拷贝关系
    
    参数:
        img1, img2: 两张输入图像
        hash_size: 哈希大小
        threshold: 判定为拷贝的阈值，距离小于此值认为是拷贝
        
    返回:
        is_copy: 布尔值，表示是否为拷贝
        distance: 两张图像的距离值
        similarity: 相似度值 (1 - 归一化距离)
    """
    # 创建哈希算法实例（如果尚未创建）
    if not hasattr(detect_image_copy, 'hasher'):
        tensor_rank = (min(10, hash_size), min(10, hash_size), min(10, hash_size))
        detect_image_copy.hasher = SaliencyTensorHash(hash_size=hash_size, tensor_rank=tensor_rank)
    
    # 检测拷贝
    return detect_image_copy.hasher.detect_copy(img1, img2, threshold)

def batch_copy_detection(query_img, reference_imgs, hash_size=16, threshold=0.25):
    """
    批量检测一张查询图像与多张参考图像的拷贝关系
    
    参数:
        query_img: 查询图像
        reference_imgs: 参考图像列表
        hash_size: 哈希大小
        threshold: 判定为拷贝的阈值
        
    返回:
        results: 包含检测结果的列表，每个元素为(索引, 是否拷贝, 距离, 相似度)
    """
    # 创建哈希算法实例（如果尚未创建）
    if not hasattr(batch_copy_detection, 'hasher'):
        tensor_rank = (min(10, hash_size), min(10, hash_size), min(10, hash_size))
        batch_copy_detection.hasher = SaliencyTensorHash(hash_size=hash_size, tensor_rank=tensor_rank)
    
    # 计算查询图像的哈希
    query_hash = batch_copy_detection.hasher.compute_hash(query_img)
    
    results = []
    # 对每张参考图像进行检测
    for i, ref_img in enumerate(reference_imgs):
        # 计算参考图像的哈希
        ref_hash = batch_copy_detection.hasher.compute_hash(ref_img)
        
        # 计算距离
        distance = batch_copy_detection.hasher.compute_distance(query_hash, ref_hash)
        
        # 归一化距离
        normalized_distance = distance / len(query_hash)
        
        # 计算相似度
        similarity = 1.0 - normalized_distance
        
        # 判断是否为拷贝
        is_copy = normalized_distance < threshold
        
        # 添加结果
        results.append((i, is_copy, normalized_distance, similarity))
    
    # 按相似度降序排序结果
    results.sort(key=lambda x: x[3], reverse=True)
    
    return results