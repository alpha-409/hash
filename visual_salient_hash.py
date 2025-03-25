"""
VisionSalientTensorHash.py
结合视觉显著性与张量分解的图像哈希算法
保持与原始框架兼容的接口格式
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import tensorly as tl
from tensorly.decomposition import tucker
from skimage.filters import sobel

tl.set_backend('pytorch')

class VisualSalientHashSystem:
    def __init__(self, hash_size=64, device=None):
        """
        初始化视觉显著性哈希系统
        参数:
            hash_size: 哈希码长度 (默认64)
            device: 计算设备 (自动检测)
        """
        # 设备配置
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 基础特征提取器 - 使用预训练的ResNet50但只保留中间层特征
        # 使用更浅的特征可能更适合哈希任务
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 只保留到layer3，layer4特征可能过于抽象
        self.base_model = nn.Sequential(
            self.base_model.conv1,
            self.base_model.bn1,
            self.base_model.relu,
            self.base_model.maxpool,
            self.base_model.layer1,
            self.base_model.layer2,
            self.base_model.layer3
        )
        self.base_model = self.base_model.to(self.device).eval()
        
        # 改进的视觉显著性模块 - 使用多尺度特征
        self.saliency_layer = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # 张量分解参数 - 更平衡的秩分配
        self.hash_size = hash_size
        # 调整秩以更好地捕获空间和通道信息
        rank_factor = min(14, hash_size // 4)  # 限制最大秩
        self.rank = [rank_factor, rank_factor, rank_factor * 2]  # 通道维度给予更高的秩
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def _build_salient_tensor(self, features):
        """构建显著性特征张量"""
        # 生成显著性图 [B, 1, H, W]
        saliency_map = self.saliency_layer(features)
        
        # 特征加权 [B, C, H, W]
        weighted_features = features * saliency_map
        
        # 处理批次大小为1的情况
        if weighted_features.size(0) == 1:
            # 移除批次维度 [1, C, H, W] -> [C, H, W]
            weighted_features = weighted_features.squeeze(0)
            
            # 构造3D张量 [H, W, C]
            tensor = weighted_features.permute(1, 2, 0)
        else:
            # 批量处理情况 - 取第一个样本
            weighted_features = weighted_features[0]
            
            # 构造3D张量 [H, W, C]
            tensor = weighted_features.permute(1, 2, 0)
        
        # 应用特征归一化以提高稳定性
        tensor = tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-8)
        
        return tensor

    def _tensor_decomposition(self, tensor):
        """执行张量分解生成哈希"""
        # Tucker分解
        core, factors = tucker(tensor, rank=self.rank)
        
        # 核心张量处理 - 使用更稳健的阈值策略
        core_vector = core.flatten()
        
        # 使用排序而不是中值，可以更好地处理偏斜分布
        sorted_values, _ = torch.sort(core_vector)
        threshold_idx = int(len(sorted_values) * 0.5)  # 50%分位点
        threshold = sorted_values[threshold_idx]
        
        # 生成二进制哈希
        hash_bits = (core_vector > threshold).cpu().numpy()
        
        # 确保哈希长度正确
        if len(hash_bits) > self.hash_size:
            # 如果太长，只保留最显著的位
            importance = np.abs(core_vector.cpu().numpy() - threshold.cpu().numpy())
            top_indices = np.argsort(importance)[-self.hash_size:]
            mask = np.zeros_like(hash_bits)
            mask[top_indices] = 1
            hash_bits = hash_bits * mask
            hash_bits = hash_bits[hash_bits != 0]
        
        # 如果太短，填充到指定长度
        if len(hash_bits) < self.hash_size:
            padding = np.zeros(self.hash_size - len(hash_bits), dtype=np.uint8)
            hash_bits = np.concatenate([hash_bits, padding])
            
        return hash_bits.astype(np.uint8)

    def _extract_deep_features(self, tensor):
        """提取深度特征向量"""
        # 确保张量是3D的 [H, W, C]
        if tensor.dim() == 4:  # [B, H, W, C]
            tensor = tensor[0]  # 取第一个样本
            
        # 使用SVD提取主要特征方向
        tensor_flat = tensor.reshape(-1, tensor.shape[-1])
        U, S, V = torch.svd(tensor_flat)
        
        # 使用奇异值加权的特征向量作为特征表示
        weighted_features = V[:, :min(512, V.shape[1])] * S[:min(512, S.shape[0])].unsqueeze(0)
        features = weighted_features.flatten()
        
        # L2归一化
        features = features / (torch.norm(features, p=2) + 1e-8)
        
        return features.cpu().numpy()

    def process_image(self, img):
        """统一图像处理流程"""
        if isinstance(img, Image.Image):
            img = self.preprocess(img).unsqueeze(0)  # 确保添加批次维度
        elif torch.is_tensor(img):
            # 如果已经是张量，确保有批次维度
            if img.dim() == 3:  # [C, H, W] -> [1, C, H, W]
                img = img.unsqueeze(0)
        return img.to(self.device)

# ================= 兼容接口 =================
def visual_salient_hash(img, hash_size=64):
    """哈希生成接口 (保持与resnet_hash相同参数格式)"""
    if not hasattr(visual_salient_hash, 'system'):
        visual_salient_hash.system = VisualSalientHashSystem(hash_size)
    
    img_tensor = visual_salient_hash.system.process_image(img)
    
    # 确保输入是4D张量
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        
    with torch.no_grad():
        features = visual_salient_hash.system.base_model(img_tensor)
        tensor = visual_salient_hash.system._build_salient_tensor(features)
        return visual_salient_hash.system._tensor_decomposition(tensor)

def visual_salient_deep(img, feature_dim=512):
    """深度特征接口 (保持与resnet_deep相同参数格式)"""
    if not hasattr(visual_salient_deep, 'system'):
        visual_salient_deep.system = VisualSalientHashSystem()
    
    img_tensor = visual_salient_deep.system.process_image(img)
    
    # 确保输入是4D张量
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        
    with torch.no_grad():
        features = visual_salient_deep.system.base_model(img_tensor)
        tensor = visual_salient_deep.system._build_salient_tensor(features)
        return visual_salient_deep.system._extract_deep_features(tensor)[:feature_dim]

def compute_visual_salient_deep_distance(feat1, feat2):
    """距离计算接口 (保持与compute_resnet_deep_distance相同格式)"""
    return 1.0 - np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

# ================= 使用示例 =================
if __name__ == "__main__":
    img = Image.open('test.jpg')
    
    # 哈希生成
    hash_code = visual_salient_hash(img)
    print(f"Salient Hash ({len(hash_code)} bits):\n{hash_code[:16]}...")
    
    # 深度特征
    deep_feature = visual_salient_deep(img)
    print(f"Deep Feature Norm: {np.linalg.norm(deep_feature):.4f}")
    
    # 距离计算
    img2 = Image.open('test2.jpg')
    feat2 = visual_salient_deep(img2)
    distance = compute_visual_salient_deep_distance(deep_feature, feat2)
    print(f"Feature Distance: {distance:.4f}")