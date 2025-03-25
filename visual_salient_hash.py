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
        
        # 基础特征提取器
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])  # 保留layer4输出
        self.base_model = self.base_model.to(self.device).eval()
        
        # 视觉显著性模块
        self.saliency_layer = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # 张量分解参数
        self.hash_size = hash_size
        self.rank = [hash_size//2, hash_size//2, hash_size//2]
        
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
        
        # 可选：应用额外的归一化
        # tensor = tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-8)
        
        return tensor

    def _tensor_decomposition(self, tensor):
        """执行张量分解生成哈希"""
        # Tucker分解
        # 修正参数名称：从 ranks 改为 rank
        core, _ = tucker(tensor, rank=self.rank)
        
        # 核心张量处理
        core_vector = core.flatten()
        hash_bits = (core_vector > core_vector.median()).cpu().numpy()
        return hash_bits.astype(np.uint8)

    def _extract_deep_features(self, tensor):
        """提取深度特征向量"""
        # 确保张量是3D的 [H, W, C]
        if tensor.dim() == 4:  # [B, H, W, C]
            tensor = tensor[0]  # 取第一个样本
            
        # 张量标准化
        norm_tensor = tensor / (torch.norm(tensor, p=2) + 1e-8)
        return norm_tensor.flatten().cpu().numpy()

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