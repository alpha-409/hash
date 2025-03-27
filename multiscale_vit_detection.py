import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from torch.nn import functional as F

class ViTFeatureExtractor:
    def __init__(self, model_name="vit_b_16", train=False):
        # 加载预训练的 Vision Transformer 模型
        if model_name == "vit_b_16":
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif model_name == "vit_b_32":
            self.model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        elif model_name == "vit_l_16":
            self.model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        elif model_name == "vit_l_32":
            self.model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # 保存模型输出的特征维度
        self.feature_dim = self.model.hidden_dim
        # 移除分类头，只保留特征提取部分
        self.model.heads = nn.Identity()
        
        # 根据 train 参数设置模型模式
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        # 检查 GPU 是否可用，并移动模型到对应设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # 图像预处理流程：缩放、中心裁剪、转为 Tensor、归一化
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, img):
        # 如果输入为 PIL.Image，则预处理
        if isinstance(img, Image.Image):
            img = self.preprocess(img)
        
        # 如果输入为 Tensor 且不在批次格式中，添加 batch 维度
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            img = img.unsqueeze(0)
        
        # 将图像移动到设备（GPU 或 CPU）
        img = img.to(self.device)
        
        # 前向传播获取特征，训练或评估均可
        with torch.no_grad():
            features = self.model(img)
        
        # 将特征转换为 numpy 数组
        features = features.cpu().numpy()
        
        # 如果只有一个样本，则去除 batch 维度
        if features.shape[0] == 1:
            features = features.squeeze(0)
        
        # 归一化特征向量
        features = features / np.linalg.norm(features)
        
        return features


def extract_multiscale_features_vit(img, scales):
    """
    使用多尺度 ViT 提取特征
    
    参数:
        img: 输入图像（PIL.Image 或 Tensor）。
        scales: 缩放因子列表 (例如：[1.0, 0.75, 0.5])。
    
    返回:
        numpy.ndarray: 特征矩阵，形状为 (n_scales, feature_dim)。
    """
    extractor = ViTFeatureExtractor()  # 默认使用评估模式
    features_list = []
    
    for scale in scales:
        if isinstance(img, torch.Tensor):
            # 保证输入为批次格式
            if img.dim() == 3:
                img_tensor = img.unsqueeze(0)
            else:
                img_tensor = img
            
            # 先按比例缩放图像，再调整到 224x224 大小
            _, C, H, W = img_tensor.shape
            new_H = max(1, int(H * scale))
            new_W = max(1, int(W * scale))
            img_scaled = F.interpolate(img_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
            img_scaled = F.interpolate(img_scaled, size=(224, 224), mode='bilinear', align_corners=False)
            img_scaled = img_scaled.squeeze(0)
            
            feature = extractor.extract_features(img_scaled)
        elif isinstance(img, Image.Image):
            # 按比例缩放
            w, h = img.size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img_scaled = img.resize(new_size, Image.LANCZOS)
            # 调整到 ViT 所需的 224x224 尺寸
            img_scaled = img_scaled.resize((224, 224), Image.LANCZOS)
            
            feature = extractor.extract_features(img_scaled)
        else:
            raise ValueError("Unsupported image type. Only PIL.Image or torch.Tensor are supported.")
        
        # 如果特征是多维，则扁平化
        if feature.ndim > 1:
            feature = feature.flatten()
        features_list.append(feature)
    
    return np.stack(features_list, axis=0)


def multiscale_vit_hash(img, hash_size=8, scales=[1.0, 0.75, 0.5]):
    """
    基于多尺度 ViT 特征和矩阵分解生成二值哈希
    
    参数:
        img: 输入图像（PIL.Image 或 Tensor）。
        hash_size: 哈希的边长（例如：8）。
        scales: 多尺度提取的缩放因子列表。
    
    返回:
        numpy.ndarray: 二值哈希向量。
    """
    features_matrix = extract_multiscale_features_vit(img, scales)
    
    # 进行融合和降维（例如，通过奇异值分解）
    U, S, Vt = np.linalg.svd(features_matrix, full_matrices=False)
    fused_feature = Vt[0]  # 取第一主成分
    target_dim = hash_size * hash_size
    
    if fused_feature.shape[0] > target_dim:
        fused_feature = fused_feature[:target_dim]
    
    median_val = np.median(fused_feature)
    binary_hash = fused_feature > median_val
    return binary_hash


def multiscale_vit_deep(img, scales=[1.0, 0.75, 0.5]):
    """
    基于多尺度 ViT 特征生成归一化深度特征向量
    
    参数:
        img: 输入图像（PIL.Image 或 Tensor）。
        scales: 多尺度提取的缩放因子列表。
    
    返回:
        numpy.ndarray: 归一化后的特征向量。
    """
    features_matrix = extract_multiscale_features_vit(img, scales)
    
    U, S, Vt = np.linalg.svd(features_matrix, full_matrices=False)
    fused_feature = Vt[0]  # 取第一主成分
    
    norm = np.linalg.norm(fused_feature)
    if norm > 0:
        fused_feature = fused_feature / norm
    
    return fused_feature


def compute_multiscale_vit_distance(feature1, feature2):
    """
    计算两个多尺度 ViT 特征向量之间的余弦距离
    
    参数:
        feature1, feature2: 特征向量 (numpy 数组)。
    
    返回:
        float: 余弦距离（值越低表示相似度越高）。
    """
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    distance = 1.0 - similarity
    return distance


# 示例：如何在 GPU 上进行训练（如果需要训练模式）
if __name__ == "__main__":
    # 检查设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 加载一张图片示例（请替换为实际图片路径）
    try:
        img = Image.open("example.jpg")
    except Exception as e:
        print("未能加载图片，请确保 'example.jpg' 文件存在。")
        img = Image.new("RGB", (256, 256), color='white')
    
    # 创建特征提取器，此处若需要训练模式则设 train=True
    extractor = ViTFeatureExtractor(train=False)
    
    # 提取单尺度特征示例
    features = extractor.extract_features(img)
    print("提取的特征向量维度:", features.shape)
    
    # 提取多尺度特征
    scales = [1.0, 0.75, 0.5]
    multiscale_features = extract_multiscale_features_vit(img, scales)
    print("多尺度特征矩阵形状:", multiscale_features.shape)
    
    # 生成二值哈希
    binary_hash = multiscale_vit_hash(img, hash_size=8, scales=scales)
    print("二值哈希:", binary_hash)
    
    # 生成深度特征向量
    deep_feature = multiscale_vit_deep(img, scales=scales)
    print("归一化深度特征向量长度:", deep_feature.shape[0])
    
    # 计算两个特征向量间的余弦距离（示例）
    distance = compute_multiscale_vit_distance(features, deep_feature)
    print("余弦距离:", distance)
