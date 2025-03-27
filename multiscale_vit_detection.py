import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# 定义ViT特征提取器，增加 minimal_preprocess 选项用于多尺度场景
class ViTFeatureExtractor:
    def __init__(self, model_name="vit_b_16", minimal_preprocess=False):
        """
        初始化ViT特征提取器，使用torchvision模型

        参数:
            model_name (str): 模型名称，可选: vit_b_16, vit_b_32, vit_l_16, vit_l_32
            minimal_preprocess (bool): 是否使用最小预处理（仅转换为Tensor并归一化），默认为False（使用固定尺寸）
        """
        # 加载预训练的ViT模型
        if model_name == "vit_b_16":
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif model_name == "vit_b_32":
            self.model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        elif model_name == "vit_l_16":
            self.model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        elif model_name == "vit_l_32":
            self.model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")

        # 移除分类头，只保留特征提取部分
        self.feature_dim = self.model.hidden_dim
        self.model.heads = nn.Identity()

        # 设置为评估模式
        self.model.eval()

        # 移动到GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # 图像预处理：
        # 默认情况下，使用固定尺寸（Resize+CenterCrop）；在多尺度场景下，选择 minimal_preprocess（仅ToTensor和Normalize）
        if minimal_preprocess:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def extract_features(self, img):
        """
        从图像中提取ViT特征

        参数:
            img: PIL图像或张量

        返回:
            特征向量（已归一化）
        """
        # 如果输入是PIL图像，进行预处理
        if isinstance(img, Image.Image):
            img = self.preprocess(img)
        # 如果输入是张量但没有 batch 维度，则添加 batch 维度
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            img = img.unsqueeze(0)
        # 移动到对应设备
        img = img.to(self.device)
        # 前向传播
        with torch.no_grad():
            features = self.model(img)
        # 转换为NumPy数组
        features = features.cpu().numpy()
        # 如果是批次，则返回第一个样本的特征
        if features.shape[0] == 1:
            features = features.squeeze(0)
        # 归一化特征向量
        features = features / np.linalg.norm(features)
        return features


# ---------------- 多尺度特征提取及融合模块 ----------------

# 全局的ViT多尺度特征提取器，避免重复初始化（使用 minimal_preprocess 版本）
_extractor = None
EXPECTED_SIZE = 224  # ViT模型要求的输入尺寸

def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = ViTFeatureExtractor(model_name="vit_b_16", minimal_preprocess=True)
    return _extractor

def adjust_tensor_size(img_tensor):
    """
    调整张量图像的尺寸到 EXPECTED_SIZE x EXPECTED_SIZE，
    如果尺寸过大则中心裁剪，尺寸过小则填充。
    参数:
        img_tensor: 张量，形状 (1, 3, H, W) 或 (3, H, W)
    返回:
        调整后的张量，形状 (1, 3, EXPECTED_SIZE, EXPECTED_SIZE)
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    _, C, H, W = img_tensor.shape
    if H > EXPECTED_SIZE or W > EXPECTED_SIZE:
        img_tensor = TF.center_crop(img_tensor, (EXPECTED_SIZE, EXPECTED_SIZE))
    elif H < EXPECTED_SIZE or W < EXPECTED_SIZE:
        pad_H = EXPECTED_SIZE - H
        pad_W = EXPECTED_SIZE - W
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        img_tensor = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return img_tensor

def adjust_pil_size(img, current_size):
    """
    调整PIL图像尺寸到 EXPECTED_SIZE x EXPECTED_SIZE，
    如果尺寸过大则中心裁剪，尺寸过小则粘贴到背景画布中
    参数:
        img: PIL.Image 图像
        current_size: (w, h) 当前尺寸
    返回:
        调整后的 PIL.Image 图像，尺寸为 (EXPECTED_SIZE, EXPECTED_SIZE)
    """
    w, h = current_size
    if w > EXPECTED_SIZE or h > EXPECTED_SIZE:
        img = TF.center_crop(img, (EXPECTED_SIZE, EXPECTED_SIZE))
    elif w < EXPECTED_SIZE or h < EXPECTED_SIZE:
        new_img = Image.new("RGB", (EXPECTED_SIZE, EXPECTED_SIZE))
        offset_x = (EXPECTED_SIZE - w) // 2
        offset_y = (EXPECTED_SIZE - h) // 2
        new_img.paste(img, (offset_x, offset_y))
        img = new_img
    return img

def extract_multiscale_features(img, scales):
    """
    对输入图像在不同尺度下进行特征提取，并返回特征矩阵

    参数:
        img: 输入图像，支持 PIL.Image 或 torch.Tensor（形状为 (3, H, W)）
        scales (list): 尺度因子列表，例如 [1.0, 0.75, 0.5]

    返回:
        numpy.ndarray: 特征矩阵，形状 (n_scales, feature_dim)
    """
    extractor = get_extractor()
    features_list = []
    # 如果输入为 tensor 且存在 batch 维度则移除
    if isinstance(img, torch.Tensor) and img.dim() == 4:
        img = img.squeeze(0)
    for scale in scales:
        # 针对 tensor 格式
        if isinstance(img, torch.Tensor):
            # 确保输入为 (3, H, W)
            if img.dim() == 3:
                img_tensor = img.unsqueeze(0)  # (1, 3, H, W)
            else:
                img_tensor = img
            _, C, H, W = img_tensor.shape
            new_H = max(1, int(H * scale))
            new_W = max(1, int(W * scale))
            # 双线性插值缩放图像
            img_scaled = F.interpolate(img_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
            # 调整尺寸到 EXPECTED_SIZE x EXPECTED_SIZE（中心裁剪或填充）
            img_scaled = adjust_tensor_size(img_scaled)
            # 移除 batch 维度
            img_scaled = img_scaled.squeeze(0)
            # 提取特征
            feature = extractor.extract_features(img_scaled)
        # 针对 PIL.Image 格式
        elif isinstance(img, Image.Image):
            w, h = img.size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img_scaled = img.resize(new_size, Image.ANTIALIAS)
            # 调整 PIL 图像到 EXPECTED_SIZE x EXPECTED_SIZE
            img_scaled = adjust_pil_size(img_scaled, new_size)
            feature = extractor.extract_features(img_scaled)
        else:
            raise ValueError("输入图像类型不受支持，仅支持 PIL.Image 或 torch.Tensor")
        # 如果特征非一维则展平
        if feature.ndim > 1:
            feature = feature.flatten()
        features_list.append(feature)
    return np.stack(features_list, axis=0)

def fuse_features(features_matrix, hash_length):
    """
    融合多尺度特征矩阵，通过 SVD 分解得到融合表示，
    并根据目标维度（hash_length^2）截断/降维

    参数:
        features_matrix (numpy.ndarray): 多尺度特征矩阵，形状 (n_scales, d)
        hash_length (int): 指定哈希的边长，最终哈希维度为 hash_length * hash_length

    返回:
        numpy.ndarray: 融合后的特征向量
    """
    U, S, Vt = np.linalg.svd(features_matrix, full_matrices=False)
    # 取第一主成分作为融合特征
    fused_feature = Vt[0]
    target_dim = hash_length * hash_length
    if fused_feature.shape[0] > target_dim:
        fused_feature = fused_feature[:target_dim]
    return fused_feature

def multiscale_vit_hash(img, hash_size=8, scales=[1.0, 0.75, 0.5]):
    """
    基于多尺度 ViT 特征和矩阵分解生成二值哈希

    参数:
        img: 输入图像，支持 PIL.Image 或 torch.Tensor（形状为 (3, H, W)）
        hash_size (int): 哈希边长，生成的哈希维度为 hash_size^2
        scales (list): 多尺度因子列表

    返回:
        numpy.ndarray: 二值哈希向量
    """
    # 提取多尺度特征矩阵
    features_matrix = extract_multiscale_features(img, scales)
    # 融合多尺度特征并降维到目标维度
    fused_feature = fuse_features(features_matrix, hash_size)
    # 采用融合特征的中值作为二值化阈值
    median_val = np.median(fused_feature)
    binary_hash = fused_feature > median_val
    return binary_hash

def multiscale_vit_deep(img, scales=[1.0, 0.75, 0.5]):
    """
    基于多尺度 ViT 特征融合生成深度特征向量（归一化后用于相似度计算）

    参数:
        img: 输入图像，支持 PIL.Image 或 torch.Tensor（形状为 (3, H, W)）
        scales (list): 多尺度因子列表

    返回:
        numpy.ndarray: 归一化的深度融合特征向量
    """
    features_matrix = extract_multiscale_features(img, scales)
    # 这里直接采用第一主成分进行融合；融合特征维度为所有尺度特征的维度
    d = features_matrix.shape[1]
    hash_length = int(np.sqrt(d))
    fused_feature = fuse_features(features_matrix, hash_length)
    norm = np.linalg.norm(fused_feature)
    if norm > 0:
        fused_feature = fused_feature / norm
    return fused_feature

def compute_multiscale_vit_distance(feature1, feature2):
    """
    计算两个多尺度深度特征之间的余弦距离

    参数:
        feature1, feature2 (numpy.ndarray): 深度特征向量

    返回:
        float: 余弦距离（值越小表示越相似）
    """
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    distance = 1.0 - similarity
    return distance


# ---------------- 示例调用 ----------------
if __name__ == "__main__":
    # 打开示例图像（确保图像路径正确）
    img_path = "example.jpg"
    img = Image.open(img_path).convert("RGB")
    
    # 生成多尺度二值哈希
    binary_hash = multiscale_vit_hash(img, hash_size=8, scales=[1.0, 0.75, 0.5])
    print("二值哈希：", binary_hash)
    
    # 生成多尺度深度特征向量
    deep_feature = multiscale_vit_deep(img, scales=[1.0, 0.75, 0.5])
    print("深度特征向量：", deep_feature)
