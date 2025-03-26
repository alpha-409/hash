import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import tensorly as tl
from tensorly.decomposition import tucker

# 设置tensorly后端为PyTorch并启用GPU
tl.set_backend('pytorch')
if torch.cuda.is_available():
    tl.set_device('cuda')

class ViTFeatureExtractor:
    def __init__(self, model_name="vit_b_16"):
        """
        初始化ViT特征提取器，使用torchvision模型
        
        参数:
            model_name (str): 模型名称，可选: vit_b_16, vit_b_32, vit_l_16, vit_l_32
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
        
        # 图像预处理
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
            features = self.model(img)
        
        # 转换为NumPy数组
        features = features.cpu().numpy()
        
        # 如果是批次，只返回第一个样本的特征
        if features.shape[0] == 1:
            features = features.squeeze(0)
        
        # 归一化特征
        features = features / np.linalg.norm(features)
        
        return features

class ViTTensorHashExtractor(ViTFeatureExtractor):
    def __init__(self, model_name="vit_b_16", ranks=(4, 4, 4)):
        super().__init__(model_name)
        self.ranks = ranks  # Tucker分解的秩
        self.layer_features = []  # 存储中间层特征
        self._register_hooks()  # 注册钩子获取中间层特征

    def _register_hooks(self):
        """注册钩子获取中间Transformer层的特征"""
        def hook_fn(module, input, output):
            # 获取每个Transformer层的输出特征
            cls_token = output[:, 0]  # [batch_size, dim]
            patch_tokens = output[:, 1:]  # [batch_size, num_patches, dim]
            pooled = patch_tokens.mean(dim=1)  # 空间维度池化
            self.layer_features.append(pooled.detach())

        # 注册钩子（示例：获取前12个Encoder层）
        for i, layer in enumerate(self.model.encoder.layers):
            if i < 12:  # 选择前12层
                layer.register_forward_hook(hook_fn)

    def extract_tensor_features(self, img):
        """提取多层特征并构造3D张量"""
        self.layer_features = []  # 清空历史特征
        
        # 预处理并前向传播
        if isinstance(img, Image.Image):
            img = self.preprocess(img)
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device)
        
        with torch.no_grad():
            _ = self.model(img)  # 触发钩子收集特征
        
        # 保持特征在GPU上，不要过早转换为NumPy
        if self.device.type == 'cuda':
            # 直接在GPU上构建张量
            tensor = torch.stack(self.layer_features)
            return tensor
        else:
            # CPU模式下的原始逻辑
            features = [f.cpu().numpy() for f in self.layer_features]
            tensor = np.stack(features)
            return torch.tensor(tensor, dtype=torch.float32)

    def generate_tensor_hash(self, img, hash_size=64):
        """基于张量分解的哈希生成"""
        # 1. 提取特征张量
        tensor = self.extract_tensor_features(img)
        
        # 2. 确保张量在GPU上
        if self.device.type == 'cuda' and tensor.device.type != 'cuda':
            tensor = tensor.to(self.device)
        
        # 3. Tucker分解降维 - 直接在GPU上执行
        core, _ = tucker(tensor, rank=self.ranks)
        
        # 4. 只在最后阶段将结果转移到CPU
        core_vector = core.flatten()
        
        # 计算阈值 - 在GPU上完成
        sorted_values, _ = torch.sort(core_vector)
        threshold_idx = int(len(sorted_values) * 0.6)
        threshold = sorted_values[threshold_idx]
        
        # 生成哈希码 - 在GPU上完成
        hash_code = (core_vector > threshold).cpu().numpy().astype(np.uint8)
        
        # 5. 截断到目标长度 - 在CPU上完成
        if len(hash_code) > hash_size:
            # 选择方差最大的位
            importance = np.abs(core_vector.cpu().numpy() - threshold.cpu().numpy())
            top_indices = np.argsort(importance)[-hash_size:]
            selected_hash = np.zeros(hash_size, dtype=np.uint8)
            for i, idx in enumerate(sorted(top_indices)):
                selected_hash[i] = hash_code[idx]
            hash_code = selected_hash
        elif len(hash_code) < hash_size:
            # 填充到指定长度
            padding = np.zeros(hash_size - len(hash_code), dtype=np.uint8)
            hash_code = np.concatenate([hash_code, padding])
        
        return hash_code.astype(bool)

# 修改原有vit_hash函数
def vit_tensor_hash(img, hash_size=64):
    """基于张量分解的ViT哈希生成函数"""
    if not hasattr(vit_tensor_hash, 'extractor'):
        vit_tensor_hash.extractor = ViTTensorHashExtractor(ranks=(4, 4, 4))
    return vit_tensor_hash.extractor.generate_tensor_hash(img, hash_size)

# 保持原有vit_hash函数不变，添加新函数
def vit_hash(img, hash_size=8):
    """
    使用ViT提取特征并生成哈希
    
    参数:
        img: 输入图像
        hash_size: 哈希大小
        
    返回:
        二进制哈希值
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(vit_hash, 'extractor'):
        vit_hash.extractor = ViTFeatureExtractor()
    
    # 提取特征
    features = vit_hash.extractor.extract_features(img)
    
    # 如果需要，可以使用PCA或其他方法降维到指定的hash_size
    # 这里简单地取前hash_size*hash_size个元素
    if hash_size * hash_size < len(features):
        features = features[:hash_size * hash_size]
    
    # 计算特征的中值
    median_value = np.median(features)
    
    # 生成二进制哈希
    hash_value = features > median_value
    
    return hash_value

def vit_deep(img, feature_dim=None):
    """
    使用ViT提取深度特征用于相似度计算
    
    参数:
        img: 输入图像
        feature_dim: 特征维度，默认为None（使用完整特征）
        
    返回:
        特征向量
    """
    # 创建特征提取器（如果尚未创建）
    if not hasattr(vit_deep, 'extractor'):
        vit_deep.extractor = ViTFeatureExtractor()
    
    # 提取特征
    features = vit_deep.extractor.extract_features(img)
    
    # 如果指定了特征维度，截取前feature_dim个元素
    if feature_dim is not None and feature_dim < len(features):
        features = features[:feature_dim]
    
    return features

def compute_vit_distance(feature1, feature2):
    """
    计算两个ViT特征向量之间的距离
    
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

# 添加张量哈希距离计算函数
def compute_vit_tensor_distance(hash1, hash2):
    """计算两个张量哈希之间的汉明距离"""
    # 确保输入是布尔数组
    if not isinstance(hash1, np.ndarray) or not isinstance(hash2, np.ndarray):
        hash1 = np.array(hash1, dtype=bool)
        hash2 = np.array(hash2, dtype=bool)
    
    # 计算汉明距离
    distance = np.sum(hash1 != hash2) / len(hash1)
    return distance