import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

class MultiScaleResNet(nn.Module):
    def __init__(self, target_size=(14, 14)):
        super().__init__()
        # 加载预训练ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 定义多尺度特征层
        self.feature_layers = {
            'layer1': self.resnet.layer1,
            'layer2': self.resnet.layer2,
            'layer3': self.resnet.layer3,
            'layer4': self.resnet.layer4
        }
        
        # 注册钩子存储特征
        self.features = {}
        for name, layer in self.feature_layers.items():
            layer.register_forward_hook(self._save_features(name))
        
        # 空间金字塔参数
        self.target_size = target_size  # 特征对齐的目标尺寸
    
    def _save_features(self, name):
        def hook(module, input, output):
            # 对特征进行自适应池化对齐
            self.features[name] = nn.AdaptiveAvgPool2d(self.target_size)(output)
        return hook
    
    def forward(self, x):
        _ = self.resnet(x)
        return self.features

class MultiScaleHashGenerator:
    def __init__(self, hash_dim=64, fusion_method='concat', decompose_method='pca'):
        """
        多尺度哈希生成器
        
        参数:
            hash_dim: 目标哈希维度
            fusion_method: 特征融合方法 ['concat', 'attention', 'gated']
            decompose_method: 矩阵分解方法 ['pca', 'grp', 'itq']
        """
        self.extractor = MultiScaleResNet()
        self.extractor.eval()
        
        # 配置参数
        self.hash_dim = hash_dim
        self.fusion_method = fusion_method
        self.decompose_method = decompose_method
        
        # 初始化融合模块
        if fusion_method == 'attention':
            self.fusion_layer = self._build_attention_fusion()
        elif fusion_method == 'gated':
            self.fusion_layer = self._build_gated_fusion()
    
    def _build_attention_fusion(self):
        """构建注意力融合模块"""
        return nn.Sequential(
            nn.Conv2d(4096, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 4, 1),
            nn.Softmax(dim=1)
        )
    
    def _build_gated_fusion(self):
        """构建门控融合模块"""
        return nn.LSTM(
            input_size=4096,
            hidden_size=1024,
            num_layers=2,
            bidirectional=True
        )
    
    def _feature_normalization(self, features):
        """特征标准化处理"""
        return [ (f - f.mean()) / (f.std() + 1e-8) for f in features ]
    
    def _spatial_pyramid_pooling(self, feature, levels=[1, 2, 4]):
        """空间金字塔池化"""
        spp = []
        for level in levels:
            h, w = feature.shape[2:]
            pool = nn.AdaptiveAvgPool2d((level, level))
            spp.append(pool(feature).view(feature.size(0), -1))
        return torch.cat(spp, dim=1)
    
    def extract_features(self, img_tensor):
        """
        提取多尺度特征并进行融合
        
        参数:
            img_tensor: 输入图像张量 (B, C, H, W)
        
        返回:
            融合后的特征向量 (B, D)
        """
        # 前向传播获取多尺度特征
        features = self.extractor(img_tensor)
        
        # 特征对齐与处理
        processed_features = []
        for name, feat in features.items():
            # 通道压缩
            reduced = nn.Conv2d(feat.size(1), 256, 1)(feat)
            # 空间金字塔池化
            spp = self._spatial_pyramid_pooling(reduced)
            processed_features.append(spp)
        
        # 特征融合
        if self.fusion_method == 'concat':
            fused = torch.cat(processed_features, dim=1)
        elif self.fusion_method == 'attention':
            attn_weights = self.fusion_layer(torch.stack(processed_features))
            fused = torch.sum(attn_weights * processed_features, dim=0)
        elif self.fusion_method == 'gated':
            lstm_out = self.fusion_layer(torch.stack(processed_features))[0]
            fused = lstm_out.mean(dim=0)
        
        return fused.squeeze()
    
    def train_decomposer(self, features):
        """
        训练矩阵分解器
        
        参数:
            features: 训练特征矩阵 (N, D)
        """
        # 标准化
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-8
        normalized = (features - self.mean) / self.std
        
        # 选择分解方法
        if self.decompose_method == 'pca':
            self.decomposer = PCA(n_components=self.hash_dim)
        elif self.decompose_method == 'grp':
            self.decomposer = GaussianRandomProjection(n_components=self.hash_dim)
        elif self.decompose_method == 'itq':
            from ITQ import ITQ  # 需要实现ITQ类
            self.decomposer = ITQ(n_bits=self.hash_dim)
        
        self.decomposer.fit(normalized)
    
    def generate_hash(self, feature):
        """
        生成二进制哈希码
        
        参数:
            feature: 单个特征向量 (D,)
        
        返回:
            二进制哈希码 (hash_dim,)
        """
        # 标准化
        normalized = (feature - self.mean) / self.std
        
        # 矩阵分解
        if self.decompose_method == 'itq':
            projected = self.decomposer.transform(normalized.reshape(1, -1))
            hash_code = (projected > 0).astype(int).flatten()
        else:
            projected = self.decomposer.transform(normalized.reshape(1, -1))
            hash_code = (projected > np.median(projected)).astype(int).flatten()
        
        return hash_code

# 使用示例
if __name__ == "__main__":
    # 初始化生成器
    hash_gen = MultiScaleHashGenerator(
        hash_dim=64,
        fusion_method='attention',
        decompose_method='itq'
    )
    
    # 训练分解器（需要准备训练数据）
    train_features = np.random.randn(1000, 4096)  # 示例训练数据
    hash_gen.train_decomposer(train_features)
    
    # 生成单个哈希
    sample_feature = np.random.randn(4096)
    hash_code = hash_gen.generate_hash(sample_feature)
    print(f"生成哈希码: {hash_code[:10]}... (维度: {len(hash_code)})")