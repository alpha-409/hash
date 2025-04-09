import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import time
from tensorly.decomposition import parafac
import tensorly as tl
tl.set_backend('pytorch')

# 设置张量库后端为PyTorch
tl.set_backend('pytorch')

class SalientRegionDetector(nn.Module):
    """视觉显著性模型 (Salient Region Detector, SRD)"""
    def __init__(self):
        super(SalientRegionDetector, self).__init__()
        # 可学习的融合权重参数
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
        # 高斯核参数
        self.sigma = 1.5
        self.radius = 5
        
        # 颜色量化参数
        self.num_colors = 16
        
        # 用于边缘检测的卷积核
        self.edge_kernel_x = nn.Parameter(torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                      dtype=torch.float32).view(1, 1, 3, 3),
                                        requires_grad=False)
        self.edge_kernel_y = nn.Parameter(torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                      dtype=torch.float32).view(1, 1, 3, 3),
                                        requires_grad=False)
        
        # 高斯核生成
        self.gaussian_kernel = self._create_gaussian_kernel(self.radius, self.sigma)
        
    def _create_gaussian_kernel(self, radius, sigma):
        """创建高斯核"""
        kernel_size = 2 * radius + 1
        x, y = torch.meshgrid(torch.linspace(-radius, radius, kernel_size),
                             torch.linspace(-radius, radius, kernel_size))
        kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return nn.Parameter(kernel.view(1, 1, kernel_size, kernel_size).float(), requires_grad=False)
    
    def _calculate_edge_density(self, x):
        """计算边缘密度"""
        # 转换为灰度图
        x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # 计算梯度
        grad_x = F.conv2d(x_gray, self.edge_kernel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.edge_kernel_y, padding=1)
        
        # 计算梯度幅度
        edge_map = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
        
        # Canny边缘的简化版: 使用自适应阈值
        batch_size = x.size(0)
        edge_binary = torch.zeros_like(edge_map)
        
        for i in range(batch_size):
            # 对每个样本计算自适应阈值
            edge = edge_map[i, 0]
            high_threshold = torch.quantile(edge.flatten(), 0.8)  # 高阈值: 80%分位数
            low_threshold = high_threshold * 0.5  # 低阈值: 高阈值的一半
            
            # 应用阈值
            strong_edges = (edge > high_threshold).float()
            weak_edges = ((edge <= high_threshold) & (edge >= low_threshold)).float()
            
            # 简化的滞后阈值处理
            final_edges = strong_edges.clone()
            for _ in range(2):  # 简化的连接过程, 迭代两次
                weak_neighbors = F.conv2d(
                    final_edges.unsqueeze(0).unsqueeze(0), 
                    torch.ones(1, 1, 3, 3).to(final_edges.device), 
                    padding=1
                ).squeeze() > 0
                final_edges = torch.logical_or(final_edges, torch.logical_and(weak_edges, weak_neighbors)).float()
            
            edge_binary[i, 0] = final_edges
        
        # 计算边缘密度: 使用高斯核对边缘二值图进行卷积
        edge_density = F.conv2d(edge_binary, self.gaussian_kernel, padding=self.radius)
        
        return edge_density
    
    def _calculate_color_sparsity(self, x):
        """计算颜色稀疏性"""
        batch_size, channels, height, width = x.size()
        
        # 将图像重塑为像素列表 [B, C, H*W] -> [B, H*W, C]
        pixels = x.reshape(batch_size, channels, -1).permute(0, 2, 1)
        
        # 对每个batch的图像单独处理
        color_sparsity_maps = []
        
        for b in range(batch_size):
            # 使用K-means聚类进行颜色量化 (PyTorch没有原生K-means, 这里模拟实现)
            # 在实际实现中可以使用scikit-learn的KMeans或更高效的颜色量化算法
            pixels_np = pixels[b].detach().cpu().numpy()
            
            # 设置kmeans的终止条件
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            # 应用kmeans
            _, labels, centers = cv2.kmeans(
                pixels_np.astype(np.float32), 
                self.num_colors, 
                None, 
                criteria, 
                10, 
                cv2.KMEANS_RANDOM_CENTERS
            )
            
            # 统计每个颜色簇的像素数量
            cluster_counts = np.bincount(labels.flatten(), minlength=self.num_colors)
            
            # 计算每个像素的颜色稀疏性 (负对数似然)
            total_pixels = height * width
            color_sparsity = -np.log((cluster_counts[labels.flatten()] / total_pixels) + 1e-9)
            
            # 重塑回原始空间尺寸 [H*W] -> [H, W]
            color_sparsity_map = torch.from_numpy(color_sparsity.reshape(height, width)).float().to(x.device)
            color_sparsity_maps.append(color_sparsity_map.unsqueeze(0))
        
        # 叠加所有batch的结果 [B, 1, H, W]
        color_sparsity = torch.cat(color_sparsity_maps, dim=0).unsqueeze(1)
        
        return color_sparsity
    
    def _normalize(self, x):
        """Min-Max标准化"""
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1)
        x_min = x_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
        x_max = x_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
        
        # 避免除零错误
        x_range = x_max - x_min
        x_range[x_range == 0] = 1.0
        
        return (x - x_min) / x_range
    
    def forward(self, x):
        """前向传播"""
        # 计算边缘密度
        edge_density = self._calculate_edge_density(x)
        
        # 计算颜色稀疏性
        color_sparsity = self._calculate_color_sparsity(x)
        
        # 归一化
        norm_edge_density = self._normalize(edge_density)
        norm_color_sparsity = self._normalize(color_sparsity)
        
        # 融合显著性图
        saliency_map = torch.sigmoid(
            self.alpha * norm_edge_density + self.beta * norm_color_sparsity
        )
        
        return saliency_map


class DeformableAttentionGate(nn.Module):
    """可变形注意力门控模块"""
    def __init__(self, in_channels):
        super(DeformableAttentionGate, self).__init__()
        # 由于PyTorch中没有直接的可变形卷积实现,这里使用普通卷积进行模拟
        # 在实际项目中,应该使用DCNv2等实现
        self.detail_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.semantic_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, detail_features, semantic_features):
        """
        输入:
            detail_features: 细节路径特征, shape [B, C_d, H, W]
            semantic_features: 语义路径特征(上采样后), shape [B, C_s, H, W]
        输出:
            权重图: shape [B, 1, H, W], 值域(0,1)
        """
        # 先将两种特征投影到相同的通道数
        detail_proj = self.detail_conv(detail_features)
        semantic_proj = self.semantic_conv(semantic_features)
        
        # 拼接特征
        concat_features = torch.cat([detail_proj, semantic_proj], dim=1)
        
        # 生成注意力权重
        x = F.relu(self.conv1(concat_features))
        x = F.relu(self.conv2(x))
        weights = torch.sigmoid(self.conv3(x))
        
        return weights


class DynamicFeatureExtractor(nn.Module):
    """动态多尺度特征金字塔 (Dynamic Multi-Scale Feature Pyramid, DFE)"""
    def __init__(self, detail_channels=64, semantic_channels=512):
        super(DynamicFeatureExtractor, self).__init__()
        
        # 细节路径 (Detail Path)
        self.detail_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(48, 56, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(56, detail_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(detail_channels),
            nn.ReLU(inplace=True)
        )
        
        # 语义路径 (Semantic Path) - ResNet-18
        resnet = models.resnet18(pretrained=True)
        self.semantic_path = nn.Sequential(*list(resnet.children())[:-2])  # 移除平均池化和FC层
        
        # 特征调整层: 将ResNet输出调整为所需的语义特征通道数
        self.semantic_projection = nn.Conv2d(512, detail_channels, kernel_size=1)  # 修改为与detail_channels相同
        
        # 可变形注意力门控
        self.dag = DeformableAttentionGate(detail_channels)
        
    def forward(self, x, saliency_map=None):
        """
        输入:
            x: 输入图像, shape [B, 3, H, W]
            saliency_map: 显著性图, shape [B, 1, H, W], 可选
        输出:
            fused_features: 融合后的特征, shape [B, C_d, H, W]
        """
        # 如果提供了显著性图, 将其与输入相乘以突出显著区域
        if saliency_map is not None:
            x_weighted = x * saliency_map
        else:
            x_weighted = x
        
        # 提取细节特征
        detail_features = self.detail_path(x_weighted)
        
        # 提取语义特征
        semantic_features = self.semantic_path(x)
        semantic_features = self.semantic_projection(semantic_features)
        
        # 上采样语义特征以匹配细节特征的空间尺寸
        semantic_upsampled = F.interpolate(
            semantic_features, 
            size=detail_features.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 应用可变形注意力门控生成权重图
        weights = self.dag(detail_features, semantic_upsampled)
        
        # 融合特征
        fused_features = semantic_upsampled * weights + detail_features * (1 - weights)
        
        return fused_features


class CPHashLayer(nn.Module):
    """基于CP分解的哈希层 (CP-Hash Layer)"""
    def __init__(self, input_shape, rank=32, hash_length=128):
        """
        参数:
            input_shape: 输入特征的形状, 例如(B, H, W, C)或(B, C, H, W)
            rank: CP分解的秩
            hash_length: 输出哈希码长度
        """
        super(CPHashLayer, self).__init__()
        self.rank = rank
        self.hash_length = hash_length
        self.input_shape = input_shape  # (H, W, C) 或 (C, H, W)
        
        # 计算分解后的因子向量总长度
        factor_length = rank * sum(input_shape)
        
        # 哈希生成MLP
        self.hash_mlp = nn.Sequential(
            nn.Linear(factor_length, factor_length // 2),
            nn.ReLU(inplace=True),
            nn.Linear(factor_length // 2, hash_length)
        )
        
    def forward(self, x, training=True):
        """
        前向传播
        
        参数:
            x: 输入特征张量, shape [B, C, H, W]
            training: 训练模式标志
            
        返回:
            binary_hash: 二值哈希码, shape [B, hash_length]
            continuous_hash: 连续哈希值(训练时用于反向传播), shape [B, hash_length]
        """
        batch_size = x.shape[0]
        
        # 重塑为三阶张量
        h, w, c = self.input_shape
        reshaped_x = x.reshape(batch_size, c, h, w)
        
        # 转换为张量库张量格式
        tl_tensor = tl.tensor(reshaped_x)
        
        # 对每个样本单独进行CP分解
        factor_vectors_batch = []
        
        for i in range(batch_size):
            # CP分解当前样本
            # 注意: 在训练时, 每次前向传播都要进行CP分解, 这可能很耗时
            # 在实际应用中, 可以考虑使用可学习的因子向量或近似方法提高效率
            factors = parafac(tl_tensor[i], rank=self.rank, n_iter_max=10, init='random')
            
            # 获取分解后的因子矩阵
            # 检查factors的类型和结构
            if isinstance(factors, tuple) and len(factors) == 3:
                A, B, C = factors
            elif hasattr(factors, 'factors'):
                # 如果factors是一个对象，尝试从其factors属性获取因子
                factor_list = factors.factors
                if len(factor_list) == 3:
                    A, B, C = factor_list
                else:
                    # 如果因子数量不是3，创建适当的空因子
                    A = factor_list[0]
                    B = factor_list[1] if len(factor_list) > 1 else torch.zeros_like(A)
                    C = factor_list[2] if len(factor_list) > 2 else torch.zeros_like(A)
            else:
                # 如果无法获取3个因子，创建空因子
                print(f"警告: CP分解返回了意外的结构: {type(factors)}")
                # 创建随机因子以避免错误
                A = torch.randn(h, self.rank, device=x.device)
                B = torch.randn(w, self.rank, device=x.device)
                C = torch.randn(c, self.rank, device=x.device)
            
            # 拼接所有因子向量
            factor_vectors = torch.cat([
                A.flatten(), B.flatten(), C.flatten()
            ])
            
            factor_vectors_batch.append(factor_vectors)
        
        # 将batch中的所有因子向量叠加
        factor_batch = torch.stack(factor_vectors_batch)
        
        # 通过MLP生成连续哈希值
        continuous_hash = self.hash_mlp(factor_batch)
        
        # 生成二值哈希码
        if training:
            # 使用Straight-Through估计器进行二值化(训练时)
            binary_hash = torch.sign(continuous_hash)
            # 直通估计器: 前向传播用二值结果, 反向传播用连续值的梯度
            # 使用detach来阻断二值化操作的梯度流
            binary_hash = binary_hash.detach() - continuous_hash.detach() + continuous_hash
        else:
            # 测试时直接二值化
            binary_hash = torch.sign(continuous_hash)
        
        return binary_hash, continuous_hash


class HashNet(nn.Module):
    """完整的哈希网络"""
    def __init__(self, hash_length=128, cp_rank=32, detail_channels=64, semantic_channels=512):
        super(HashNet, self).__init__()
        
        # 视觉显著性模型
        self.srd = SalientRegionDetector()
        
        # 动态多尺度特征提取器
        self.dfe = DynamicFeatureExtractor(detail_channels=detail_channels, semantic_channels=semantic_channels)
        
        # 特征池化
        self.gap = nn.AdaptiveAvgPool2d((7, 7))  # 输出固定大小特征图
        
        # CP哈希层
        self.cp_hash = CPHashLayer(input_shape=(7, 7, detail_channels), rank=cp_rank, hash_length=hash_length)
        
    def forward(self, x, training=True):
        """
        前向传播
        
        参数:
            x: 输入图像, shape [B, 3, H, W]
            training: 训练模式标志
            
        返回:
            binary_hash: 二值哈希码
            continuous_hash: 连续哈希值(用于训练)
            saliency_map: 显著性图
        """
        # 生成显著性图
        saliency_map = self.srd(x)
        
        # 特征提取
        features = self.dfe(x, saliency_map)
        
        # 特征池化
        pooled_features = self.gap(features)
        
        # 哈希码生成
        binary_hash, continuous_hash = self.cp_hash(pooled_features, training)
        
        return binary_hash, continuous_hash, saliency_map


# 哈希检索三重损失函数
class TripletHashLoss(nn.Module):
    """哈希检索的三重损失函数"""
    def __init__(self, lambda1=1.0, lambda2=0.5, lambda3=0.1):
        super(TripletHashLoss, self).__init__()
        self.lambda1 = lambda1  # 相似性保持损失权重
        self.lambda2 = lambda2  # 量化损失权重
        self.lambda3 = lambda3  # 比特平衡损失权重
        
    def forward(self, binary_hash, continuous_hash, positives):
        """
        计算三重损失
        
        参数:
            binary_hash: 二值哈希码, shape [B, hash_length]
            continuous_hash: 连续哈希值, shape [B, hash_length]
            positives: 正样本对列表, 每对为(query_idx, db_idx)
            
        返回:
            total_loss: 总损失
            loss_components: 各组成部分损失
        """
        # 相似性保持损失
        similarity_loss = self._similarity_preserving_loss(binary_hash, positives)
        
        # 量化损失
        quantization_loss = self._quantization_loss(continuous_hash)
        
        # 比特平衡损失
        bit_balance_loss = self._bit_balance_loss(binary_hash)
        
        # 总损失
        total_loss = (
            self.lambda1 * similarity_loss + 
            self.lambda2 * quantization_loss + 
            self.lambda3 * bit_balance_loss
        )
        
        loss_components = {
            "similarity_loss": similarity_loss.item(),
            "quantization_loss": quantization_loss.item(),
            "bit_balance_loss": bit_balance_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _similarity_preserving_loss(self, binary_hash, positives):
        """相似性保持损失"""
        if not positives:
            return torch.tensor(0.0, device=binary_hash.device)
        
        batch_size = binary_hash.shape[0]
        hash_length = binary_hash.shape[1]
        margin = 0.5 * hash_length  # 设置合适的边界值
        
        loss = 0.0
        num_pairs = 0
        
        # 计算正样本对的损失
        for q_idx, db_idx in positives:
            if q_idx < batch_size and db_idx < batch_size:
                # 计算哈希码间的点积
                dot_product = torch.sum(binary_hash[q_idx] * binary_hash[db_idx])
                
                # 计算损失: max(0, margin - dot_product)^2 (对于正样本对)
                pair_loss = torch.clamp(margin - dot_product, min=0.0) ** 2
                loss += pair_loss
                num_pairs += 1
        
        # 防止除零
        if num_pairs == 0:
            return torch.tensor(0.0, device=binary_hash.device)
        
        # 添加负样本对损失
        # 这里简化处理: 随机选择一些非正样本对作为负样本对
        neg_loss = 0.0
        num_neg_pairs = 0
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and (i, j) not in positives and (j, i) not in positives:
                    # 计算哈希码间的点积
                    dot_product = torch.sum(binary_hash[i] * binary_hash[j])
                    
                    # 计算损失: max(0, dot_product + margin)^2 (对于负样本对)
                    pair_loss = torch.clamp(dot_product + margin, min=0.0) ** 2
                    neg_loss += pair_loss
                    num_neg_pairs += 1
                    
                    # 限制负样本对数量与正样本对相当
                    if num_neg_pairs >= num_pairs:
                        break
            if num_neg_pairs >= num_pairs:
                break
        
        # 将正样本损失和负样本损失组合
        total_loss = (loss + neg_loss) / (num_pairs + num_neg_pairs)
        
        return total_loss
    
    def _quantization_loss(self, continuous_hash):
        """量化损失"""
        # 计算连续哈希值的L1范数与全1向量之间的差异
        # |continuous_hash| - 1
        return torch.mean(torch.abs(torch.abs(continuous_hash) - 1.0))
    
    def _bit_balance_loss(self, binary_hash):
        """比特平衡损失"""
        # 计算每个位的平均值, 理想情况下应接近0
        bit_means = torch.mean(binary_hash, dim=0)
        
        # 计算均值的L2范数的平方
        return torch.sum(bit_means ** 2)
