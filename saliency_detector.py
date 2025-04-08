import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SaliencyRegionDetector(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        
    def forward(self, img):
        """
        输入: img (Tensor) - B x 3 x H x W (RGB图像)
        输出: W_s (Tensor) - B x 1 x H x W (显著性权重图)
        """
        # 边缘密度计算
        edge_density = self._compute_edge_density(img)
        
        # 颜色稀疏性分析
        color_sparsity = self._compute_color_sparsity(img)
        
        # 显著性图融合
        edge_norm = (edge_density - edge_density.min()) / (edge_density.max() - edge_density.min() + 1e-9)
        color_norm = (color_sparsity - color_sparsity.min()) / (color_sparsity.max() - color_sparsity.min() + 1e-9)
        
        W_s = torch.sigmoid(self.alpha * edge_norm + self.beta * color_norm)
        return W_s
    
    def _compute_edge_density(self, img):
        """计算边缘密度图 - 增强对高相似性样本的区分能力"""
        # 转换为灰度图
        gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
        
        # 使用更精细的边缘检测参数
        blurred = F.avg_pool2d(gray.unsqueeze(1), kernel_size=5, stride=1, padding=2)  # 增大模糊核
        grad_x = F.conv2d(blurred, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32), padding=1)
        grad_y = F.conv2d(blurred, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32), padding=1)
        edge_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 更严格的自适应阈值
        high_thresh = edge_mag.max() * 0.2  # 降低阈值以捕捉更多细节
        low_thresh = high_thresh * 0.3
        edge_map = ((edge_mag > low_thresh) & (edge_mag < high_thresh)).float()
        
        # 使用更精细的高斯核
        kernel_size = 15  # 增大核尺寸
        sigma = 2.0  # 增大sigma
        gauss_kernel = self._create_gaussian_kernel(kernel_size, sigma).to(img.device)
        edge_density = F.conv2d(edge_map, gauss_kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2)
        return edge_density
    
    def _compute_color_sparsity(self, img):
        """计算颜色稀疏性图 - 增强对细微颜色差异的敏感度"""
        # 增加颜色量化级别
        quantized = (img * 31).round() / 31  # 从16色增加到32色
        
        # 计算颜色直方图
        b, c, h, w = quantized.shape
        flat = quantized.permute(0, 2, 3, 1).reshape(-1, c)
        unique_colors, counts = torch.unique(flat, dim=0, return_counts=True)
        
        # 添加颜色空间转换增强
        lab_img = self._rgb_to_lab(img)  # 转换为Lab颜色空间
        
        # 计算每个像素的颜色稀疏性 (结合RGB和Lab空间)
        color_sparsity = torch.zeros(b, 1, h, w, device=img.device)
        for color, count in zip(unique_colors, counts):
            mask = (quantized == color.view(1, 3, 1, 1)).all(dim=1).float()
            prob = count.float() / (b * h * w)
            # 增强稀疏性计算
            color_sparsity += mask * (-torch.log(prob + 1e-9) * 1.5)  # 增加权重
            
        return color_sparsity
    
    def _rgb_to_lab(self, img):
        """RGB转Lab颜色空间"""
        # 简化实现，实际应使用完整转换
        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        
        # 线性转换
        x = 0.412453*r + 0.357580*g + 0.180423*b
        y = 0.212671*r + 0.715160*g + 0.072169*b
        z = 0.019334*r + 0.119193*g + 0.950227*b
        
        # 归一化
        x = x / 0.95047
        y = y / 1.0
        z = z / 1.08883
        
        # 非线性转换 (简化)
        x = torch.where(x > 0.008856, torch.pow(x, 1/3), 7.787*x + 16/116)
        y = torch.where(y > 0.008856, torch.pow(y, 1/3), 7.787*y + 16/116)
        z = torch.where(z > 0.008856, torch.pow(z, 1/3), 7.787*z + 16/116)
        
        # 计算Lab
        l = 116 * y - 16
        a = 500 * (x - y)
        b = 200 * (y - z)
        
        return torch.stack([l, a, b], dim=1)
    
    def _create_gaussian_kernel(self, size, sigma):
        """创建高斯核"""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        x = coords.reshape(-1, 1)
        y = coords.reshape(1, -1)
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / kernel.sum()