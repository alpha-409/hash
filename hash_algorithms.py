"""
图像哈希算法实现，支持CPU和CUDA加速
"""

import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple, Optional
from scipy.fft import dct
import cv2

class ImageHash:
    def __init__(self, device: str = 'cpu'):
        """
        初始化图像哈希类
        
        参数:
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
    @staticmethod
    def preprocess_image(image: Union[str, np.ndarray], size: Tuple[int, int], 
                        grayscale: bool = True) -> torch.Tensor:
        """
        预处理图像：加载、调整大小、转换为灰度
        """
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            raise TypeError("Unsupported image type")
            
        if grayscale:
            img = img.convert('L')
        img = img.resize(size, Image.Resampling.LANCZOS)
        return torch.from_numpy(np.array(img)).float()
    
    def compute_hash(self, img: torch.Tensor, hash_size: int = 8, method: str = 'phash') -> str:
        """
        计算图像哈希值
        
        参数:
            img: 输入图像
            hash_size: 哈希大小
            method: 哈希方法 ('ahash', 'dhash', 或 'phash')
            
        返回:
            十六进制哈希字符串
        """
        img = img.to(self.device)
        
        if method == 'ahash':
            hash_bits = self._average_hash(img, hash_size)
        elif method == 'dhash':
            hash_bits = self._difference_hash(img, hash_size)
        else:  # phash
            hash_bits = self._perceptual_hash(img, hash_size)
            
        # 转回CPU计算十六进制
        hash_bits = hash_bits.cpu().numpy()
        hash_bytes = np.packbits(hash_bits)
        return ''.join(f'{byte:02x}' for byte in hash_bytes)
    
    def _average_hash(self, img: torch.Tensor, hash_size: int) -> torch.Tensor:
        """平均哈希算法"""
        # 已经是灰度图且调整过大小的图像
        mean = img.mean()
        return (img >= mean).flatten()
    
    def _difference_hash(self, img: torch.Tensor, hash_size: int) -> torch.Tensor:
        """差异哈希算法"""
        # 计算水平差异
        diff = img[:, 1:] - img[:, :-1]
        return (diff >= 0).flatten()
    
    def _perceptual_hash(self, img: torch.Tensor, hash_size: int) -> torch.Tensor:
        """感知哈希算法"""
        # 如果在GPU上，需要转到CPU进行DCT
        if self.device == 'cuda':
            img = img.cpu()
            
        # 应用DCT变换
        dct_data = dct(dct(img.numpy(), axis=0), axis=1)
        dct_data = torch.from_numpy(dct_data)
        
        if self.device == 'cuda':
            dct_data = dct_data.cuda()
            
        # 取低频部分
        dct_low = dct_data[:hash_size, :hash_size]
        # 计算中值（不包括DC分量）
        median = torch.median(dct_low[1:].flatten())
        return (dct_low[1:] >= median).flatten()
    
    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """
        计算两个哈希值的汉明距离
        """
        # 将十六进制转换为二进制数组
        def hex_to_bits(hex_str):
            hash_bytes = bytes.fromhex(hex_str)
            return np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))
            
        bits1 = hex_to_bits(hash1)
        bits2 = hex_to_bits(hash2)
        return np.count_nonzero(bits1 != bits2)
    
    def compute_batch_hash(self, images: list, hash_size: int = 8, 
                          method: str = 'phash', batch_size: int = 32) -> list:
        """
        批量计算图像哈希值
        
        参数:
            images: 图像路径或numpy数组列表
            hash_size: 哈希大小
            method: 哈希方法
            batch_size: 批处理大小
            
        返回:
            哈希值列表
        """
        hashes = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            # 预处理批次图像
            processed_images = [
                self.preprocess_image(img, (hash_size, hash_size))
                for img in batch
            ]
            # 堆叠为批次张量
            batch_tensor = torch.stack(processed_images)
            if self.device == 'cuda':
                batch_tensor = batch_tensor.cuda()
                
            # 计算批次哈希值
            batch_hashes = [
                self.compute_hash(img, hash_size, method)
                for img in batch_tensor
            ]
            hashes.extend(batch_hashes)
            
        return hashes
