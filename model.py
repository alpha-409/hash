import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import imagehash
import torch
from tqdm import tqdm
import time

class HashImageCopyDetector:
    """基于图像哈希的拷贝检测器，支持CUDA加速"""
    
    def __init__(self, hash_size=8, hash_method='phash', use_cuda=False):
        """
        初始化哈希图像拷贝检测器
        
        参数:
            hash_size (int): 哈希大小 (hash_size x hash_size 位)
            hash_method (str): 哈希方法, 'phash', 'dhash', 或 'ahash'
            use_cuda (bool): 是否使用CUDA加速哈希比较
        """
        self.hash_size = hash_size
        self.hash_method = hash_method
        self.reference_hashes = {}  # {image_id: hash_value}
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if self.use_cuda:
            print(f"CUDA加速已启用，使用设备: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA加速未启用，使用CPU")
        
    def compute_hash(self, image_path_or_data):
        """
        计算图像的哈希值
        
        参数:
            image_path_or_data: 图像路径或已加载的图像数据
            
        返回:
            imagehash: 图像的哈希值
        """
        if isinstance(image_path_or_data, str):
            img = Image.open(image_path_or_data).convert('RGB')
        else:
            img = Image.fromarray(image_path_or_data).convert('RGB')
            
        if self.hash_method == 'phash':
            return imagehash.phash(img, hash_size=self.hash_size)
        elif self.hash_method == 'dhash':
            return imagehash.dhash(img, hash_size=self.hash_size)
        elif self.hash_method == 'ahash':
            return imagehash.average_hash(img, hash_size=self.hash_size)
        else:
            raise ValueError(f"不支持的哈希方法: {self.hash_method}")
    
    def index_reference_images(self, original_images: Dict[str, Dict[str, Any]]):
        """
        为所有参考图像计算哈希值并建立索引
        
        参数:
            original_images: 原始图像字典 (来自load_copydays_dataset函数)
        """
        print(f"为参考图像建立{self.hash_method}哈希索引...")
        for image_id, image_info in tqdm(original_images.items()):
            # 检查图像数据是否已加载
            if image_info['data'] is not None:
                hash_value = self.compute_hash(image_info['data'])
            else:
                hash_value = self.compute_hash(image_info['path'])
                
            self.reference_hashes[image_id] = hash_value
            
        print(f"索引完成，共{len(self.reference_hashes)}张参考图像")
        
        # 如果使用CUDA，预先将哈希值转换为张量
        if self.use_cuda:
            self._prepare_cuda_tensors()
    
    def _prepare_cuda_tensors(self):
        """为CUDA加速准备哈希张量"""
        # 获取所有参考哈希值和ID
        self.ref_ids = list(self.reference_hashes.keys())
        
        # 将哈希转换为二进制数组，然后转换为张量
        hash_arrays = []
        for ref_id in self.ref_ids:
            # 将imagehash对象转换为二进制数组
            hash_bin = []
            hash_hex = str(self.reference_hashes[ref_id])
            for h in hash_hex:
                bin_value = bin(int(h, 16))[2:].zfill(4)
                hash_bin.extend([int(b) for b in bin_value])
                
            # 确保二进制数组长度正确（特别是对于大哈希值）
            expected_len = self.hash_size * self.hash_size
            if len(hash_bin) < expected_len:
                # 某些哈希方法可能返回不足的位数，使用填充
                hash_bin.extend([0] * (expected_len - len(hash_bin)))
            elif len(hash_bin) > expected_len:
                # 或者截断多余的位
                hash_bin = hash_bin[:expected_len]
                
            hash_arrays.append(hash_bin)
        
        # 创建张量并移动到GPU
        self.ref_hash_tensor = torch.tensor(hash_arrays, dtype=torch.float32).cuda()
    
    def compute_similarity_cuda(self, query_hash):
        """使用CUDA计算查询哈希与所有参考哈希的相似度"""
        # 将查询哈希转为二进制数组
        query_bin = []
        hash_hex = str(query_hash)
        for h in hash_hex:
            bin_value = bin(int(h, 16))[2:].zfill(4)
            query_bin.extend([int(b) for b in bin_value])
            
        # 确保长度正确
        expected_len = self.hash_size * self.hash_size
        if len(query_bin) < expected_len:
            query_bin.extend([0] * (expected_len - len(query_bin)))
        elif len(query_bin) > expected_len:
            query_bin = query_bin[:expected_len]
        
        # 转为CUDA张量
        query_tensor = torch.tensor(query_bin, dtype=torch.float32).cuda()
        
        # 广播计算汉明距离（使用XOR和SUM）
        xor_result = torch.logical_xor(
            query_tensor.view(1, -1), 
            self.ref_hash_tensor.to(torch.bool)
        ).to(torch.float32)
        hamming_distances = torch.sum(xor_result, dim=1)
        
        # 转换为相似度（1.0 - 归一化的距离）
        max_bits = self.hash_size * self.hash_size
        similarity_scores = 1.0 - (hamming_distances / max_bits)
        
        # 返回到CPU并转换为列表
        return similarity_scores.cpu().numpy()
    
    def compute_similarity(self, hash1, hash2):
        """
        计算两个哈希值的相似度 (0-1范围，1表示完全匹配)
        
        参数:
            hash1, hash2: 两个imagehash值
            
        返回:
            float: 相似度得分
        """
        # 计算哈希之间的汉明距离
        distance = hash1 - hash2
        # 转换为相似度 (0-1)
        max_bits = self.hash_size * self.hash_size
        similarity = 1.0 - (distance / max_bits)
        return similarity
    
    def search(self, query_images: Dict[str, Dict[str, Any]], 
               top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        使用哈希搜索查询图像的匹配项
        
        参数:
            query_images: 查询图像字典
            top_k: 每个查询返回的最大匹配数
            
        返回:
            dict: 查询结果，格式: {query_id: [(match_id, similarity_score), ...]}
        """
        search_results = {}
        start_time = time.time()
        
        print(f"使用{self.hash_method}哈希搜索匹配图像...")
        for query_id, query_info in tqdm(query_images.items()):
            # 计算查询图像哈希
            if query_info['data'] is not None:
                query_hash = self.compute_hash(query_info['data'])
            else:
                query_hash = self.compute_hash(query_info['path'])
            
            if self.use_cuda:
                # 使用GPU计算所有相似度
                similarity_scores = self.compute_similarity_cuda(query_hash)
                similarities = [(self.ref_ids[i], float(score)) for i, score in enumerate(similarity_scores)]
            else:
                # CPU版本: 与所有参考图像比较
                similarities = []
                for ref_id, ref_hash in self.reference_hashes.items():
                    similarity = self.compute_similarity(query_hash, ref_hash)
                    similarities.append((ref_id, similarity))
            
            # 按相似度从高到低排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            search_results[query_id] = similarities[:top_k]
        
        elapsed_time = time.time() - start_time
        print(f"搜索完成，用时：{elapsed_time:.2f}秒")
        return search_results
