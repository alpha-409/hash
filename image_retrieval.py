"""
图像检索和评估模块
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from hash_algorithms import ImageHash
import torch
from tqdm import tqdm

class ImageRetrieval:
    def __init__(self, hash_method: str = 'phash', hash_size: int = 8, device: str = 'cpu'):
        """
        初始化图像检索系统
        
        参数:
            hash_method: 哈希方法 ('ahash', 'dhash', 或 'phash')
            hash_size: 哈希大小
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.hash_method = hash_method
        self.hash_size = hash_size
        self.hasher = ImageHash(device)
        self.database = {}  # 存储图像哈希值
        
    def index_images(self, images: Dict[str, Dict], batch_size: int = 32) -> None:
        """
        为图像数据库建立索引
        """
        image_paths = [(id_, info['path']) for id_, info in images.items()]
        image_ids, paths = zip(*image_paths)
        
        print(f"Indexing {len(paths)} images...")
        hashes = self.hasher.compute_batch_hash(
            list(paths),
            self.hash_size,
            self.hash_method,
            batch_size
        )
        
        self.database = dict(zip(image_ids, hashes))
    
    def search(self, query_images: Dict[str, Dict], 
              threshold: float = 0.8) -> Dict[str, List[Tuple[str, float]]]:
        """
        搜索相似图像
        
        参数:
            query_images: 查询图像字典
            threshold: 相似度阈值 (用于μAP计算)
            
        返回:
            Dict[query_id, List[(database_id, similarity_score)]]
        """
        results = {}
        # 批量计算查询图像哈希值
        query_paths = [(id_, info['path']) for id_, info in query_images.items()]
        query_ids, paths = zip(*query_paths)
        
        print(f"Computing hashes for {len(paths)} query images...")
        query_hashes = self.hasher.compute_batch_hash(
            list(paths),
            self.hash_size,
            self.hash_method
        )
        query_hash_dict = dict(zip(query_ids, query_hashes))
        
        print("Performing image retrieval...")
        max_hamming = self.hash_size * self.hash_size
        for query_id, query_hash in tqdm(query_hash_dict.items()):
            # 计算与所有数据库图像的距离
            distances = []
            for db_id, db_hash in self.database.items():
                hamming_dist = self.hasher.hamming_distance(query_hash, db_hash)
                similarity = 1 - hamming_dist / max_hamming
                distances.append((db_id, similarity))
            
            # 按相似度降序排序
            distances.sort(key=lambda x: x[1], reverse=True)
            results[query_id] = distances
            
        return results

def calculate_map(search_results: Dict[str, List[Tuple[str, float]]], 
                 ground_truth: Dict[str, str]) -> float:
    """
    计算mAP (Mean Average Precision)
    """
    aps = []
    for query_id, results in search_results.items():
        if query_id not in ground_truth:
            continue
            
        true_match = ground_truth[query_id]
        precision_list = []
        
        for rank, (result_id, _) in enumerate(results, 1):
            if result_id == true_match:
                precision = 1.0 / rank
                precision_list.append(precision)
                
        ap = np.mean(precision_list) if precision_list else 0
        aps.append(ap)
    
    return np.mean(aps) if aps else 0

def calculate_micro_ap(search_results: Dict[str, List[Tuple[str, float]]], 
                      ground_truth: Dict[str, str],
                      threshold: float = 0.8) -> float:
    """
    计算μAP (Micro Average Precision)
    
    参数:
        search_results: 检索结果
        ground_truth: 真实匹配关系
        threshold: 相似度阈值
    """
    total_tp = 0
    total_fp = 0
    
    for query_id, results in search_results.items():
        if query_id not in ground_truth:
            continue
            
        true_match = ground_truth[query_id]
        
        # 只考虑相似度大于阈值的结果
        filtered_results = [(id_, score) for id_, score in results if score >= threshold]
        
        # 统计TP和FP
        for result_id, _ in filtered_results:
            if result_id == true_match:
                total_tp += 1
            else:
                total_fp += 1
                
    # 计算μAP
    return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

def evaluate_retrieval(search_results: Dict[str, List[Tuple[str, float]]], 
                      ground_truth: Dict[str, str],
                      threshold: float = 0.8) -> Dict[str, float]:
    """
    综合评估检索性能
    """
    metrics = {}
    
    # 计算mAP
    metrics['mAP'] = calculate_map(search_results, ground_truth)
    
    # 计算μAP
    metrics['μAP'] = calculate_micro_ap(search_results, ground_truth, threshold)
    
    # 计算不同k值的precision和recall
    k_values = [1, 5, 10]
    for k in k_values:
        precisions = []
        recalls = []
        
        for query_id, results in search_results.items():
            if query_id not in ground_truth:
                continue
                
            true_match = ground_truth[query_id]
            top_k_results = [r[0] for r in results[:k]]
            
            # Precision@k
            correct = sum(1 for r in top_k_results if r == true_match)
            precision = correct / k if k > 0 else 0
            precisions.append(precision)
            
            # Recall@k
            recall = 1.0 if true_match in top_k_results else 0.0
            recalls.append(recall)
        
        metrics[f'P@{k}'] = np.mean(precisions) if precisions else 0
        metrics[f'R@{k}'] = np.mean(recalls) if recalls else 0
    
    return metrics
