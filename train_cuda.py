import os
import pickle
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import imagehash
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import hamming
import torch
import time
from data import load_copydays_dataset
class HashImageCopyDetector:
    """基于图像哈希的拷贝检测器，支持CUDA加速"""
    
    def __init__(self, hash_size=16, hash_method='phash', use_cuda=False):
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


class CopyDetectionEvaluator:
    """图像拷贝检测算法的评估器"""
    
    def __init__(self, ground_truth: Dict[str, str]):
        """
        初始化评估器
        
        参数:
            ground_truth: 真实匹配关系，格式: {query_id: original_id}
        """
        self.ground_truth = ground_truth
        
    def calculate_map(self, search_results: Dict[str, List[Tuple[str, float]]]) -> float:
        """
        计算平均精度均值 (mAP)
        
        参数:
            search_results: 搜索结果，格式: {query_id: [(match_id, score), ...]}
            
        返回:
            float: mAP值
        """
        aps = []
        
        for query_id, results in search_results.items():
            if query_id not in self.ground_truth:
                continue
                
            true_match = self.ground_truth[query_id]
            
            # 找到真实匹配在结果中的位置
            rank = None
            for i, (match_id, _) in enumerate(results):
                if match_id == true_match:
                    rank = i
                    break
            
            # 计算AP
            if rank is not None:
                # AP = 1/(rank+1) 因为rank从0开始
                ap = 1.0 / (rank + 1)
                aps.append(ap)
            else:
                aps.append(0.0)  # 未找到匹配
        
        # 计算mAP
        mAP = np.mean(aps) if aps else 0.0
        return mAP
    
    def calculate_uap(self, search_results: Dict[str, List[Tuple[str, float]]], 
                      thresholds: List[float]) -> Dict[float, float]:
        """
        计算微平均精度 (μAP) 在不同阈值下
        
        参数:
            search_results: 搜索结果
            thresholds: 相似度阈值列表
            
        返回:
            dict: {threshold: uAP_value} 不同阈值下的μAP
        """
        uap_scores = {}
        
        for threshold in thresholds:
            tp = 0  # 真正例数
            fp = 0  # 假正例数
            
            for query_id, results in search_results.items():
                if query_id not in self.ground_truth:
                    continue
                    
                true_match = self.ground_truth[query_id]
                
                # 检查高于阈值的匹配
                matches_above_threshold = [(match_id, score) for match_id, score in results 
                                         if score >= threshold]
                
                found_true_match = False
                for match_id, _ in matches_above_threshold:
                    if match_id == true_match:
                        tp += 1
                        found_true_match = True
                    else:
                        fp += 1
                        
                # 如果没有找到真实匹配，但有其他匹配超过阈值，这是假正例
                if not found_true_match and matches_above_threshold:
                    fp += 1
            
            # 计算μAP = TP/(TP+FP)
            if tp + fp > 0:
                uap_scores[threshold] = tp / (tp + fp)
            else:
                uap_scores[threshold] = 0.0
                
        return uap_scores
    
    def evaluate(self, search_results: Dict[str, List[Tuple[str, float]]], 
                thresholds: List[float] = None) -> Dict[str, Any]:
        """
        全面评估拷贝检测算法性能
        
        参数:
            search_results: 搜索结果
            thresholds: 用于计算μAP的阈值列表，默认为[0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            
        返回:
            dict: 包含mAP和各阈值下μAP的结果字典
        """
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            
        mAP = self.calculate_map(search_results)
        uAP = self.calculate_uap(search_results, thresholds)
        
        return {
            'mAP': mAP,
            'uAP': uAP
        }
    
    def evaluate_by_transformation(self, search_results: Dict[str, List[Tuple[str, float]]],
                                 query_images: Dict[str, Dict[str, Any]],
                                 thresholds: List[float] = None) -> Dict[str, Dict[str, Any]]:
        """
        按变换类型评估算法性能
        
        参数:
            search_results: 搜索结果
            query_images: 查询图像字典，包含变换信息
            thresholds: 阈值列表
            
        返回:
            dict: {transformation_type: {metrics...}} 按变换类型划分的指标
        """
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            
        # 按变换类型分组查询
        transform_groups = defaultdict(dict)
        for query_id, results in search_results.items():
            if query_id in query_images:
                transform = query_images[query_id].get('transform', 'unknown')
                transform_groups[transform][query_id] = results
        
        # 对每种变换类型评估
        transform_metrics = {}
        for transform, transform_results in transform_groups.items():
            mAP = self.calculate_map(transform_results)
            uAP = self.calculate_uap(transform_results, thresholds)
            
            transform_metrics[transform] = {
                'mAP': mAP,
                'uAP': uAP
            }
            
        return transform_metrics


def compare_hash_methods(original_images: Dict[str, Dict[str, Any]], 
                        query_images: Dict[str, Dict[str, Any]], 
                        ground_truth: Dict[str, str], 
                        hash_methods: List[str] = None, 
                        hash_size: int = 8,
                        thresholds: List[float] = None,
                        use_cuda: bool = False):
    """
    比较不同哈希方法的性能
    
    参数:
        original_images: 原始图像字典
        query_images: 查询图像字典
        ground_truth: 真实匹配关系
        hash_methods: 哈希方法列表，默认为['phash', 'dhash', 'ahash']
        hash_size: 哈希大小
        thresholds: 用于μAP计算的阈值列表
        use_cuda: 是否使用CUDA加速
    """
    if hash_methods is None:
        hash_methods = ['phash', 'dhash', 'ahash']
        
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    all_results = {}
    evaluator = CopyDetectionEvaluator(ground_truth)
    
    for method in hash_methods:
        print(f"\n评估 {method} 哈希方法 (大小: {hash_size}x{hash_size})...")
        
        # 初始化检测器
        detector = HashImageCopyDetector(hash_size=hash_size, hash_method=method, use_cuda=use_cuda)
        
        # 建立索引
        detector.index_reference_images(original_images)
        
        # 执行搜索
        search_results = detector.search(query_images)
        
        # 评估性能
        results = evaluator.evaluate(search_results, thresholds)
        all_results[method] = results
        
        # 输出结果
        print(f"{method} mAP: {results['mAP']:.4f}")
        print(f"{method} μAP at different thresholds:")
        for t in sorted(results['uAP'].keys()):
            print(f"  Threshold {t:.2f}: {results['uAP'][t]:.4f}")
    
    # 找出性能最佳的方法
    best_method = max(hash_methods, key=lambda m: all_results[m]['mAP'])
    print(f"\n最佳哈希方法: {best_method} (mAP: {all_results[best_method]['mAP']:.4f})")
    
    return all_results


def main():
    """主函数：运行拷贝检测实验"""
    # 设置数据集路径
    base_dir = './data'  # 修改为您的数据集实际路径
    
    # 设置参数
    hash_size = 4
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    use_cuda = True  # 是否使用CUDA加速
    
    # 加载Copydays数据集
    print("加载Copydays数据集...")
    original_images, query_images, ground_truth = load_copydays_dataset(
        base_dir, preprocess=True, target_size=(224, 224))
    
    print(f"数据集加载完成: {len(original_images)}张原始图像, {len(query_images)}张查询图像")
    print(f"Ground truth关系: {len(ground_truth)}个")
    
    # 比较不同哈希方法
    hash_methods = ['phash', 'dhash', 'ahash']
    hash_results = compare_hash_methods(
        original_images, query_images, ground_truth, 
        hash_methods, hash_size, thresholds, use_cuda
    )
    
    # 详细分析最佳方法
    best_method = max(hash_methods, key=lambda m: hash_results[m]['mAP'])
    print(f"\n最佳哈希方法: {best_method} (mAP: {hash_results[best_method]['mAP']:.4f})")
    
    # 使用最佳方法按变换类型进行评估
    detector = HashImageCopyDetector(hash_size=hash_size, hash_method=best_method, use_cuda=use_cuda)
    detector.index_reference_images(original_images)
    search_results = detector.search(query_images)
    
    evaluator = CopyDetectionEvaluator(ground_truth)
    transform_results = evaluator.evaluate_by_transformation(
        search_results, query_images, thresholds
    )
    
    # 输出按变换类型的性能
    print("\n按变换类型的性能评估:")
    for transform, metrics in transform_results.items():
        print(f"{transform} - mAP: {metrics['mAP']:.4f}")
        print(f"{transform} - μAP at different thresholds:")
        for t in sorted(metrics['uAP'].keys()):
            print(f"  Threshold {t:.2f}: {metrics['uAP'][t]:.4f}")


if __name__ == "__main__":
    main()

