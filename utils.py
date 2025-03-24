import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

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
