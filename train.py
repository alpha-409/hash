from typing import Dict, List, Tuple, Any
from model import HashImageCopyDetector
from utils import CopyDetectionEvaluator

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


def compare_hash_sizes(original_images, query_images, ground_truth, 
                      hash_method='ahash', 
                      hash_sizes=[4, 8, 16, 32], 
                      thresholds=None,
                      use_cuda=False):
    """
    比较不同哈希大小的性能
    
    参数:
        original_images: 原始图像字典
        query_images: 查询图像字典
        ground_truth: 真实匹配关系
        hash_method: 使用的哈希方法
        hash_sizes: 哈希大小列表
        thresholds: 用于μAP计算的阈值列表
        use_cuda: 是否使用CUDA加速
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    all_results = {}
    evaluator = CopyDetectionEvaluator(ground_truth)
    
    print(f"\n比较{hash_method}算法在不同哈希大小下的性能...")
    
    for size in hash_sizes:
        print(f"\n评估 {hash_method} 哈希方法 (大小: {size}x{size})...")
        
        # 初始化检测器
        detector = HashImageCopyDetector(hash_size=size, hash_method=hash_method, use_cuda=use_cuda)
        
        # 建立索引
        detector.index_reference_images(original_images)
        
        # 执行搜索
        search_results = detector.search(query_images)
        
        # 评估性能
        results = evaluator.evaluate(search_results, thresholds)
        all_results[size] = results
        
        # 输出结果
        print(f"{hash_method}-{size}x{size} mAP: {results['mAP']:.4f}")
        print(f"{hash_method}-{size}x{size} μAP at Threshold 0.90: {results['uAP'][0.9]:.4f}")
    
    # 找出性能最佳的哈希大小
    best_size = max(hash_sizes, key=lambda s: all_results[s]['mAP'])
    print(f"\n最佳哈希大小: {best_size}x{best_size} (mAP: {all_results[best_size]['mAP']:.4f})")
    
    return all_results


def evaluate_best_configuration(original_images, query_images, ground_truth, 
                              hash_method, hash_size, thresholds=None, use_cuda=False):
    """
    评估特定哈希配置的性能并按变换类型细分
    
    参数:
        original_images: 原始图像字典
        query_images: 查询图像字典
        ground_truth: 真实匹配关系  
        hash_method: 哈希方法
        hash_size: 哈希大小
        thresholds: 阈值列表
        use_cuda: 是否使用CUDA
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        
    print(f"\n详细评估 {hash_method}-{hash_size}x{hash_size} 配置...")
    
    # 初始化检测器和评估器
    detector = HashImageCopyDetector(hash_size=hash_size, hash_method=hash_method, use_cuda=use_cuda)
    evaluator = CopyDetectionEvaluator(ground_truth)
    
    # 建立索引
    detector.index_reference_images(original_images)
    
    # 执行搜索
    search_results = detector.search(query_images)
    
    # 按变换类型评估
    transform_results = evaluator.evaluate_by_transformation(
        search_results, query_images, thresholds
    )
    
    # 输出按变换类型的性能
    print("\n按变换类型的性能评估:")
    for transform, metrics in transform_results.items():
        print(f"{transform} - mAP: {metrics['mAP']:.4f}")
        best_threshold = max(metrics['uAP'].items(), key=lambda x: x[1])
        print(f"{transform} - 最佳μAP: {best_threshold[1]:.4f} (阈值={best_threshold[0]:.2f})")
        print(f"{transform} - μAP at different thresholds:")
        for t in sorted(metrics['uAP'].keys()):
            print(f"  Threshold {t:.2f}: {metrics['uAP'][t]:.4f}")
            
    return transform_results
