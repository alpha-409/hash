"""
运行图像哈希拷贝检测评估
"""

import os
import json
from data import load_copydays_dataset
from image_retrieval import ImageRetrieval, evaluate_retrieval
import argparse
from datetime import datetime

def evaluate_hash_method(original_images, query_images, ground_truth,
                        hash_method, hash_size, device, threshold):
    """评估单个哈希方法"""
    print(f"\n评估 {hash_method} (hash_size={hash_size}, device={device}):")
    
    # 初始化检索系统
    retrieval = ImageRetrieval(
        hash_method=hash_method,
        hash_size=hash_size,
        device=device
    )
    
    # 建立索引
    retrieval.index_images(original_images)
    
    # 执行检索
    search_results = retrieval.search(query_images, threshold)
    
    # 计算评估指标
    metrics = evaluate_retrieval(search_results, ground_truth, threshold)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='评估图像哈希拷贝检测性能')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='计算设备 (cpu 或 cuda)')
    parser.add_argument('--hash-size', type=int, default=8,
                        help='哈希大小 (默认: 8)')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='相似度阈值 (默认: 0.8)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批处理大小 (默认: 32)')
    args = parser.parse_args()
    
    # 加载数据集
    print("加载 Copydays 数据集...")
    base_dir = os.getcwd()  # 当前目录
    original_images, query_images, ground_truth = load_copydays_dataset(
        base_dir,
        preprocess=False  # 在哈希算法中进行预处理
    )
    
    print(f"数据集统计:")
    print(f"原始图像数量: {len(original_images)}")
    print(f"查询图像数量: {len(query_images)}")
    print(f"Ground truth 映射数量: {len(ground_truth)}")
    
    # 评估所有哈希方法
    hash_methods = ['ahash', 'dhash', 'phash']
    results = {}
    
    for method in hash_methods:
        metrics = evaluate_hash_method(
            original_images,
            query_images,
            ground_truth,
            method,
            args.hash_size,
            args.device,
            args.threshold
        )
        results[method] = metrics
        
        # 打印结果
        print(f"\n{method} 结果:")
        print(f"mAP: {metrics['mAP']:.4f}")
        print(f"μAP: {metrics['μAP']:.4f}")
        for k in [1, 5, 10]:
            print(f"P@{k}: {metrics[f'P@{k}']:.4f}, R@{k}: {metrics[f'R@{k}']:.4f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = os.path.join(
        results_dir, 
        f"hash_evaluation_{args.hash_size}_{args.device}_{timestamp}.json"
    )
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'parameters': vars(args),
            'results': results
        }, f, indent=2, ensure_ascii=False)
        
    print(f"\n结果已保存至: {save_path}")

if __name__ == "__main__":
    main()
