import os
import torch
from data import load_copydays_dataset
from model import HashImageCopyDetector
from utils import CopyDetectionEvaluator
from train import compare_hash_methods, compare_hash_sizes, evaluate_best_configuration

def main():
    """主函数：运行拷贝检测实验"""
    # 设置数据集路径
    base_dir = './data'  # 修改为您的数据集实际路径
    
    # 设置参数
    hash_size = 8  # 根据实验结果修改为最佳大小，这里使用8x8
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    use_cuda = torch.cuda.is_available()  # 自动检测是否使用CUDA加速
    
    # 加载Copydays数据集
    print("加载Copydays数据集...")
    original_images, query_images, ground_truth = load_copydays_dataset(
        base_dir, preprocess=True, target_size=(224, 224))
    
    print(f"数据集加载完成: {len(original_images)}张原始图像, {len(query_images)}张查询图像")
    print(f"Ground truth关系: {len(ground_truth)}个")
    
    # 运行模式选择
    run_mode = input("选择运行模式 (1: 比较哈希方法, 2: 比较哈希大小, 3: 评估最佳配置, 4: 全部运行): ")
    
    if run_mode == '1' or run_mode == '4':
        # 比较不同哈希方法 (phash, dhash, ahash)
        hash_methods = ['phash', 'dhash', 'ahash']
        hash_results = compare_hash_methods(
            original_images, query_images, ground_truth, 
            hash_methods, hash_size, thresholds, use_cuda
        )
        best_method = max(hash_methods, key=lambda m: hash_results[m]['mAP'])
    else:
        # 默认使用ahash作为最佳方法（根据以前的实验结果）
        best_method = 'ahash'
    
    if run_mode == '2' or run_mode == '4':
        # 比较不同哈希大小 (4x4, 8x8, 16x16, 32x32)
        hash_sizes = [4, 8, 16, 32]
        size_results = compare_hash_sizes(
            original_images, query_images, ground_truth,
            hash_method=best_method, hash_sizes=hash_sizes,
            thresholds=thresholds, use_cuda=use_cuda
        )
        best_size = max(hash_sizes, key=lambda s: size_results[s]['mAP'])
    else:
        # 默认使用8x8作为最佳大小（根据以前的实验结果）
        best_size = 8
        
    if run_mode == '3' or run_mode == '4':
        # 评估最佳配置并按变换类型细分
        evaluate_best_configuration(
            original_images, query_images, ground_truth,
            hash_method=best_method, hash_size=best_size,
            thresholds=thresholds, use_cuda=use_cuda
        )
    
    print(f"\n最佳配置总结: {best_method}-{best_size}x{best_size}")
    print("实验完成！")


if __name__ == "__main__":
    main()
