import os
import argparse
import numpy as np
import csv
from datetime import datetime
from utils import load_data
from image_hash import (
    average_hash, perceptual_hash, difference_hash, 
    wavelet_hash, color_hash, marr_hildreth_hash
)
from resnet_detection import resnet_hash, resnet_deep, compute_resnet_deep_distance
from multiscale_detection import multiscale_hash, multiscale_deep, compute_multiscale_distance
from vit_detection import vit_hash, vit_deep, compute_vit_distance
# 添加多尺度ViT哈希算法导入
from multiscale_vit_detection import multiscale_vit_hash, multiscale_vit_deep, compute_multiscale_vit_distance
# 添加显著性增强ResNet哈希算法导入
from salient_resnet_detection import salient_resnet_hash, salient_resnet_deep, compute_salient_resnet_deep_distance

from evaluate import evaluate_hash

def main():
    parser = argparse.ArgumentParser(description='评估图像哈希算法')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--hash_size', type=int, default=8, help='哈希大小')
    parser.add_argument('--output_dir', type=str, default='./results', help='结果输出目录')
    # 添加数据集选择参数
    parser.add_argument('--dataset', type=str, default='copydays', choices=['copydays', 'scid','copy2'], 
                        help='要评估的数据集，可选: copydays, scid,copy2')
    # 修改算法帮助文本，添加新的算法选项
    parser.add_argument('--algorithms', type=str, nargs='+', 
                    default=['all'], 
                    help='要评估的算法，可选: aHash, pHash, dHash, wHash, cHash, mhHash, resnet-hash, resnet-deep, multiscale-hash, multiscale-deep, vit-hash, vit-deep, multiscale-vit-hash, multiscale-vit-deep, salient-resnet-hash, salient-resnet-deep, all')
    parser.add_argument('--scales', type=float, nargs='+', 
                        default=[1.0, 0.75, 0.5], 
                        help='多尺度特征提取的缩放比例')
    # 添加显著性算法选择参数
    parser.add_argument('--saliency_algorithm', type=str, default='SR', choices=['SR', 'FT'],
                        help='显著性算法选择 (SR: Spectral Residual, FT: Frequency-Tuned)')
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"已创建结果输出目录: {args.output_dir}")
    
    # 加载数据集
    print(f"加载 {args.dataset.capitalize()} 数据集...")
    data = load_data(args.dataset, args.data_dir)
    
    # 在all_hash_algorithms字典中添加新的算法
    all_hash_algorithms = {
        'aHash': {'name': 'Average Hash (aHash)', 'func': average_hash, 'is_deep': False, 'distance_func': None},
        'pHash': {'name': 'Perceptual Hash (pHash)', 'func': perceptual_hash, 'is_deep': False, 'distance_func': None},
        'dHash': {'name': 'Difference Hash (dHash)', 'func': difference_hash, 'is_deep': False, 'distance_func': None},
        'wHash': {'name': 'Wavelet Hash (wHash)', 'func': wavelet_hash, 'is_deep': False, 'distance_func': None},
        'cHash': {'name': 'Color Hash (cHash)', 'func': color_hash, 'is_deep': False, 'distance_func': None},
        'mhHash': {'name': 'Marr-Hildreth Hash (mhHash)', 'func': marr_hildreth_hash, 'is_deep': False, 'distance_func': None},
        'resnet-hash': {'name': 'ResNet50 Hash', 'func': resnet_hash, 'is_deep': False, 'distance_func': None},
        'resnet-deep': {'name': 'ResNet50 Deep Features', 'func': resnet_deep, 'is_deep': True, 'distance_func': compute_resnet_deep_distance},
        'multiscale-hash': {'name': 'Multiscale ResNet50 Hash', 'func': multiscale_hash, 'is_deep': False, 'distance_func': None},
        'multiscale-deep': {'name': 'Multiscale ResNet50 Deep Features', 'func': multiscale_deep, 'is_deep': True, 'distance_func': compute_multiscale_distance},
        'vit-hash': {'name': 'ViT Hash', 'func': vit_hash, 'is_deep': False, 'distance_func': None},
        'vit-deep': {'name': 'ViT Deep Features', 'func': vit_deep, 'is_deep': True, 'distance_func': compute_vit_distance},
        # 添加多尺度ViT哈希算法
        'multiscale-vit-hash': {'name': '多尺度ViT哈希', 'func': multiscale_vit_hash, 'is_deep': False, 'distance_func': None},
        'multiscale-vit-deep': {'name': '多尺度ViT深度特征', 'func': multiscale_vit_deep, 'is_deep': True, 'distance_func': compute_multiscale_vit_distance},
        # 添加显著性增强ResNet哈希算法
        'salient-resnet-hash': {'name': '显著性增强ResNet哈希', 'func': salient_resnet_hash, 'is_deep': False, 'distance_func': None},
        'salient-resnet-deep': {'name': '显著性增强ResNet深度特征', 'func': salient_resnet_deep, 'is_deep': True, 'distance_func': compute_salient_resnet_deep_distance},
        }
    
    # 选择要评估的算法
    hash_algorithms = {}
    if 'all' in args.algorithms:
        hash_algorithms = {k: v for k, v in all_hash_algorithms.items()}
    else:
        for algo in args.algorithms:
            if algo in all_hash_algorithms:
                hash_algorithms[algo] = all_hash_algorithms[algo]
            else:
                print(f"警告: 未知算法 '{algo}'，将被跳过")
    
    # 评估每个哈希算法
    results = {}
    for key, algo_info in hash_algorithms.items():
        name = algo_info['name']
        hash_func = algo_info['func']
        is_deep = algo_info['is_deep']
        distance_func = algo_info['distance_func']
        
        print(f"\n评估 {name}...")
        
        # 为显著性增强ResNet算法添加特殊参数处理
        if 'salient-resnet' in key:
            # 使用命令行参数指定的显著性算法
            kwargs = {'use_saliency': True, 'saliency_algorithm': args.saliency_algorithm}
            # 将kwargs传递给hash_func而不是evaluate_hash
            mAP, μAP = evaluate_hash(
                lambda img, hash_size=args.hash_size: hash_func(img, hash_size, **kwargs),
                data, args.hash_size, distance_func, is_deep
            )
        else:
            mAP, μAP = evaluate_hash(hash_func, data, args.hash_size, distance_func, is_deep)
            
        results[name] = {'mAP': mAP, 'μAP': μAP}
        print(f"{name} - mAP: {mAP:.4f}, μAP: {μAP:.4f}")
    
    # 打印汇总结果
    print("\n===== 结果汇总 =====")
    print(f"数据集: {args.dataset.capitalize()}")
    print(f"哈希大小: {args.hash_size}")
    if any('salient-resnet' in key for key in hash_algorithms.keys()):
        print(f"显著性算法: {args.saliency_algorithm}")
    print(f"{'算法':<40} {'mAP':<10} {'μAP':<10}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<40} {metrics['mAP']:<10.4f} {metrics['μAP']:<10.4f}")
    
    # 将结果保存到CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(args.output_dir, f"{args.dataset}_hash_evaluation_results_{timestamp}.csv")
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['算法', 'mAP', 'μAP', '数据集', '哈希大小', '时间戳', '缩放比例', '显著性算法']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for name, metrics in results.items():
            row_data = {
                '算法': name,
                'mAP': f"{metrics['mAP']:.6f}",
                'μAP': f"{metrics['μAP']:.6f}",
                '数据集': args.dataset,
                '哈希大小': args.hash_size,
                '时间戳': timestamp,
                '缩放比例': str(args.scales) if 'multiscale' in name.lower() else 'N/A',
                '显著性算法': args.saliency_algorithm if 'salient' in name.lower() else 'N/A'
            }
            writer.writerow(row_data)
    
    print(f"\n结果已保存到: {csv_filename}")
    
    # 创建一个汇总CSV文件，用于追加所有评估结果
    summary_csv = os.path.join(args.output_dir, f"{args.dataset}_hash_evaluation_summary.csv")
    file_exists = os.path.isfile(summary_csv)
    
    with open(summary_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['算法', 'mAP', 'μAP', '数据集', '哈希大小', '时间戳', '缩放比例', '显著性算法']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for name, metrics in results.items():
            row_data = {
                '算法': name,
                'mAP': f"{metrics['mAP']:.6f}",
                'μAP': f"{metrics['μAP']:.6f}",
                '数据集': args.dataset,
                '哈希大小': args.hash_size,
                '时间戳': timestamp,
                '缩放比例': str(args.scales) if 'multiscale' in name.lower() else 'N/A',
                '显著性算法': args.saliency_algorithm if 'salient' in name.lower() else 'N/A'
            }
            writer.writerow(row_data)
    
    print(f"结果已追加到汇总文件: {summary_csv}")

if __name__ == '__main__':
    main()