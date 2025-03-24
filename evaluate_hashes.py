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
from evaluate import evaluate_hash

def main():
    parser = argparse.ArgumentParser(description='评估图像哈希算法')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--hash_size', type=int, default=8, help='哈希大小')
    parser.add_argument('--output_dir', type=str, default='./results', help='结果输出目录')
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"已创建结果输出目录: {args.output_dir}")
    
    # 加载数据集
    print("加载 Copydays 数据集...")
    data = load_data('copydays', args.data_dir)
    
    # 定义要评估的哈希算法
    hash_algorithms = {
        'Average Hash (aHash)': average_hash,
        'Perceptual Hash (pHash)': perceptual_hash,
        'Difference Hash (dHash)': difference_hash,
        'Wavelet Hash (wHash)': wavelet_hash,
        'Color Hash (cHash)': color_hash,
        'Marr-Hildreth Hash (mhHash)': marr_hildreth_hash
    }
    
    # 评估每个哈希算法
    results = {}
    for name, hash_func in hash_algorithms.items():
        print(f"\n评估 {name}...")
        mAP, μAP = evaluate_hash(hash_func, data, args.hash_size)
        results[name] = {'mAP': mAP, 'μAP': μAP}
        print(f"{name} - mAP: {mAP:.4f}, μAP: {μAP:.4f}")
    
    # 打印汇总结果
    print("\n===== 结果汇总 =====")
    print(f"{'算法':<25} {'mAP':<10} {'μAP':<10}")
    print("-" * 45)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['mAP']:<10.4f} {metrics['μAP']:<10.4f}")
    
    # 将结果保存到CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(args.output_dir, f"hash_evaluation_results_{timestamp}.csv")
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['算法', 'mAP', 'μAP', '哈希大小', '时间戳']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow({
                '算法': name,
                'mAP': f"{metrics['mAP']:.6f}",
                'μAP': f"{metrics['μAP']:.6f}",
                '哈希大小': args.hash_size,
                '时间戳': timestamp
            })
    
    print(f"\n结果已保存到: {csv_filename}")
    
    # 创建一个汇总CSV文件，用于追加所有评估结果
    summary_csv = os.path.join(args.output_dir, "hash_evaluation_summary.csv")
    file_exists = os.path.isfile(summary_csv)
    
    with open(summary_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['算法', 'mAP', 'μAP', '哈希大小', '时间戳']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for name, metrics in results.items():
            writer.writerow({
                '算法': name,
                'mAP': f"{metrics['mAP']:.6f}",
                'μAP': f"{metrics['μAP']:.6f}",
                '哈希大小': args.hash_size,
                '时间戳': timestamp
            })
    
    print(f"结果已追加到汇总文件: {summary_csv}")

if __name__ == '__main__':
    main()