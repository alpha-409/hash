import os
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from musc import MultiScaleHashGenerator
from utils import load_data
# 修改导入语句，使用正确的函数名
from evaluate import evaluate_hash  # 导入evaluate_hash函数
from evaluate import compute_micro_average_precision as compute_micro_ap

def parse_args():
    parser = argparse.ArgumentParser(description='评估MultiScaleHashGenerator在Copydays数据集上的性能')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据集目录')
    parser.add_argument('--hash_dim', type=int, default=64, help='哈希码维度')
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'attention', 'gated'], help='特征融合方法')
    parser.add_argument('--decompose', type=str, default='pca', choices=['pca', 'grp', 'itq'], help='矩阵分解方法')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载线程数')
    return parser.parse_args()

def compute_hash_distance(hash1, hash2):
    """计算两个哈希码之间的汉明距离"""
    return np.sum(hash1 != hash2)

def evaluate_musc_hash(args):
    print(f"\n{'='*50}")
    print(f"开始评估 MultiScaleHashGenerator")
    print(f"参数: 哈希维度={args.hash_dim}, 融合方法={args.fusion}, 分解方法={args.decompose}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    # 加载数据集
    print("加载Copydays数据集...")
    data = load_data('copydays', data_dir=args.data_dir, num_workers=args.num_workers)
    
    # 初始化哈希生成器
    print("\n初始化MultiScaleHashGenerator...")
    hash_generator = MultiScaleHashGenerator(
        hash_dim=args.hash_dim,
        fusion_method=args.fusion,
        decompose_method=args.decompose
    )
    
    # 准备设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hash_generator.extractor.to(device)
    print(f"使用设备: {device}")
    
    # 提取特征并训练分解器
    print("\n提取训练特征并训练分解器...")
    
    # 使用数据库图像的一部分作为训练集
    train_size = min(1000, len(data['db_images']))
    train_indices = np.random.choice(len(data['db_images']), train_size, replace=False)
    
    # 批量提取特征
    train_features = []
    batch_size = args.batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(train_indices), batch_size), desc="提取训练特征"):
            batch_indices = train_indices[i:i+batch_size]
            batch_images = [data['db_images'][idx] for idx in batch_indices]
            
            # 确保批次是张量
            if isinstance(batch_images[0], torch.Tensor):
                batch_tensor = torch.stack(batch_images).to(device)
            else:
                # 如果是PIL图像，需要转换
                batch_tensor = torch.stack([transforms.ToTensor()(img) for img in batch_images]).to(device)
            
            # 提取特征
            batch_features = hash_generator.extract_features(batch_tensor)
            train_features.append(batch_features.cpu().numpy())
    
    # 合并所有特征
    train_features = np.vstack(train_features)
    print(f"训练特征形状: {train_features.shape}")
    
    # 训练分解器
    print("训练分解器...")
    hash_generator.train_decomposer(train_features)
    
    # 生成查询图像的哈希码
    print("\n为查询图像生成哈希码...")
    query_hashes = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data['query_images']), batch_size), desc="处理查询图像"):
            batch_images = data['query_images'][i:i+batch_size]
            
            # 确保批次是张量
            if isinstance(batch_images, list):
                if isinstance(batch_images[0], torch.Tensor):
                    batch_tensor = torch.stack(batch_images).to(device)
                else:
                    # 如果是PIL图像，需要转换
                    batch_tensor = torch.stack([transforms.ToTensor()(img) for img in batch_images]).to(device)
            else:
                # 如果已经是批次张量
                batch_tensor = batch_images[i:i+batch_size].to(device)
            
            # 提取特征
            batch_features = hash_generator.extract_features(batch_tensor)
            
            # 生成哈希
            for j in range(len(batch_features)):
                feature = batch_features[j].cpu().numpy()
                hash_code = hash_generator.generate_hash(feature)
                query_hashes.append(hash_code)
    
    # 生成数据库图像的哈希码
    print("\n为数据库图像生成哈希码...")
    db_hashes = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data['db_images']), batch_size), desc="处理数据库图像"):
            batch_images = data['db_images'][i:i+batch_size]
            
            # 确保批次是张量
            if isinstance(batch_images, list):
                if isinstance(batch_images[0], torch.Tensor):
                    batch_tensor = torch.stack(batch_images).to(device)
                else:
                    # 如果是PIL图像，需要转换
                    batch_tensor = torch.stack([transforms.ToTensor()(img) for img in batch_images]).to(device)
            else:
                # 如果已经是批次张量
                batch_tensor = batch_images[i:i+batch_size].to(device)
            
            # 提取特征
            batch_features = hash_generator.extract_features(batch_tensor)
            
            # 生成哈希
            for j in range(len(batch_features)):
                feature = batch_features[j].cpu().numpy()
                hash_code = hash_generator.generate_hash(feature)
                db_hashes.append(hash_code)
    
    # 计算距离矩阵
    print("\n计算距离矩阵...")
    distances = np.zeros((len(query_hashes), len(db_hashes)))
    
    for i in tqdm(range(len(query_hashes)), desc="计算查询-数据库距离"):
        for j in range(len(db_hashes)):
            distances[i, j] = compute_hash_distance(query_hashes[i], db_hashes[j])
    
    # 计算评价指标
    print("\n计算评价指标...")
    positives = data['positives']
    # 使用evaluate_hash函数计算性能指标
    mAP, μAP = evaluate_hash(lambda img, _: hash_generator.generate_hash(hash_generator.extract_features(img).cpu().numpy()), 
                             data, args.hash_dim, compute_hash_distance, is_deep=False)
    mAP = compute_map(distances, positives)
    μAP = compute_micro_ap(distances, positives)
    
    # 输出结果
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"评估完成! 总耗时: {total_time:.2f}秒")
    print(f"MultiScaleHashGenerator ({args.fusion}+{args.decompose}) 性能:")
    print(f"- 哈希维度: {args.hash_dim}")
    print(f"- mAP: {mAP:.4f}")
    print(f"- μAP: {μAP:.4f}")
    print(f"{'='*50}")
    
    return mAP, μAP

if __name__ == "__main__":
    args = parse_args()
    evaluate_musc_hash(args)