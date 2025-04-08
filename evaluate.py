import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import average_precision_score
from functools import lru_cache

def compute_hamming_distance(hash1, hash2):
    """
    计算两个哈希值之间的汉明距离
    """
    return np.sum(hash1 != hash2)

def compute_distances(query_hashes, db_hashes, distance_func=None):
    """
    计算查询哈希与数据库哈希之间的距离
    
    参数:
        query_hashes: 查询哈希列表
        db_hashes: 数据库哈希列表
        distance_func: 距离计算函数，默认为None（使用汉明距离）
    """
    # 将哈希值转换为numpy数组以加速计算
    query_hashes = np.array(query_hashes)
    db_hashes = np.array(db_hashes)
    
    # 使用向量化操作计算距离
    distances = np.zeros((len(query_hashes), len(db_hashes)))
    
    # 批量计算距离以提高效率
    batch_size = 50  # 可以根据内存大小调整
    for i in range(0, len(query_hashes), batch_size):
        end_i = min(i + batch_size, len(query_hashes))
        q_batch = query_hashes[i:end_i]
        
        for j in range(0, len(db_hashes), batch_size):
            end_j = min(j + batch_size, len(db_hashes))
            db_batch = db_hashes[j:end_j]
            
            # 计算当前批次的距离矩阵
            for k, q in enumerate(q_batch):
                for l, db in enumerate(db_batch):
                    if distance_func is None:
                        # 默认使用汉明距离
                        distances[i+k, j+l] = compute_hamming_distance(q, db)
                    else:
                        # 使用自定义距离函数
                        distances[i+k, j+l] = distance_func(q, db)
    
    return distances

def evaluate_hash(hash_func, data, hash_size=8, distance_func=None, is_deep_feature=False):
    """
    评估哈希算法在数据集上的性能
    
    参数:
        hash_func: 哈希函数
        data: 数据集字典
        hash_size: 哈希大小
        distance_func: 距离计算函数，默认为None（使用汉明距离）
        is_deep_feature: 是否是深度特征（而非二进制哈希）
        
    返回:
        mAP和μAP
    """
    query_images = data['query_images']
    db_images = data['db_images']
    positives = data['positives']
    
    # 检查数据集是否为空
    if len(query_images) == 0 or len(db_images) == 0:
        print("警告: 数据集为空!")
        return 0.0, 0.0
    
    # 使用GPU加速（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(query_images, torch.Tensor):
        query_images = query_images.to(device)
    if isinstance(db_images, torch.Tensor):
        db_images = db_images.to(device)
    
    # 计算查询图像的哈希或特征
    query_features = []
    for img in tqdm(query_images, desc="计算查询特征"):
        if is_deep_feature:
            feature = hash_func(img)
        else:
            feature = hash_func(img, hash_size)
        query_features.append(feature)
    
    # 计算数据库图像的哈希或特征
    db_features = []
    for img in tqdm(db_images, desc="计算数据库特征"):
        if is_deep_feature:
            feature = hash_func(img)
        else:
            feature = hash_func(img, hash_size)
        db_features.append(feature)
    
    # 计算距离
    print("计算特征距离...")
    distances = compute_distances(query_features, db_features, distance_func)
    
    # 计算AP和mAP
    print("计算评估指标...")
    aps = []
    for q_idx in range(len(query_images)):
        # 获取真实正样本
        true_positives = [db_idx for (qidx, db_idx) in positives if qidx == q_idx]
        
        if not true_positives:
            continue
        
        # 创建真实标签
        y_true = np.zeros(len(db_images))
        y_true[true_positives] = 1
        
        # 距离越小，相似度越高，所以使用负距离作为分数
        y_score = -distances[q_idx]
        
        # 计算AP
        ap = average_precision_score(y_true, y_score)
        aps.append(ap)
    
    # 计算mAP
    mAP = np.mean(aps) if aps else 0.0
    
    # 计算μAP (micro-AP)
    all_y_true = []
    all_y_score = []
    
    for q_idx in range(len(query_images)):
        true_positives = [db_idx for (qidx, db_idx) in positives if qidx == q_idx]
        
        if not true_positives:
            continue
        
        y_true = np.zeros(len(db_images))
        y_true[true_positives] = 1
        
        y_score = -distances[q_idx]
        
        all_y_true.extend(y_true)
        all_y_score.extend(y_score)
    
    # 检查是否有足够的数据计算μAP
    if not all_y_true or not all_y_score:
        print("警告: 没有足够的数据计算μAP!")
        μAP = 0.0
    else:
        μAP = average_precision_score(all_y_true, all_y_score)
    
    return mAP, μAP


def evaluate_model(model, dataloader, device):
    model.eval()
    all_hashes = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            binary_hash, _, _ = model(imgs)
            all_hashes.append(binary_hash.cpu().numpy())
            all_labels.append(labels.numpy())
    
    hashes = np.concatenate(all_hashes)
    labels = np.concatenate(all_labels)
    
    # 计算mAP和μAP
    n_samples = len(hashes)
    aps = []
    all_y_true = []
    all_y_score = []
    
    for i in range(n_samples):
        # 计算当前样本与其他样本的汉明距离
        distances = np.sum(hashes != hashes[i], axis=1)
        # 相似性标签 (1表示正样本，0表示负样本)
        y_true = (labels == labels[i]).astype(int)
        # 排除自身
        y_true[i] = 0
        distances[i] = np.inf
        
        # 按距离排序
        sorted_indices = np.argsort(distances)
        y_true_sorted = y_true[sorted_indices]
        y_score_sorted = -distances[sorted_indices]
        
        # 计算AP
        ap = average_precision_score(y_true_sorted, y_score_sorted)
        aps.append(ap)
        
        # 收集μAP计算所需数据
        all_y_true.extend(y_true)
        all_y_score.extend(-distances)
    
    # 计算mAP
    mAP = np.mean(aps) if aps else 0.0
    
    # 计算μAP
    if not all_y_true or not all_y_score:
        print("警告: 没有足够的数据计算μAP!")
        μAP = 0.0
    else:
        μAP = average_precision_score(all_y_true, all_y_score)
    
    print(f"mAP: {mAP:.4f}, μAP: {μAP:.4f}")
    return mAP, μAP