import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import average_precision_score

def compute_hamming_distance(hash1, hash2):
    """
    计算两个哈希值之间的汉明距离
    """
    return np.sum(hash1 != hash2)

def compute_distances(query_hashes, db_hashes):
    """
    计算查询哈希与数据库哈希之间的距离
    """
    distances = []
    for q_hash in query_hashes:
        q_distances = []
        for db_hash in db_hashes:
            dist = compute_hamming_distance(q_hash, db_hash)
            q_distances.append(dist)
        distances.append(q_distances)
    return np.array(distances)

def evaluate_hash(hash_func, data, hash_size=8):
    """
    评估哈希算法在数据集上的性能
    
    参数:
        hash_func: 哈希函数
        data: 数据集字典
        hash_size: 哈希大小
        
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
    
    # 计算查询图像的哈希
    query_hashes = []
    for img in tqdm(query_images, desc="计算查询哈希"):
        hash_value = hash_func(img, hash_size)
        query_hashes.append(hash_value)
    
    # 计算数据库图像的哈希
    db_hashes = []
    for img in tqdm(db_images, desc="计算数据库哈希"):
        hash_value = hash_func(img, hash_size)
        db_hashes.append(hash_value)
    
    # 计算距离
    distances = compute_distances(query_hashes, db_hashes)
    
    # 计算AP和mAP
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