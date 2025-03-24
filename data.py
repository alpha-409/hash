import os
import pickle
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt

def load_original_images(original_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    加载Copydays数据集中的原始图像
    
    参数:
        original_dir (str): 包含原始图像的目录路径
    
    返回:
        dict: 键为图像ID (通常是文件名)，值为图像数据的字典
             格式: {image_id: {"path": 图像路径, "data": 图像数据}}
    """
    images = {}
    for filename in os.listdir(original_dir):
        if filename.lower().endswith('.jpg'):
            image_id = os.path.splitext(filename)[0]
            image_path = os.path.join(original_dir, filename)
            images[image_id] = {
                "path": image_path,
                "data": None  # 延迟加载，实际使用时才读取图像数据
            }
    return images

def load_query_images(query_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    加载Copydays数据集中的查询图像(变换后的图像)
    
    参数:
        query_dir (str): 包含查询图像的目录路径
    
    返回:
        dict: 键为图像ID，值为包含图像数据和变换类型的字典
             格式: {image_id: {"path": 图像路径, "data": 图像数据, 
                             "transform": 变换类型}}
    """
    images = {}
    transform_prefixes = {
        "crops": "裁剪",
        "jpegqual": "JPEG压缩"
    }
    
    for filename in os.listdir(query_dir):
        if not filename.lower().endswith('.jpg'):
            continue
            
        image_id = os.path.splitext(filename)[0]
        image_path = os.path.join(query_dir, filename)
        
        # 解析变换类型
        transform_type = "original"
        for prefix, desc in transform_prefixes.items():
            if prefix in filename:
                transform_type = desc
                break
                
        images[image_id] = {
            "path": image_path,
            "data": None,  # 延迟加载
            "transform": transform_type
        }
    return images

def get_ground_truth_mapping(pkl_path: str) -> Dict[str, str]:
    """
    加载并创建查询图像到原始图像的映射关系（真实匹配）
    
    参数:
        pkl_path (str): gnd_copydays.pkl文件的路径
    
    返回:
        dict: 查询图像ID到对应原始图像ID的映射
              格式: {query_image_id: original_image_id}
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    ground_truth = {}
    query_list = data['qimlist']
    image_list = data['imlist']
    gnd = data['gnd']
    
    for query_idx, query_id in enumerate(query_list):
        matches = gnd[query_idx]
        # 合并所有匹配类型
        all_matches = []
        for match_type in ['strong', 'crops', 'jpegqual']:
            if match_type in matches:
                all_matches.extend(matches[match_type])
        
        # 对每个查询图像，记录其对应的所有原始图像
        for match_idx in all_matches:
            if match_idx < len(image_list):
                ground_truth[query_id] = image_list[match_idx]
                
    return ground_truth

def get_transformation_groups(query_images: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    根据变换类型对查询图像进行分组
    
    参数:
        query_images (dict): 由load_query_images函数返回的查询图像字典
    
    返回:
        dict: 各种变换类型对应的查询图像ID列表
              格式: {transformation_type: [query_image_id1, query_image_id2, ...]}
    """
    groups = {}
    for image_id, image_info in query_images.items():
        transform = image_info['transform']
        if transform not in groups:
            groups[transform] = []
        groups[transform].append(image_id)
    return groups

def preprocess_image(image_path: str, target_size: Optional[Tuple[int, int]] = None, 
                    grayscale: bool = False) -> np.ndarray:
    """
    对图像进行预处理，如调整大小、转为灰度等
    
    参数:
        image_path: 图像文件路径
        target_size (tuple, optional): 目标图像尺寸，如(width, height)
        grayscale (bool): 是否转换为灰度图像
    
    返回:
        处理后的图像数据(NumPy数组)
    """
    try:
        img = Image.open(image_path)
        
        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
            
        if target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
        return np.array(img)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def create_evaluation_batches(original_images: Dict[str, Dict[str, Any]], 
                            query_images: Dict[str, Dict[str, Any]], 
                            batch_size: Optional[int] = None) -> List[Tuple[Dict, Dict]]:
    """
    创建用于评估的数据批次
    
    参数:
        original_images (dict): 原始图像字典
        query_images (dict): 查询图像字典
        batch_size (int, optional): 每批处理的查询图像数量
    
    返回:
        list: 批次列表，每个批次包含一组查询图像和所有原始图像
    """
    if batch_size is None or batch_size >= len(query_images):
        return [(query_images, original_images)]
        
    batches = []
    query_ids = list(query_images.keys())
    
    for i in range(0, len(query_ids), batch_size):
        batch_query_ids = query_ids[i:i + batch_size]
        batch_queries = {qid: query_images[qid] for qid in batch_query_ids}
        batches.append((batch_queries, original_images))
        
    return batches

def calculate_metrics(search_results: Dict[str, List[Tuple[str, float]]], 
                     ground_truth: Dict[str, str], 
                     k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    计算检索性能评价指标
    
    参数:
        search_results: 每个查询图像的检索结果
        ground_truth: 真实匹配关系
        k_values: 计算Precision@k和Recall@k的k值列表
    
    返回:
        dict: 包含mAP和各k值下的precision、recall的字典
    """
    metrics = {}
    
    # 计算各k值下的precision和recall
    for k in k_values:
        precisions = []
        recalls = []
        
        for query_id, results in search_results.items():
            if query_id not in ground_truth:
                continue
                
            true_match = ground_truth[query_id]
            top_k_results = [r[0] for r in results[:k]]
            
            # Precision@k
            correct = sum(1 for r in top_k_results if r == true_match)
            precision = correct / k if k > 0 else 0
            precisions.append(precision)
            
            # Recall@k
            recall = 1.0 if true_match in top_k_results else 0.0
            recalls.append(recall)
        
        metrics[f'P@{k}'] = np.mean(precisions) if precisions else 0
        metrics[f'R@{k}'] = np.mean(recalls) if recalls else 0
    
    # 计算mAP
    aps = []
    for query_id, results in search_results.items():
        if query_id not in ground_truth:
            continue
            
        true_match = ground_truth[query_id]
        precision_list = []
        
        for rank, (result_id, _) in enumerate(results, 1):
            if result_id == true_match:
                precision = 1.0 / rank
                precision_list.append(precision)
                
        ap = np.mean(precision_list) if precision_list else 0
        aps.append(ap)
    
    metrics['mAP'] = np.mean(aps) if aps else 0
    
    return metrics

def visualize_results(query_path: str, retrieved_paths: List[Tuple[str, float]], 
                     ground_truth_path: Optional[str] = None, top_k: int = 5) -> None:
    """
    可视化检索结果
    
    参数:
        query_path: 查询图像路径
        retrieved_paths: 检索出的图像路径及其相似度分数列表
        ground_truth_path: 真实匹配的原始图像路径(可选)
        top_k: 显示前k个检索结果
    """
    k = min(top_k, len(retrieved_paths))
    n_cols = k + 1
    if ground_truth_path:
        n_cols += 1
    
    plt.figure(figsize=(3*n_cols, 3))
    
    # 显示查询图像
    plt.subplot(1, n_cols, 1)
    query_img = plt.imread(query_path)
    plt.imshow(query_img)
    plt.title('Query Image')
    plt.axis('off')
    
    # 显示ground truth（如果有）
    if ground_truth_path:
        plt.subplot(1, n_cols, 2)
        gt_img = plt.imread(ground_truth_path)
        plt.imshow(gt_img)
        plt.title('Ground Truth')
        plt.axis('off')
    
    # 显示检索结果
    start_idx = 2 if not ground_truth_path else 3
    for i, (path, score) in enumerate(retrieved_paths[:k]):
        plt.subplot(1, n_cols, start_idx + i)
        result_img = plt.imread(path)
        plt.imshow(result_img)
        plt.title(f'Rank {i+1}\nScore: {score:.3f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def load_copydays_dataset(base_dir: str, preprocess: bool = True, 
                         target_size: Optional[Tuple[int, int]] = None) -> Tuple[Dict, Dict, Dict]:
    """
    加载完整的Copydays数据集并返回评估所需的所有数据
    
    参数:
        base_dir (str): Copydays数据集的根目录
        preprocess (bool): 是否对图像进行预处理
        target_size (tuple, optional): 预处理后的图像尺寸
    
    返回:
        tuple: (original_images, query_images, ground_truth)
               包含所有必要的数据用于后续的拷贝检测实验
    """
    # 确保使用正确的路径分隔符
    jpg_dir = os.path.join(base_dir, 'copydays', 'jpg')
    pkl_path = os.path.join(base_dir, 'copydays', 'gnd_copydays.pkl')
    
    # 验证必要文件的存在
    if not os.path.exists(jpg_dir) or not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Copydays dataset not found in {base_dir}")
        
    # 加载数据集结构信息
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    # 处理原始图像
    original_images = {}
    for img_name in data['imlist']:
        img_path = os.path.join(jpg_dir, f"{img_name}.jpg")
        if os.path.exists(img_path):
            original_images[img_name] = {
                "path": img_path,
                "data": preprocess_image(img_path, target_size) if preprocess else None
            }
            
    # 处理查询图像
    query_images = {}
    for img_name in data['qimlist']:
        img_path = os.path.join(jpg_dir, f"{img_name}.jpg")
        if os.path.exists(img_path):
            query_images[img_name] = {
                "path": img_path,
                "data": preprocess_image(img_path, target_size) if preprocess else None,
                "transform": "original"  # 可以根据文件名解析具体的变换类型
            }
            
    # 获取ground truth映射
    ground_truth = get_ground_truth_mapping(pkl_path)
    
    return original_images, query_images, ground_truth
