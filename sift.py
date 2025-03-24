import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import time

from data import load_copydays_dataset

def extract_sift_features(image: np.ndarray, use_gpu: bool = False) -> Tuple[List, np.ndarray]:
    """
    提取图像的SIFT特征
    
    参数:
        image: 输入图像
        use_gpu: 是否使用GPU加速
    
    返回:
        tuple: (关键点列表, 描述符数组)
    """
    # 根据是否使用GPU选择SIFT实现
    if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # 使用CUDA加速的SIFT
        cuda_image = cv2.cuda_GpuMat()
        
        # 转换为灰度图像(如果是彩色图像)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        cuda_image.upload(gray)
        
        # 创建CUDA SIFT特征提取器
        cuda_sift = cv2.cuda.SIFT_create()
        
        # 检测关键点并计算描述符
        keypoints, descriptors = cuda_sift.detectAndCompute(cuda_image, None)
        
        # 下载描述符到CPU
        if descriptors is not None:
            descriptors = descriptors.download()
    else:
        # 使用CPU的SIFT
        sift = cv2.SIFT_create()
        
        # 转换为灰度图像(如果是彩色图像)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # 检测关键点并计算描述符
        keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def match_features(des1: np.ndarray, des2: np.ndarray, 
                  ratio_threshold: float = 0.75,
                  use_gpu: bool = False) -> List[cv2.DMatch]:
    """
    使用最近邻比率测试进行特征匹配
    
    参数:
        des1: 第一个图像的描述符
        des2: 第二个图像的描述符
        ratio_threshold: 最近邻比率测试的阈值
        use_gpu: 是否使用GPU加速
    
    返回:
        list: 匹配结果列表
    """
    # 确保两个图像都有描述符
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []
    
    matches = []
    
    if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        try:
            # 将描述符上传到GPU
            gpu_des1 = cv2.cuda_GpuMat()
            gpu_des2 = cv2.cuda_GpuMat()
            gpu_des1.upload(np.float32(des1))
            gpu_des2.upload(np.float32(des2))
            
            # 创建GPU匹配器
            matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
            
            # 对每个描述符找到两个最佳匹配
            gpu_matches = matcher.knnMatch(gpu_des1, gpu_des2, k=2)
            
            # 应用比率测试
            for k in range(len(gpu_matches)):
                if len(gpu_matches[k]) == 2:
                    m, n = gpu_matches[k]
                    if m.distance < ratio_threshold * n.distance:
                        matches.append(m)
        except Exception as e:
            print(f"GPU匹配出错，回退到CPU: {e}")
            use_gpu = False
    
    # 如果GPU匹配失败或不使用GPU，则使用CPU
    if not use_gpu:
        # 创建FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 对每个描述符找到两个最佳匹配
        all_matches = flann.knnMatch(des1, des2, k=2)
        
        # 应用比率测试
        for m, n in all_matches:
            if m.distance < ratio_threshold * n.distance:
                matches.append(m)
    
    return matches

def verify_with_ransac(kp1: List, kp2: List, matches: List[cv2.DMatch], 
                      min_inliers: int = 10) -> Tuple[float, np.ndarray]:
    """
    使用RANSAC进行几何验证
    
    参数:
        kp1: 第一个图像的关键点
        kp2: 第二个图像的关键点
        matches: 特征匹配结果
        min_inliers: 被认为是有效拷贝的最小内点数量
    
    返回:
        tuple: (相似度得分, 内点掩码)
    """
    # 如果匹配数量不足，返回低相似度
    if len(matches) < 4:
        return 0.0, None
    
    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # 使用RANSAC计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 计算内点数量
    mask = mask.ravel().tolist()
    inliers_count = sum(mask)
    
    # 计算相似度得分
    if inliers_count >= min_inliers:
        # 基于内点比例和绝对数量的得分
        inlier_ratio = inliers_count / len(matches)
        score = inlier_ratio * (np.log(inliers_count) / np.log(min_inliers))
        return min(score, 1.0), mask
    else:
        return 0.0, mask

def detect_image_copy(query_image: np.ndarray, original_images: Dict[str, Dict[str, Any]], 
                     min_score: float = 0.2, use_gpu: bool = False) -> List[Tuple[str, float, np.ndarray]]:
    """
    检测查询图像是否为原始图像的拷贝
    
    参数:
        query_image: 查询图像
        original_images: 原始图像字典
        min_score: 最小相似度阈值
        use_gpu: 是否使用GPU加速
    
    返回:
        list: 按照相似度排序的(原始图像ID, 相似度得分, 内点掩码)列表
    """
    # 提取查询图像的SIFT特征
    query_kp, query_des = extract_sift_features(query_image, use_gpu)
    
    results = []
    
    # 与每个原始图像比较
    for img_id, img_info in original_images.items():
        # 如果图像数据尚未加载，则加载它
        if img_info["data"] is None:
            img_info["data"] = cv2.imread(img_info["path"])
        
        # 提取原始图像的SIFT特征
        orig_kp, orig_des = extract_sift_features(img_info["data"], use_gpu)
        
        # 进行特征匹配
        matches = match_features(query_des, orig_des, use_gpu=use_gpu)
        
        # 使用RANSAC进行几何验证
        score, mask = verify_with_ransac(query_kp, orig_kp, matches)
        
        # 如果得分超过阈值，将结果加入列表
        if score >= min_score:
            results.append((img_id, score, mask))
    
    # 按相似度得分降序排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def run_copy_detection(query_images: Dict[str, Dict[str, Any]], 
                      original_images: Dict[str, Dict[str, Any]],
                      ground_truth: Optional[Dict[str, str]] = None,
                      use_gpu: bool = False) -> Dict[str, List[Tuple[str, float]]]:
    """
    在整个数据集上运行拷贝检测算法
    
    参数:
        query_images: 查询图像字典
        original_images: 原始图像字典
        ground_truth: 可选的真实匹配关系
        use_gpu: 是否使用GPU加速
    
    返回:
        dict: 每个查询图像的检测结果
    """
    results = {}
    
    # 检查GPU可用性
    if use_gpu:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_count > 0:
            print(f"使用GPU加速 (找到 {gpu_count} 个CUDA设备)")
        else:
            print("未找到可用的CUDA设备，回退到CPU模式")
            use_gpu = False
    else:
        print("使用CPU模式")
    
    start_time = time.time()
    total = len(query_images)
    for i, (query_id, query_info) in enumerate(query_images.items()):
        print(f"处理查询图像 {i+1}/{total}: {query_id}")
        
        # 如果图像数据尚未加载，则加载它
        if query_info["data"] is None:
            query_info["data"] = cv2.imread(query_info["path"])
        
        # 进行拷贝检测
        detection_results = detect_image_copy(query_info["data"], original_images, use_gpu=use_gpu)
        
        # 仅保留原始图像ID和得分
        results[query_id] = [(img_id, score) for img_id, score, _ in detection_results]
        
        # 如果提供了真实匹配，则评估该查询的精确度
        if ground_truth and query_id in ground_truth:
            true_match = ground_truth[query_id]
            found = False
            for rank, (img_id, score) in enumerate(results[query_id]):
                if img_id == true_match:
                    print(f"  在第 {rank+1} 位找到真实匹配，得分 {score:.4f}")
                    found = True
                    break
            if not found and results[query_id]:
                print(f"  未在结果中找到真实匹配。最高匹配: {results[query_id][0][0]}, 得分 {results[query_id][0][1]:.4f}")
    
    total_time = time.time() - start_time
    print(f"总处理时间: {total_time:.2f}秒，平均每张图像: {total_time/total:.2f}秒")
    
    return results

def calculate_metrics(results: Dict[str, List[Tuple[str, float]]], 
                     ground_truth: Dict[str, str]) -> Dict[str, float]:
    """
    计算评价指标
    
    参数:
        results: 检测结果
        ground_truth: 真实匹配关系
    
    返回:
        dict: 包含各种评价指标的字典
    """
    # 初始化评价指标
    precision_at_1 = 0
    precision_at_5 = 0
    recall_at_1 = 0
    recall_at_5 = 0
    average_precision = 0
    
    # 遍历每个查询图像的结果
    for query_id, matches in results.items():
        if query_id not in ground_truth:
            continue
        
        true_match = ground_truth[query_id]
        
        # 计算平均精度 (Average Precision)
        precision_sum = 0
        num_correct = 0
        
        for i, (img_id, _) in enumerate(matches):
            if img_id == true_match:
                num_correct += 1
                precision_sum += num_correct / (i + 1)
                
                # 对于P@k和R@k
                if i == 0:  # P@1 和 R@1
                    precision_at_1 += 1
                    recall_at_1 += 1
                if i < 5:  # P@5 和 R@5
                    precision_at_5 += 1
                    recall_at_5 += 1
        
        # 如果找到了真实匹配，计算AP
        if num_correct > 0:
            average_precision += precision_sum / num_correct
    
    # 计算平均值
    num_queries = len([q_id for q_id in results if q_id in ground_truth])
    
    metrics = {
        "mAP": average_precision / num_queries if num_queries > 0 else 0,
        "P@1": precision_at_1 / num_queries if num_queries > 0 else 0,
        "P@5": precision_at_5 / (5 * num_queries) if num_queries > 0 else 0,
        "R@1": recall_at_1 / num_queries if num_queries > 0 else 0,
        "R@5": recall_at_5 / num_queries if num_queries > 0 else 0
    }
    
    return metrics

# 主函数示例
def main(dataset_dir: str, use_gpu: bool = False):
    """
    主函数
    
    参数:
        dataset_dir: 数据集目录
        use_gpu: 是否使用GPU加速
    """
    # 加载数据集 (假设load_copydays_dataset函数已经实现)
    print("加载数据集...")
    original_images, query_images, ground_truth = load_copydays_dataset(dataset_dir, preprocess=False)
    
    print(f"加载了 {len(original_images)} 张原始图像和 {len(query_images)} 张查询图像")
    
    # 运行拷贝检测
    results = run_copy_detection(query_images, original_images, ground_truth, use_gpu=use_gpu)
    
    # 计算评价指标
    if ground_truth:
        metrics = calculate_metrics(results, ground_truth)
        
        # 打印评价指标
        print("\n评价指标:")
        print(f"  mAP: {metrics['mAP']:.4f}")
        print(f"  P@1: {metrics['P@1']:.4f}")
        print(f"  P@5: {metrics['P@5']:.4f}")
        print(f"  R@1: {metrics['R@1']:.4f}")
        print(f"  R@5: {metrics['R@5']:.4f}")
    
    return results

if __name__ == "__main__":
    # 修改为您的数据集目录
    dataset_dir = "./data"
    
    # 设置是否使用GPU加速
    use_gpu = False  # 改为False以禁用GPU加速
    
    main(dataset_dir, use_gpu)
