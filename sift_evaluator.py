import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class SIFTCopyDetectionEvaluator:
    """基于SIFT特征的拷贝检测评估器"""
    
    def __init__(self, ground_truth: Dict[str, str]):
        """
        初始化评估器
        
        参数:
            ground_truth: 真实匹配关系，格式: {query_id: original_id}
        """
        self.ground_truth = ground_truth
        
    def calculate_map(self, search_results: Dict[str, List[Tuple[str, float]]]) -> float:
        """
        计算平均精度均值 (mAP)
        
        参数:
            search_results: 搜索结果，格式: {query_id: [(match_id, score), ...]}
            
        返回:
            float: mAP值
        """
        aps = []
        
        for query_id, results in search_results.items():
            if query_id not in self.ground_truth:
                continue
                
            true_match = self.ground_truth[query_id]
            
            # 找到真实匹配在结果中的位置
            rank = None
            for i, (match_id, _) in enumerate(results):
                if match_id == true_match:
                    rank = i
                    break
            
            # 计算AP
            if rank is not None:
                # AP = 1/(rank+1) 因为rank从0开始
                ap = 1.0 / (rank + 1)
                aps.append(ap)
            else:
                aps.append(0.0)  # 未找到匹配
        
        # 计算mAP
        mAP = np.mean(aps) if aps else 0.0
        return mAP
    
    def calculate_uap(self, search_results: Dict[str, List[Tuple[str, float]]], 
                     thresholds: List[float]) -> Dict[float, float]:
        """
        计算微平均精度 (μAP) 在不同阈值下
        
        参数:
            search_results: 搜索结果
            thresholds: 相似度阈值列表
            
        返回:
            dict: {threshold: uAP_value} 不同阈值下的μAP
        """
        uap_scores = {}
        
        for threshold in thresholds:
            tp = 0  # 真正例数
            fp = 0  # 假正例数
            
            for query_id, results in search_results.items():
                if query_id not in self.ground_truth:
                    continue
                    
                true_match = self.ground_truth[query_id]
                
                # 检查高于阈值的匹配
                matches_above_threshold = [(match_id, score) for match_id, score in results 
                                         if score >= threshold]
                
                found_true_match = False
                for match_id, _ in matches_above_threshold:
                    if match_id == true_match:
                        tp += 1
                        found_true_match = True
                    else:
                        fp += 1
                        
                # 如果没有找到真实匹配，但有其他匹配超过阈值，是假正例
                if not found_true_match and matches_above_threshold:
                    fp += 1
            
            # 计算μAP = TP/(TP+FP)
            if tp + fp > 0:
                uap_scores[threshold] = tp / (tp + fp)
            else:
                uap_scores[threshold] = 0.0
                
        return uap_scores
    
    def calculate_precision_recall_curve(self, search_results: Dict[str, List[Tuple[str, float]]],
                                       thresholds: List[float]):
        """
        计算精确率-召回率曲线数据
        
        参数:
            search_results: 搜索结果
            thresholds: 阈值列表
            
        返回:
            dict: 包含precision, recall和thresholds的字典
        """
        precisions = []
        recalls = []
        
        total_positives = len(self.ground_truth)
        
        for threshold in thresholds:
            tp = 0  # 真正例
            fp = 0  # 假正例
            
            for query_id, results in search_results.items():
                if query_id not in self.ground_truth:
                    continue
                    
                true_match = self.ground_truth[query_id]
                
                # 检查高于阈值的匹配
                matches = [(match_id, score) for match_id, score in results 
                          if score >= threshold]
                
                # 检查是否有真实匹配
                match_found = any(match_id == true_match for match_id, _ in matches)
                
                if match_found:
                    tp += 1
                elif matches:  # 如果有其他匹配但没有正确匹配
                    fp += 1
            
            # 计算精确率和召回率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / total_positives if total_positives > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        return {
            'precision': precisions,
            'recall': recalls,
            'thresholds': thresholds
        }
    
    def evaluate(self, search_results: Dict[str, List[Tuple[str, float]]], 
                thresholds: List[float] = None) -> Dict[str, Any]:
        """
        全面评估拷贝检测算法性能
        
        参数:
            search_results: 搜索结果
            thresholds: 阈值列表
            
        返回:
            dict: 评估指标字典
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
        mAP = self.calculate_map(search_results)
        uAP = self.calculate_uap(search_results, thresholds)
        pr_curve = self.calculate_precision_recall_curve(search_results, thresholds)
        
        # 找到最佳阈值（基于F1分数）
        best_f1 = 0
        best_threshold = 0
        
        for i, threshold in enumerate(thresholds):
            precision = pr_curve['precision'][i]
            recall = pr_curve['recall'][i]
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        return {
            'mAP': mAP,
            'uAP': uAP,
            'pr_curve': pr_curve,
            'best_threshold': best_threshold,
            'best_f1': best_f1
        }
    
    def evaluate_by_transformation(self, search_results: Dict[str,# 基于SIFT特征和RANSAC几何验证的拷贝检测算法实现

我将实现一个基于SIFT特征提取和RANSAC几何验证的经典拷贝检测算法。这个算法比简单的哈希方法更强大，能够处理更复杂的图像变换。我会将代码分成几个文件进行组织。

## `sift_model.py`

```python:sift_model.py
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from tqdm import tqdm

class SIFTCopyDetector:
    """基于SIFT特征和RANSAC几何验证的图像拷贝检测器"""
    
    def __init__(self, 
                 min_match_count=4, 
                 ratio_threshold=0.75, 
                 ransac_reproj_threshold=5.0,
                 max_features=None,
                 enable_gpu=False):
        """
        初始化SIFT拷贝检测器
        
        参数:
            min_match_count (int): RANSAC验证所需的最小匹配点数
            ratio_threshold (float): Lowe比率测试阈值（0-1）
            ransac_reproj_threshold (float): RANSAC重投影阈值
            max_features (int): 每幅图像提取的最大特征数量，None表示不限制
            enable_gpu (bool): 是否启用GPU加速（需要OpenCV CUDA支持）
        """
        self.min_match_count = min_match_count
        self.ratio_threshold = ratio_threshold
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.max_features = max_features
        
        # 初始化SIFT特征提取器
        try:
            if enable_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print(f"GPU加速已启用，使用设备: {cv2.cuda.getDevice()}")
                # 注意: OpenCV CUDA版本可能不同，根据实际情况调整API
                self.sift = cv2.cuda_SIFT.create(
                    nfeatures=max_features if max_features else 0
                )
                self.use_gpu = True
            else:
                # 如果没有GPU支持或未启用，使用CPU版本
                self.sift = cv2.SIFT_create(
                    nfeatures=max_features if max_features else 0
                )
                self.use_gpu = False
                if enable_gpu:
                    print("未检测到GPU支持，使用CPU")
                else:
                    print("GPU加速未启用，使用CPU")
        except AttributeError:
            # 旧版OpenCV可能使用不同的SIFT创建方法
            self.sift = cv2.xfeatures2d.SIFT_create(
                nfeatures=max_features if max_features else 0
            )
            self.use_gpu = False
            print("使用OpenCV xfeatures2d.SIFT，在CPU上运行")
        
        # 特征匹配器
        self.matcher = cv2.BFMatcher()
        
        # 参考图像的特征
        self.reference_features = {}  # {image_id: (keypoints, descriptors)}
        self.reference_dimensions = {}  # {image_id: (height, width)}
        
    def extract_features(self, image_data_or_path):
        """
        从图像提取SIFT特征
        
        参数:
            image_data_or_path: 图像数据(numpy数组)或图像路径
            
        返回:
            tuple: (关键点列表, 特征描述符)
        """
        # 加载图像
        if isinstance(image_data_or_path, str):
            image = cv2.imread(image_data_or_path, cv2.IMREAD_GRAYSCALE)
        else:
            if len(image_data_or_path.shape) == 3:
                # 如果是彩色图像，转换为灰度
                image = cv2.cvtColor(image_data_or_path, cv2.COLOR_RGB2GRAY)
            else:
                image = image_data_or_path
        
        # 图像预处理（例如，直方图均衡化）
        image = cv2.equalizeHist(image)
        
        if self.use_gpu:
            # GPU版本的SIFT特征提取
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            gpu_keypoints, gpu_descriptors = self.sift.detectAndCompute(gpu_img, None)
            keypoints = gpu_keypoints.download()
            descriptors = gpu_descriptors.download()
        else:
            # CPU版本的SIFT特征提取
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
        
        # 记录图像尺寸，用于后续几何变换计算
        height, width = image.shape[:2]
        
        return keypoints, descriptors, (height, width)
    
    def index_reference_images(self, original_images: Dict[str, Dict[str, Any]]):
        """
        为所有参考图像提取SIFT特征并建立索引
        
        参数:
            original_images: 原始图像字典
        """
        print("正在为参考图像提取SIFT特征...")
        for image_id, image_info in tqdm(original_images.items()):
            # 提取特征
            if image_info['data'] is not None:
                keypoints, descriptors, dimensions = self.extract_features(image_info['data'])
            else:
                keypoints, descriptors, dimensions = self.extract_features(image_info['path'])
            
            # 存储特征和图像尺寸
            if descriptors is not None and len(keypoints) > 0:
                self.reference_features[image_id] = (keypoints, descriptors)
                self.reference_dimensions[image_id] = dimensions
        
        print(f"索引完成，共{len(self.reference_features)}张参考图像")
    
    def match_features(self, query_keypoints, query_descriptors, ref_keypoints, ref_descriptors):
        """
        使用Lowe比率测试进行特征匹配
        
        参数:
            query_keypoints: 查询图像关键点
            query_descriptors: 查询图像特征描述符
            ref_keypoints: 参考图像关键点
            ref_descriptors: 参考图像特征描述符
            
        返回:
            list: 好的匹配对
        """
        # 匹配特征
        if query_descriptors is None or ref_descriptors is None:
            return []
            
        matches = self.matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
        
        # 应用Lowe比率测试筛选好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
                
        return good_matches
    
    def verify_geometry(self, query_keypoints, ref_keypoints, good_matches, ref_dimensions):
        """
        使用RANSAC进行几何验证
        
        参数:
            query_keypoints: 查询图像关键点
            ref_keypoints: 参考图像关键点
            good_matches: 匹配点对
            ref_dimensions: 参考图像尺寸(height, width)
            
        返回:
            tuple: (匹配是否成功, 匹配质量分数, 内点比例, 变换矩阵)
        """
        # 如果匹配点数不足，无法进行几何验证
        if len(good_matches) < self.min_match_count:
            return False, 0.0, 0.0, None
        
        # 提取匹配点坐标
        src_pts = np.float32([query_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([ref_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC找到单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj_threshold)
        
        # 计算内点比例作为匹配质量指标
        matchesMask = mask.ravel().tolist()
        inlier_ratio = sum(matchesMask) / len(matchesMask) if matchesMask else 0
        
        # 检查变换的有效性
        if H is not None and inlier_ratio > 0.3:  # 至少30%的匹配点是内点
            # 计算匹配质量分数 (结合匹配点数和内点比例)
            # 匹配点数越多，内点比例越高，分数越高
            score = inlier_ratio * min(1.0, len(good_matches) / 50.0)
            return True, score, inlier_ratio, H
        else:
            return False, 0.0, inlier_ratio, None
    
    def search(self, query_images: Dict[str, Dict[str, Any]], 
               top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        搜索查询图像的匹配项
        
        参数:
            query_images: 查询图像字典
            top_k: 每个查询返回的最大匹配数
            
        返回:
            dict: 查询结果，格式: {query_id: [(match_id, similarity_score), ...]}
        """
        search_results = {}
        start_time = time.time()
        
        print("使用SIFT+RANSAC搜索匹配图像...")
        for query_id, query_info in tqdm(query_images.items()):
            # 提取查询图像特征
            if query_info['data'] is not None:
                query_keypoints, query_descriptors, query_dims = self.extract_features(query_info['data'])
            else:
                query_keypoints, query_descriptors, query_dims = self.extract_features(query_info['path'])
            
            # 如果未提取到特征，则跳过
            if query_descriptors is None or len(query_keypoints) == 0:
                search_results[query_id] = []
                continue
            
            # 与所有参考图像进行匹配
            matches = []
            for ref_id, (ref_keypoints, ref_descriptors) in self.reference_features.items():
                # 特征匹配
                good_matches = self.match_features(query_keypoints, query_descriptors, ref_keypoints, ref_descriptors)
                
                # 几何验证
                if good_matches:
                    verified, score, inlier_ratio, H = self.verify_geometry(
                        query_keypoints, ref_keypoints, good_matches, self.reference_dimensions[ref_id]
                    )
                    
                    if verified:
                        matches.append((ref_id, score, len(good_matches), inlier_ratio, H))
                    elif len(good_matches) >= self.min_match_count:
                        # 特征匹配数量够多，但几何验证失败，仍然可能是相关图像
                        weak_score = len(good_matches) / 100.0 * 0.4  # 弱匹配分数
                        matches.append((ref_id, weak_score, len(good_matches), inlier_ratio, None))
            
            # 按相似度分数排序
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # 仅保留top-k结果
            search_results[query_id] = [(match_id, score) for match_id, score, _, _, _ in matches[:top_k]]
        
        elapsed_time = time.time() - start_time
        print(f"搜索完成，用时：{elapsed_time:.2f}秒")
        return search_results
    
    def visualize_match(self, query_image, reference_image, query_keypoints, ref_keypoints, good_matches, H=None):
        """
        可视化匹配结果
        
        参数:
            query_image: 查询图像
            reference_image: 参考图像
            query_keypoints: 查询图像关键点
            ref_keypoints: 参考图像关键点
            good_matches: 好的匹配点
            H: 单应性矩阵（变换矩阵）
            
        返回:
            numpy.ndarray: 可视化结果图像
        """
        # 如果输入是路径，加载图像
        if isinstance(query_image, str):
            query_image = cv2.imread(query_image)
        if isinstance(reference_image, str):
            reference_image = cv2.imread(reference_image)
            
        # 转换为BGR格式（如果需要）
        if query_image.shape[2] == 1:
            query_image = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
        if reference_image.shape[2] == 1:
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
            
        # 绘制匹配
        draw_params = dict(
            matchColor=(0, 255, 0),  # 绿色连线表示匹配
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # 使用单应性矩阵标记变换
        if H is not None:
            # 获取查询图像的四个角点
            h, w = query_image.shape[:2]
            corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            # 将角点变换到参考图像坐标系
            transformed_corners = cv2.perspectiveTransform(corners, H)
            
            # 在参考图像上绘制边框
            reference_image = cv2.polylines(
                reference_image, 
                [np.int32(transformed_corners)], 
                True, (0, 0, 255), 3, cv2.LINE_AA
            )
            
        # 绘制匹配点连线
        result_image = cv2.drawMatches(
            query_image, query_keypoints, 
            reference_image, ref_keypoints, 
            good_matches, None, **draw_params
        )
        
        return result_image
