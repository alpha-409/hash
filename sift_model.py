import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import time
from tqdm import tqdm

class SIFTCopyDetector:
    """基于SIFT特征和RANSAC几何验证的图像拷贝检测器"""
    
    def __init__(self, min_matches=10, ransac_reproj_threshold=5.0, use_gpu=False):
        """
        初始化SIFT拷贝检测器
        
        参数:
            min_matches (int): 判定为匹配所需的最小特征点数量
            ransac_reproj_threshold (float): RANSAC重投影阈值
            use_gpu (bool): 是否使用GPU加速SIFT特征提取
        """
        self.min_matches = min_matches
        self.ransac_threshold = ransac_reproj_threshold
        self.use_gpu = use_gpu
        
        # 创建SIFT特征提取器
        if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("GPU加速SIFT已启用")
            self.sift = cv2.cuda.SIFT_create()
        else:
            if use_gpu:
                print("GPU不可用，使用CPU运行SIFT")
            else:
                print("使用CPU运行SIFT")
            self.sift = cv2.SIFT_create()
                
        # 创建特征匹配器 - 使用FLANN匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 存储参考图像特征
        self.reference_features = {}  # {image_id: (keypoints, descriptors)}
        
    def extract_features(self, image_path_or_data):
        """
        提取图像的SIFT特征
        
        参数:
            image_path_or_data: 图像路径或已加载的图像数据
            
        返回:
            tuple: (keypoints, descriptors)
        """
        # 加载图像
        if isinstance(image_path_or_data, str):
            img = cv2.imread(image_path_or_data, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_path_or_data, np.ndarray):
            if len(image_path_or_data.shape) == 3:
                img = cv2.cvtColor(image_path_or_data, cv2.COLOR_RGB2GRAY)
            else:
                img = image_path_or_data
        else:
            raise ValueError("不支持的图像数据类型")
            
        if img is None or img.size == 0:
            return [], None
            
        # 提取SIFT特征
        if self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # GPU版本
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            keypoints, descriptors = self.sift.detectAndCompute(gpu_img, None)
            if descriptors is not None:
                descriptors = descriptors.download()
        else:
            # CPU版本
            keypoints, descriptors = self.sift.detectAndCompute(img, None)
            
        return keypoints, descriptors
        
    def index_reference_images(self, original_images: Dict[str, Dict[str, Any]]):
        """
        为所有参考图像提取SIFT特征并建立索引
        
        参数:
            original_images: 原始图像字典
        """
        print("为参考图像提取SIFT特征并建立索引...")
        start_time = time.time()
        
        for image_id, image_info in tqdm(original_images.items()):
            # 提取特征
            if image_info['data'] is not None:
                keypoints, descriptors = self.extract_features(image_info['data'])
            else:
                keypoints, descriptors = self.extract_features(image_info['path'])
                
            # 存储特征
            if descriptors is not None and len(keypoints) > 0:
                self.reference_features[image_id] = (keypoints, descriptors)
            
        elapsed_time = time.time() - start_time
        print(f"索引完成，共{len(self.reference_features)}张参考图像，用时：{elapsed_time:.2f}秒")
        
    def match_features(self, desc1, desc2):
        """
        匹配两组特征描述子
        
        参数:
            desc1, desc2: 两组特征描述子
            
        返回:
            list: 特征匹配
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
            
        # 使用2NN匹配并进行比率测试以过滤不良匹配
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # 应用David Lowe的比率测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # 比率阈值
                good_matches.append(m)
                
        return good_matches
        
    def geometric_verification(self, keypoints1, keypoints2, matches):
        """
        使用RANSAC进行几何验证
        
        参数:
            keypoints1, keypoints2: 两组关键点
            matches: 特征匹配
            
        返回:
            tuple: (是否通过验证, 单应矩阵, 内点数量, 内点百分比)
        """
        if len(matches) < self.min_matches:
            return False, None, 0, 0.0
            
        # 提取匹配点坐标
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC计算单应矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        
        # 计算内点数量和百分比
        inliers = np.sum(mask) if mask is not None else 0
        inlier_ratio = inliers / len(matches) if len(matches) > 0 else 0
        
        # 验证是否通过几何检验
        is_verified = inliers >= self.min_matches
        
        return is_verified, H, inliers, inlier_ratio
        
    def compute_similarity(self, query_features, ref_features):
        """
        计算两个图像特征之间的相似度
        
        参数:
            query_features: 查询图像特征 (keypoints, descriptors)
            ref_features: 参考图像特征 (keypoints, descriptors)
            
        返回:
            dict: 相似度信息: {
                'score': 相似度得分,
                'matches': 匹配数,
                'inliers': 内点数,
                'ratio': 内点比例,
                'verified': 是否通过验证
            }
        """
        query_kp, query_desc = query_features
        ref_kp, ref_desc = ref_features
        
        # 匹配特征
        matches = self.match_features(query_desc, ref_desc)
        
        # 几何验证
        verified, homography, inliers, inlier_ratio = self.geometric_verification(
            query_kp, ref_kp, matches)
            
        # 计算相似度得分
        # 基于内点数量和比例的综合得分
        if verified:
            base_score = min(1.0, inliers / 100.0)  # 根据内点数量计算基础分数
            score = base_score * (0.5 + 0.5 * inlier_ratio)  # 考虑内点比例提高分数
        else:
            score = 0.0
            
        return {
            'score': score,
            'matches': len(matches),
            'inliers': inliers,
            'ratio': inlier_ratio,
            'verified': verified
        }
        
    def search(self, query_images: Dict[str, Dict[str, Any]], 
              top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        搜索匹配图像
        
        参数:
            query_images: 查询图像字典
            top_k: 每个查询返回的最大匹配数
            
        返回:
            dict: 查询结果, {query_id: [(match_id, similarity_score), ...]}
        """
        search_results = {}
        start_time = time.time()
        
        print("使用SIFT+RANSAC搜索匹配图像...")
        for query_id, query_info in tqdm(query_images.items()):
            # 提取查询图像特征
            if query_info['data'] is not None:
                query_kp, query_desc = self.extract_features(query_info['data'])
            else:
                query_kp, query_desc = self.extract_features(query_info['path'])
                
            # 跳过无特征的图像
            if query_desc is None or len(query_kp) == 0:
                search_results[query_id] = []
                continue
                
            # 与所有参考图像比较
            similarities = []
            for ref_id, ref_features in self.reference_features.items():
                similarity_info = self.compute_similarity(
                    (query_kp, query_desc), ref_features)
                
                similarities.append((ref_id, similarity_info['score'], similarity_info))
                
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 仅保留分数
            search_results[query_id] = [(s[0], s[1]) for s in similarities[:top_k]]
            
        elapsed_time = time.time() - start_time
        print(f"搜索完成，用时：{elapsed_time:.2f}秒")
        return search_results
    
    def visualize_matches(self, query_image, reference_image, query_id, ref_id):
        """
        可视化两个图像之间的匹配
        
        参数:
            query_image: 查询图像数据
            reference_image: 参考图像数据
            query_id: 查询图像ID
            ref_id: 参考图像ID
            
        返回:
            ndarray: 可视化结果图像
        """
        # 确保两个图像都存在
        if isinstance(query_image, str):
            query_img = cv2.imread(query_image)
        else:
            query_img = query_image.copy()
            if len(query_img.shape) == 2:
                query_img = cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR)
                
        if isinstance(reference_image, str):
            ref_img = cv2.imread(reference_image)
        else:
            ref_img = reference_image.copy()
            if len(ref_img.shape) == 2:
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                
        # 提取特征
        query_kp, query_desc = self.extract_features(query_img)
        
        # 获取参考图像的特征
        if ref_id in self.reference_features:
            ref_kp, ref_desc = self.reference_features[ref_id]
        else:
            ref_kp, ref_desc = self.extract_features(ref_img)
            
        # 匹配特征
        matches = self.match_features(query_desc, ref_desc)
        
        # 几何验证
        verified, homography, inliers, inlier_ratio = self.geometric_verification(
            query_kp, ref_kp, matches)
            
        # 将匹配转换为列表以便可视化
        good_matches = []
        for m in matches:
            good_matches.append(m)
            
        # 绘制匹配
        result_img = cv2.drawMatches(query_img, query_kp, ref_img, ref_kp, 
                                   good_matches, None, 
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # 添加验证结果标题
        status = "验证通过" if verified else "验证失败"
        title = f"匹配: {len(matches)}  内点: {inliers}  比例: {inlier_ratio:.2f}  {status}"
        
        # 在图像上添加文字
        cv2.putText(result_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 255, 0) if verified else (0, 0, 255), 2)
        
        return result_img
