import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

def preprocess_image_for_sift(image, target_size=None, enhance_contrast=False):
    """
    预处理图像以提高SIFT特征提取效果
    
    参数:
        image: 输入图像
        target_size: 可选的目标大小(宽,高)
        enhance_contrast: 是否增强对比度
        
    返回:
        处理后的图像
    """
    # 确保图像是灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 调整大小
    if target_size is not None:
        gray = cv2.resize(gray, target_size)
        
    # 对比度增强
    if enhance_contrast:
        # 使用CLAHE算法增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
    return gray
    
def visualize_sift_features(image, keypoints, title="SIFT特征点"):
    """
    可视化SIFT特征点
    
    参数:
        image: 输入图像
        keypoints: SIFT关键点
        title: 图像标题
    """
    # 确保图像是BGR格式用于可视化
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
        
    # 绘制关键点
    img_keypoints = cv2.drawKeypoints(vis_img, keypoints, None, 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # 显示图像
    plt.figure(figsize=(10, 6))
    plt.title(f"{title} - {len(keypoints)}个特征点")
    plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def evaluate_transformation_robustness(detector, reference_image, transformations):
    """
    评估SIFT检测器对各种图像变换的鲁棒性
    
    参数:
        detector: SIFTCopyDetector实例
        reference_image: 参考图像
        transformations: 变换函数列表，每个函数接收图像并返回变换后的图像
        
    返回:
        dict: 对每种变换的检测性能
    """
    results = {}
    
    # 加载参考图像
    if isinstance(reference_image, str):
        ref_img = cv2.imread(reference_image)
    else:
        ref_img = reference_image.copy()
        
    # 提取参考图像特征
    ref_kp, ref_desc = detector.extract_features(ref_img)
    ref_features = (ref_kp, ref_desc)
    
    # 应用每种变换并评估
    for transform_name, transform_func in transformations.items():
        print(f"评估变换: {transform_name}")
        
        # 应用变换
        transformed_img = transform_func(ref_img)
        
        # 提取变换图像特征
        trans_kp, trans_desc = detector.extract_features(transformed_img)
        
        if trans_desc is not None and len(trans_kp) > 0:
            # 计算相似度
            similarity = detector.compute_similarity(
                (trans_kp, trans_desc), ref_features)
                
            results[transform_name] = {
                'similarity': similarity,
                'keypoints': len(trans_kp),
                'matches': similarity['matches'],
                'inliers': similarity['inliers'],
                'ratio': similarity['ratio'],
                'verified': similarity['verified']
            }
        else:
            results[transform_name] = {
                'similarity': {'score': 0, 'matches': 0, 'inliers': 0, 'ratio': 0, 'verified': False},
                'keypoints': 0,
                'matches': 0,
                'inliers': 0,
                'ratio': 0,
                'verified': False
            }
    
    return results

def generate_transformations():
    """
    生成一组常见的图像变换函数
    
    返回:
        dict: {变换名称: 变换函数}
    """
    transformations = {}
    
    # 旋转变换
    transformations['rotation_10'] = lambda img: _rotate_image(img, 10)
    transformations['rotation_45'] = lambda img: _rotate_image(img, 45)
    
    # 缩放变换
    transformations['scale_0.5'] = lambda img: cv2.resize(img, None, fx=0.5, fy=0.5)
    transformations['scale_2.0'] = lambda img: cv2.resize(img, None, fx=2.0, fy=2.0)
    
    # JPEG压缩
    transformations['jpeg_50'] = lambda img: _jpeg_compression(img, 50)
    transformations['jpeg_20'] = lambda img: _jpeg_compression(img, 20)
    
    # 亮度变化
    transformations['brightness_+50'] = lambda img: _adjust_brightness(img, 50)
    transformations['brightness_-50'] = lambda img: _adjust_brightness(img, -50)
    
    # 裁剪
    transformations['crop_30%'] = lambda img: _crop_image(img, 0.3)
    
    # 高斯模糊
    transformations['gaussian_blur'] = lambda img: cv2.GaussianBlur(img, (5, 5), 0)
    
    # 透视变换
    transformations['perspective'] = _apply_perspective_transform
    
    return transformations

# 辅助变换函数
def _rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def _jpeg_compression(image, quality):
    # 保存为JPEG并重新加载
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)

def _adjust_brightness(image, beta):
    return np.clip(image.astype(np.int16) + beta, 0, 255).astype(np.uint8)

def _crop_image(image, crop_ratio):
    height, width = image.shape[:2]
    crop_px = int(min(height, width) * crop_ratio)
    start_x = crop_px
    start_y = crop_px
    end_x = width - crop_px
    end_y = height - crop_px
    return image[start_y:end_y, start_x:end_x].copy()

def _apply_perspective_transform(image):
    height, width = image.shape[:2]
    
    # 定义源点和目标点
    src_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ])
    
    # 轻微透视变换的目标点
    offset = int(0.1 * min(width, height))
    dst_points = np.float32([
        [offset, 0],
        [width - 1 - offset, offset],
        [offset, height - 1],
        [width - 1 - offset, height - 1 - offset]
    ])
    
    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, M, (width, height))
