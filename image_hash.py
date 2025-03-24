import cv2
import numpy as np
from PIL import Image
import imagehash
from scipy.fftpack import dct
import torch

def average_hash(image, hash_size=8):
    """
    计算平均哈希 (aHash)
    
    参数:
        image: PIL图像或Tensor
        hash_size: 哈希大小
        
    返回:
        哈希值 (numpy数组)
    """
    if isinstance(image, torch.Tensor):
        # 转换Tensor为PIL图像
        if image.dim() == 4:  # 批处理
            image = image[0]
        
        # 确保图像是3通道的
        if image.shape[0] == 3:
            image = image.cpu().numpy().transpose(1, 2, 0)
            # 如果有归一化，需要反归一化
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)
        else:
            raise ValueError(f"不支持的图像通道数: {image.shape[0]}")
    
    # 使用imagehash库计算平均哈希
    try:
        hash_value = imagehash.average_hash(image, hash_size)
        # 转换为二进制数组
        hash_array = np.array(list(hash_value.hash.flatten())).astype(np.float32)
        return hash_array
    except Exception as e:
        print(f"计算平均哈希时出错: {e}")
        # 返回一个全零的哈希值
        return np.zeros(hash_size * hash_size, dtype=np.float32)

def perceptual_hash(image, hash_size=8):
    """
    计算感知哈希 (pHash)
    
    参数:
        image: PIL图像或Tensor
        hash_size: 哈希大小
        
    返回:
        哈希值 (numpy数组)
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    hash_value = imagehash.phash(image, hash_size)
    hash_array = np.array(list(hash_value.hash.flatten())).astype(np.float32)
    return hash_array

def difference_hash(image, hash_size=8):
    """
    计算差值哈希 (dHash)
    
    参数:
        image: PIL图像或Tensor
        hash_size: 哈希大小
        
    返回:
        哈希值 (numpy数组)
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    hash_value = imagehash.dhash(image, hash_size)
    hash_array = np.array(list(hash_value.hash.flatten())).astype(np.float32)
    return hash_array

def wavelet_hash(image, hash_size=8):
    """
    计算小波哈希 (wHash)
    
    参数:
        image: PIL图像或Tensor
        hash_size: 哈希大小
        
    返回:
        哈希值 (numpy数组)
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    hash_value = imagehash.whash(image, hash_size)
    hash_array = np.array(list(hash_value.hash.flatten())).astype(np.float32)
    return hash_array

def color_hash(image, hash_size=8):
    """
    计算颜色哈希 (cHash)
    
    参数:
        image: PIL图像或Tensor
        hash_size: 哈希大小
        
    返回:
        哈希值 (numpy数组)
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    hash_value = imagehash.colorhash(image, binbits=hash_size//2)
    hash_array = np.array(list(hash_value.hash.flatten())).astype(np.float32)
    return hash_array

def marr_hildreth_hash(image, hash_size=8):
    """
    计算Marr-Hildreth哈希 (mhHash)
    
    参数:
        image: PIL图像或Tensor
        hash_size: 哈希大小
        
    返回:
        哈希值 (numpy数组)
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    # 使用默认参数
    hash_value = imagehash.phash_simple(image, hash_size, highfreq_factor=4)
    hash_array = np.array(list(hash_value.hash.flatten())).astype(np.float32)
    return hash_array