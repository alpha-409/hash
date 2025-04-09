import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import logging
from model_code import HashNet, TripletHashLoss

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 导入数据加载工具
from utils import load_data

class HashDataset(Dataset):
    """哈希检索数据集"""
    def __init__(self, data_dict, transform=None, is_query=False):
        """
        初始化数据集
        
        参数:
            data_dict: 包含数据集信息的字典
            transform: 数据预处理转换
            is_query: 是否为查询数据集
        """
        self.is_query = is_query
        
        if is_query:
            self.images = data_dict['query_images']
            self.paths = data_dict['query_paths']
        else:
            self.images = data_dict['db_images']
            self.paths = data_dict['db_paths']
        
        self.transform = transform
        self.positives = data_dict['positives']  # 正样本对列表
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        path = self.paths[idx]
        
        # 如果图像不是张量, 应用变换
        if not isinstance(image, torch.Tensor) and self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'idx': idx,
            'path': path
        }


def collate_with_positives(batch):
    """自定义整批函数, 收集正样本对信息"""
    images = torch.stack([item['image'] for item in batch])
    indices = [item['idx'] for item in batch]
    paths = [item['path'] for item in batch]
    
    return {
        'images': images,
        'indices': indices,
        'paths': paths
    }


def train(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    """训练一个周期"""
    model.train()
    running_loss = 0.0
    loss_components_sum = {"similarity_loss": 0.0, "quantization_loss": 0.0, "bit_balance_loss": 0.0}
    
    epoch_start_time = time.time()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        indices = batch['indices']
        
        # 在当前批次图像索引上筛选出正样本对
        batch_positives = []
        for q_idx, db_idx in train_loader.dataset.positives:
            # 将全局索引转换为批次内的局部索引
            q_local_idx = indices.index(q_idx) if q_idx in indices else -1
            db_local_idx = indices.index(db_idx) if db_idx in indices else -1
            
            if q_local_idx >= 0 and db_local_idx >= 0:
                batch_positives.append((q_local_idx, db_local_idx))
        
        # 前向传播
        binary_hash, continuous_hash, _ = model(images)
        
        # 计算损失
        loss, loss_components = criterion(binary_hash, continuous_hash, batch_positives)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累积损失
        running_loss += loss.item()
        for key, value in loss_components.items():
            if key in loss_components_sum:
                loss_components_sum[key] += value
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'sim_loss': f"{loss_components['similarity_loss']:.4f}",
            'quant_loss': f"{loss_components['quantization_loss']:.4f}"
        })
    
    # 计算平均损失
    epoch_loss = running_loss / len(train_loader)
    avg_loss_components = {k: v / len(train_loader) for k, v in loss_components_sum.items()}
    
    epoch_time = time.time() - epoch_start_time
    
    # 日志记录
    logger.info(f"Epoch {epoch}/{total_epochs} completed in {epoch_time:.2f}s")
    logger.info(f"Training Loss: {epoch_loss:.4f}")
    logger.info(f"Loss Components: Similarity={avg_loss_components['similarity_loss']:.4f}, "
                f"Quantization={avg_loss_components['quantization_loss']:.4f}, "
                f"Bit Balance={avg_loss_components['bit_balance_loss']:.4f}")
    
    return epoch_loss, avg_loss_components
