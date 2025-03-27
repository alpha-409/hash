# contrastive_detection.py
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import tensorly as tl
from tensorly.decomposition import parafac
import os
import random
from torch.utils.data import Dataset, DataLoader

class CopyDaysTripletDataset(Dataset):
    def __init__(self, query_images, db_images, positives, num_negatives=5, augment=True):
        """
        参数:
            query_images (list): 查询图像张量列表
            db_images (list): 数据库图像张量列表
            positives (list): 正样本对列表 [(q_idx, db_idx), ...]
            num_negatives (int): 每个正样本对生成的负样本数
            augment (bool): 是否应用数据增强
        """
        self.query = query_images
        self.db = db_images
        self.pos_pairs = positives
        self.num_negatives = num_negatives
        self.augment = augment
        
        # 预生成负样本索引映射：{q_idx: [负样本db_idx列表]}
        self.neg_map = {}
        for q_idx, pos_db_idx in positives:
            # 获取所有可能的负样本（排除正样本）
            all_db_indices = set(range(len(db_images))) - {pos_db_idx}
            self.neg_map[q_idx] = random.sample(list(all_db_indices), num_negatives)
        
        # 数据增强变换
        self.augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pos_pairs) * self.num_negatives  # 每个正样本生成多个负样本

    def __getitem__(self, idx):
        # 计算对应的正样本对索引和负样本序号
        pair_idx = idx // self.num_negatives
        neg_idx = idx % self.num_negatives
        
        q_idx, pos_db_idx = self.pos_pairs[pair_idx]
        neg_db_idx = self.neg_map[q_idx][neg_idx]
        
        # 获取原始图像张量
        anchor = self.query[q_idx].clone().detach()
        positive = self.db[pos_db_idx].clone().detach()
        negative = self.db[neg_db_idx].clone().detach()
        
        # 应用数据增强（仅对锚点和正样本）
        if self.augment:
            anchor = self._augment(anchor)
            positive = self._augment(positive)
            negative = self._augment(negative)  # 可选是否增强负样本
            
        return anchor, positive, negative

    def _augment(self, img_tensor):
        """应用随机增强到张量"""
        # 转换为PIL图像进行增强
        img_pil = transforms.ToPILImage()(img_tensor.cpu())
        # 应用增强变换
        img_aug = self.augment_transform(img_pil)
        return img_aug

class ContrastiveHashModel(nn.Module):
    def __init__(self, base_model='resnet50', projection_dim=128, hash_size=64):
        super().__init__()
        # 加载预训练的对比学习模型
        self.contrastive_model = ContrastiveModel(base_model, projection_dim)
        # 添加哈希生成层
        self.hash_layer = nn.Linear(2048, hash_size)
        self.hash_size = hash_size
        
    def forward(self, x):
        features, _ = self.contrastive_model(x)
        hashes = torch.sigmoid(self.hash_layer(features))
        return features, hashes
class ContrastiveLearner:
    def __init__(self, base_model='resnet50', projection_dim=128, hash_size=64):
        self.model = ContrastiveHashModel(base_model, projection_dim, hash_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, data_dir, epochs=20, batch_size=64):
        # 加载数据
        train_loader = prepare_dataloaders(data_dir, batch_size)
        
        # 定义优化器和损失函数
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.TripletMarginLoss(margin=1.0)
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (anchors, positives, negatives) in enumerate(train_loader):
                # 移动到设备
                anchors = anchors.to(self.device)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)
                
                # 前向传播
                _, a_proj = self.model(anchors)
                _, p_proj = self.model(positives)
                _, n_proj = self.model(negatives)
                
                # 计算损失
                loss = criterion(a_proj, p_proj, n_proj)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 每100个批次打印进度
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}')
            
            # 保存检查点
            torch.save(self.model.state_dict(), f'contrastive_model_epoch{epoch+1}.pth')
            print(f'Epoch [{epoch+1}/{epochs}] Avg Loss: {total_loss/len(train_loader):.4f}')
def prepare_dataloaders(data_dir='./data', batch_size=64, num_workers=4):
    """加载Copydays数据并构造三元组DataLoader"""
    # 加载原始数据
    from utils import load_data
    data = load_data('copydays', data_dir)
    
    # 转换为张量格式
    query_tensors = [img for img in data['query_images']]
    db_tensors = [img for img in data['db_images']]
    
    # 创建三元组数据集
    triplet_dataset = CopyDaysTripletDataset(
        query_images=query_tensors,
        db_images=db_tensors,
        positives=data['positives'],
        num_negatives=5
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 确保最后一个不完整批次被丢弃
    )
    
    return train_loader

def train_contrastive_model(model, data_loader, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.TripletMarginLoss(margin=1.0)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for anchor, positive, negative in data_loader:
            anchor = anchor.to(model.device)
            positive = positive.to(model.device)
            negative = negative.to(model.device)
            
            _, a_proj = model(anchor)
            _, p_proj = model(positive)
            _, n_proj = model(negative)
            
            loss = criterion(a_proj, p_proj, n_proj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader):.4f}")

def tensor_hash(features, rank=32):
    """张量分解哈希生成"""
    # 将特征重组为三维张量 (样本数 x H x W)
    tensor = features.reshape(-1, 8, 8)  # 假设原始特征维度为512
    
    # 使用CP分解
    factors = parafac(tensor, rank=rank)
    core_factor = factors[1]  # 取核心因子
    
    # 生成二进制哈希码
    median = np.median(core_factor)
    hash_code = (core_factor > median).astype(int)
    return hash_code.flatten()

def contrastive_hash(img, hash_size=64):
    """对比学习哈希生成"""
    if not hasattr(contrastive_hash, 'model'):
        # 加载预训练模型
        model = ContrastiveHashModel(hash_size=hash_size)
        model.load_state_dict(torch.load('./models/contrastive_hash.pth'))
        model.eval()
        contrastive_hash.model = model
    
    # 提取特征
    img_tensor = preprocess(img).unsqueeze(0).to(model.device)
    _, hashes = contrastive_hash.model(img_tensor)
    return (hashes.cpu().detach().numpy() > 0.5).astype(int).flatten()

def compute_contrastive_distance(hash1, hash2):
    """计算汉明距离"""
    return np.sum(np.bitwise_xor(hash1, hash2)) / len(hash1)


# 数据加载器需要生成三元组（锚点、正样本、负样本）
from datasets import load_triplet_loader

# 初始化模型
# model = ContrastiveHashModel()
learner = ContrastiveLearner()
train_loader = load_triplet_loader(data_dir='./data')

# 微调模型
train_contrastive_model(model, train_loader, epochs=1)
torch.save(model.state_dict(), './models/contrastive_hash.pth')
def extract_tensor_features(model, images):
    features = []
    with torch.no_grad():
        for img in images:
            img = img.unsqueeze(0).to(model.device)
            feat, _ = model(img)
            features.append(feat.cpu().numpy())
    return np.stack(features)
def generate_tensor_hash(features):
    """多尺度张量分解哈希"""
    hashes = []
    for scale in [8, 16, 32]:  # 多尺度分解
        tensor = features.reshape(-1, scale, scale)
        factors = parafac(tensor, rank=scale//2)
        core = factors[1].flatten()
        hashes.extend((core > np.median(core)).astype(int))
    return np.array(hashes)
