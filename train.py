import torch
import torch.nn as nn
from torchvision import models

class ContrastiveModel(nn.Module):
    def __init__(self, base_model='resnet50', projection_dim=128):
        super(ContrastiveModel, self).__init__()
        # 加载预训练的ResNet50
        self.base_model = models.resnet50(weights='IMAGENET1K_V1')
        self.base_model.fc = nn.Identity()  # 移除全连接层
        # 投影头：将2048维特征映射到128维
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        features = self.base_model(x)  # 输出2048维特征
        projections = self.projection_head(features)  # 输出128维投影
        return features, projections

from torchvision import transforms

# 数据增强，用于生成正样本
data_augment = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 标准预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据加载（简化为示例，实际需调用load_copydays）
def prepare_data(data_dir='./data'):
    from utils import load_data
    data = load_data('copydays', data_dir, transform=preprocess)
    return data['query_images'], data['db_images'], data['positives']
import torch.nn.functional as F

def info_nce_loss(proj1, proj2, temperature=0.1):
    # 计算相似度矩阵
    proj1 = F.normalize(proj1, dim=1)
    proj2 = F.normalize(proj2, dim=1)
    similarity = torch.matmul(proj1, proj2.T) / temperature
    labels = torch.arange(proj1.size(0)).to(proj1.device)
    return F.cross_entropy(similarity, labels)

def train_contrastive_model(model, query_images, db_images, positives, epochs=10, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(positives), batch_size):
            batch_pos = positives[i:i+batch_size]
            img1_list, img2_list = [], []
            
            for q_idx, db_idx in batch_pos:
                img1 = query_images[q_idx]
                img2 = db_images[db_idx]
                # 对正样本应用数据增强
                img1_aug = data_augment(transforms.ToPILImage()(img1))
                img2_aug = data_augment(transforms.ToPILImage()(img2))
                img1_list.append(img1_aug)
                img2_list.append(img2_aug)
            
            img1_batch = torch.stack(img1_list).to(model.device)
            img2_batch = torch.stack(img2_list).to(model.device)
            
            _, proj1 = model(img1_batch)
            _, proj2 = model(img2_batch)
            
            loss = info_nce_loss(proj1, proj2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
def extract_features(model, images):
    model.eval()
    features = []
    with torch.no_grad():
        for img in images:
            img = img.unsqueeze(0).to(model.device)
            feat, _ = model(img)
            features.append(feat.cpu().numpy().flatten())
    return np.array(features)
import numpy as np
from scipy.linalg import svd

def generate_hash_codes(features, hash_size=64):
    # features: N x 2048 矩阵
    U, S, Vt = svd(features, full_matrices=False)
    # 保留前hash_size个奇异向量
    Uk = U[:, :hash_size]
    # 生成二进制哈希码
    hash_codes = (Uk > 0).astype(int)
    return hash_codes
from sklearn.metrics import hamming_loss

def detect_copies(query_hash, db_hashes):
    distances = hamming_loss(query_hash.reshape(1, -1), db_hashes)[0]
    sorted_indices = np.argsort(distances)
    return sorted_indices, distances
def main():
    # 数据准备
    query_images, db_images, positives = prepare_data(data_dir='./data')
    
    # 初始化模型
    model = ContrastiveModel()
    
    # 训练模型
    train_contrastive_model(model, query_images, db_images, positives, epochs=10)
    
    # 提取特征
    query_features = extract_features(model, query_images)
    db_features = extract_features(model, db_images)
    
    # 生成哈希码
    all_features = np.vstack((query_features, db_features))
    hash_codes = generate_hash_codes(all_features)
    query_hashes = hash_codes[:len(query_images)]
    db_hashes = hash_codes[len(query_images):]
    
    # 示例：检测第一个查询图像的拷贝
    indices, distances = detect_copies(query_hashes[0], db_hashes)
    print(f"Top 5 similar images (indices): {indices[:5]}")
    print(f"Distances: {distances[indices[:5]]}")

if __name__ == "__main__":
    main()