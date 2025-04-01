import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os

# 数据增强定义
class TwoCropsTransform:
    """将图像的两个增强视图组合成一个样本"""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

# 自定义对比学习数据集
class ContrastiveDataset(Dataset):
    def __init__(self, pil_images, transform=None):
        self.images = pil_images
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform:
            views = self.transform(img)
            return views
        return img, img

    def __len__(self):
        return len(self.images)

# 修正后的MoCo模型实现
class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # 创建编码器并分离特征提取层
        self.encoder_q = base_encoder(pretrained=True)
        self.encoder_k = base_encoder(pretrained=True)

        # 分离特征提取器和全连接层
        self.feature_extractor_q = nn.Sequential(*list(self.encoder_q.children())[:-1])
        self.feature_extractor_k = nn.Sequential(*list(self.encoder_k.children())[:-1])

        # 修改全连接层为投影头
        in_features = self.encoder_q.fc.in_features
        self.projection_q = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, dim)
        )
        self.projection_k = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, dim)
        )

        # 初始化键编码器
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 创建队列
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新键编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """维护队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列中的旧数据
        if ptr + batch_size > self.K:
            self.queue[:, ptr:] = keys[:self.K-ptr].T
            self.queue[:, :(batch_size - (self.K - ptr))] = keys[self.K-ptr:].T
        else:
            self.queue[:, ptr:ptr+batch_size] = keys.T
        
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        # 特征提取
        q_features = self.feature_extractor_q(im_q).flatten(1)
        q = self.projection_q(q_features)
        q = nn.functional.normalize(q, dim=1)

        # 键特征
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_features = self.feature_extractor_k(im_k).flatten(1)
            k = self.projection_k(k_features)
            k = nn.functional.normalize(k, dim=1)

        # 计算相似度
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # 正样本
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # 负样本

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k)

        return logits, labels

# 训练函数
def train_moco(model, train_loader, optimizer, criterion, epochs=5):
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for views in progress_bar:
            im_q, im_k = views[0].cuda(), views[1].cuda()
            
            optimizer.zero_grad()
            logits, labels = model(im_q, im_k)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), './model/best_moco_query_batch157_model.pth')
            print(f"Saved new best model with loss: {best_loss:.4f}")
        if epoch == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), f'./model/moco_epoch_{epoch+1}_query_batch157_model.pth')
            print(f"Saved model at epoch {epoch+1} with loss: {avg_loss:.4f}")

# 修正后的特征提取函数
def extract_features(model, dataloader):
    model.eval()
    features = []
    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.cuda()
            feat = model.feature_extractor_q(images)
            feat = feat.flatten(1)  # 展平特征
            features.append(feat.cpu().numpy())
    return np.concatenate(features)

if __name__ == "__main__":
    # 加载数据集
    from utils import load_data
    
    # 数据预处理
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    contrastive_transform = TwoCropsTransform(transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ]))
    
    data = load_data('copydays', transform=base_transform, simulate_images=False)
    
    # 创建对比学习数据集
    to_pil = transforms.ToPILImage()
    data['query_images'] = [to_pil(img) for img in data['query_images']]
    data['db_images']= [to_pil(img) for img in data['db_images']]
    train_dataset = ContrastiveDataset(data['query_images'], transform=contrastive_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=157,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = MoCo(base_encoder=models.resnet50).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.03, weight_decay=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    train_moco(model, train_loader, optimizer, criterion, epochs=5)
    
    # 特征提取示例
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # 创建特征提取数据集
    class FeatureDataset(Dataset):
        def __init__(self, images, transform=None):
            self.images = images
            self.transform = transform
        
        def __getitem__(self, index):
            img = self.images[index]
            if self.transform:
                img = self.transform(img)
            return img
        
        def __len__(self):
            return len(self.images)
    
    query_feature_dataset = FeatureDataset(data['query_images'], test_transform)
    db_feature_dataset = FeatureDataset(data['db_images'], test_transform)
    
    query_loader = DataLoader(query_feature_dataset, batch_size=64, shuffle=False)
    db_loader = DataLoader(db_feature_dataset, batch_size=64, shuffle=False)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('./model/best_moco_query_batch157_model.pth'))
    
    # 提取特征
    query_features = extract_features(model, query_loader)
    db_features = extract_features(model, db_loader)
    
    # 保存特征
    np.save('query_features_query_batch157.npy', query_features)
    np.save('db_features_query_batch157.npy', db_features)