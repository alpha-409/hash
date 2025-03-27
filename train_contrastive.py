import torch
from torch.utils.data import DataLoader
from contrastive_model import ContrastiveHash, ContrastiveLoss
from utils import build_dataloader
def train_contrastive(data_dict, hash_dim=64, epochs=10):
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContrastiveHash(hash_dim=hash_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 数据加载
    dataloader = build_dataloader(data_dict)
    
    # 训练循环
    for epoch in range(epochs):
        for batch in dataloader:
            anchors, others, labels = batch
            anchors = anchors.to(device)
            others = others.to(device)
            
            # 获取特征
            anchor_feat = model(anchors, return_hash=False)
            other_feat = model(others, return_hash=False)
            
            # 计算损失
            loss = criterion(anchor_feat, other_feat[labels==1], other_feat[labels==0])
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    return model