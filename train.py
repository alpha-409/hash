import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from evaluate import evaluate_model
from tqdm import tqdm
from screen_hash_model import ScreenHashModel, ScreenHashLoss

def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4):
    # 初始化模型和损失
    criterion = ScreenHashLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):  # 使用传入的train_loader
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            binary_hash, hash_logits, _ = model(imgs)
            
            # 计算损失
            loss, loss_dict = criterion(hash_logits, binary_hash, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(dataloader)} | '
                     f'Loss: {loss.item():.4f} | Sim: {loss_dict["sim_loss"]:.4f} | '
                     f'Quant: {loss_dict["quant_loss"]:.4f} | Balance: {loss_dict["balance_loss"]:.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}')
    
    return model


# 修改损失函数，增加对高相似性样本的区分能力
class ScreenHashLoss(nn.Module):
    def __init__(self, lambda_sim=1.0, lambda_quant=0.5, lambda_balance=0.1, margin=1.0):
        super().__init__()
        self.lambda_sim = lambda_sim
        self.lambda_quant = lambda_quant
        self.lambda_balance = lambda_balance
        self.margin = margin
        
    def forward(self, hash_logits, binary_hash, labels):
        B = binary_hash.size(0)
        
        # 增强的相似性保持损失
        sim_loss = 0
        pos_pairs = 0
        neg_pairs = 0
        
        for i in range(B):
            for j in range(B):
                if i != j:
                    dot_product = torch.dot(binary_hash[i], binary_hash[j])
                    if labels[i,j] == 1:  # 正样本对
                        # 对正样本使用更严格的约束
                        sim_loss += torch.clamp(0.9 - dot_product, min=0)**2  # 提高正样本相似度阈值
                        pos_pairs += 1
                    else:  # 负样本对
                        sim_loss += torch.clamp(dot_product + 0.1, min=0)**2  # 增加负样本分离度
                        neg_pairs += 1
        
        if pos_pairs > 0:
            sim_loss = sim_loss / (pos_pairs + neg_pairs)
        
        # 其余损失保持不变
        quant_loss = torch.mean(torch.abs(torch.abs(hash_logits) - 1))
        balance_loss = torch.mean(binary_hash, dim=0).pow(2).sum()
        
        total_loss = (self.lambda_sim * sim_loss + 
                     self.lambda_quant * quant_loss + 
                     self.lambda_balance * balance_loss)
        
        return total_loss