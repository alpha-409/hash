import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# 导入自定义模块
from model_code import HashNet, TripletHashLoss
from training_code import HashDataset, collate_with_positives, train
from utils import load_data

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 设置随机种子
    set_seed(42)
    
    # 配置参数
    config = {
        'hash_length': 128,        # 哈希码长度
        'cp_rank': 32,             # CP分解的秩
        'batch_size': 32,          # 批次大小
        'num_epochs': 50,          # 训练周期数
        'learning_rate': 0.001,    # 学习率
        'weight_decay': 1e-5,      # 权重衰减
        'lambda1': 1.0,            # 相似性保持损失权重
        'lambda2': 0.5,            # 量化损失权重
        'lambda3': 0.1,            # 比特平衡损失权重
        'save_dir': 'checkpoints', # 模型保存目录
        'data_dir': 'copydays',    # 数据目录
        'detail_channels': 64,     # 细节路径通道数
        'semantic_channels': 512,  # 语义路径通道数
    }
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据
    print("正在加载数据...")
    data_dict = load_data(config['data_dir'])
    
    # 创建数据集
    train_dataset = HashDataset(data_dict, transform=transform, is_query=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_with_positives,
        pin_memory=True
    )
    
    # 初始化模型
    model = HashNet(
        hash_length=config['hash_length'],
        cp_rank=config['cp_rank'],
        detail_channels=config['detail_channels'],
        semantic_channels=config['semantic_channels']
    ).to(device)
    
    # 初始化损失函数
    criterion = TripletHashLoss(
        lambda1=config['lambda1'],
        lambda2=config['lambda2'],
        lambda3=config['lambda3']
    )
    
    # 初始化优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # 训练记录
    train_losses = []
    loss_components_history = {
        'similarity_loss': [],
        'quantization_loss': [],
        'bit_balance_loss': []
    }
    
    # 开始训练
    print(f"开始训练，共{config['num_epochs']}个周期...")
    best_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        # 训练一个周期
        epoch_loss, loss_components = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=config['num_epochs']
        )
        
        # 更新学习率
        scheduler.step(epoch_loss)
        
        # 记录损失
        train_losses.append(epoch_loss)
        for key, value in loss_components.items():
            if key in loss_components_history:
                loss_components_history[key].append(value)
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"保存最佳模型，损失: {best_loss:.4f}")
        
        # 每10个周期保存一次检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pth'))
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 8))
    
    # 总损失
    plt.subplot(2, 1, 1)
    plt.plot(range(1, config['num_epochs'] + 1), train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 损失组件
    plt.subplot(2, 1, 2)
    for key, values in loss_components_history.items():
        plt.plot(range(1, config['num_epochs'] + 1), values, label=key)
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.close()
    
    print("训练完成！")
    
    # 加载最佳模型进行评估
    print("\n加载最佳模型进行评估...")
    best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    
    # 创建测试数据集和数据加载器
    test_dataset = HashDataset(data_dict, transform=transform, is_query=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_with_positives
    )
    
    # 评估函数
    def evaluate(model, test_loader, device, data_dict):
        model.eval()
        query_hashes = []
        query_indices = []
        db_hashes = []
        db_indices = []
        
        # 获取查询集和数据库集的哈希码
        with torch.no_grad():
            # 处理查询集
            query_dataset = HashDataset(data_dict, transform=transform, is_query=True)
            query_loader = DataLoader(
                query_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4,
                collate_fn=collate_with_positives
            )
            
            for batch in tqdm(query_loader, desc="处理查询集"):
                images = batch['images'].to(device)
                indices = batch['indices']
                binary_hash, _, _ = model(images, training=False)
                query_hashes.append(binary_hash.cpu())
                query_indices.extend(indices)
            
            # 处理数据库集
            db_dataset = HashDataset(data_dict, transform=transform, is_query=False)
            db_loader = DataLoader(
                db_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4,
                collate_fn=collate_with_positives
            )
            
            for batch in tqdm(db_loader, desc="处理数据库集"):
                images = batch['images'].to(device)
                indices = batch['indices']
                binary_hash, _, _ = model(images, training=False)
                db_hashes.append(binary_hash.cpu())
                db_indices.extend(indices)
        
        # 合并所有哈希码
        query_hashes = torch.cat(query_hashes, dim=0)
        db_hashes = torch.cat(db_hashes, dim=0)
        
        # 计算汉明距离矩阵
        dist_matrix = 0.5 * (config['hash_length'] - query_hashes @ db_hashes.T)
        
        # 计算mAP和μAP
        aps = []
        precisions = {k: [] for k in [1, 5, 10]}  # 计算top-1, top-5, top-10的precision
        
        for i in range(len(query_indices)):
            q_idx = query_indices[i]
            # 获取真实相关图像
            relevant = [p[1] for p in data_dict['positives'] if p[0] == q_idx]
            if not relevant:
                continue
                
            # 获取排序后的检索结果
            sorted_indices = np.argsort(dist_matrix[i])
            retrieved = [db_indices[j] for j in sorted_indices]
            
            # 计算AP
            ap = 0.0
            correct = 0
            for k in range(len(retrieved)):
                if retrieved[k] in relevant:
                    correct += 1
                    ap += correct / (k + 1)
            ap /= len(relevant)
            aps.append(ap)
            
            # 计算precision@k
            for k in precisions.keys():
                top_k = retrieved[:k]
                precisions[k].append(len([x for x in top_k if x in relevant]) / k)
        
        # 计算mAP和μAP
        mAP = np.mean(aps) if aps else 0.0
        μAP = sum(aps) / len(aps) if aps else 0.0
        
        # 计算平均precision@k
        avg_precisions = {k: np.mean(v) if v else 0.0 for k, v in precisions.items()}
        
        return {
            'mAP': mAP,
            'μAP': μAP,
            'precisions': avg_precisions
        }
    
    # 执行评估
    eval_results = evaluate(model, test_loader, device, data_dict)
    print(f"\n评估结果:")
    print(f"mAP: {eval_results['mAP']:.4f}")
    print(f"μAP: {eval_results['μAP']:.4f}")
    for k, prec in eval_results['precisions'].items():
        print(f"Precision@{k}: {prec:.4f}")

if __name__ == "__main__":
    main()