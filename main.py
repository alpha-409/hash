import torch
from torch.utils.data import Dataset, DataLoader
from screen_hash_model import ScreenHashModel  # 修改导入路径
from utils import load_data
from train import train_model
from evaluate import evaluate_model

class HashDataset(Dataset):
    def __init__(self, data):
        self.images = data['db_images']
        self.labels = self._generate_labels(data)
        
    def _generate_labels(self, data):
        # 为每张图像生成唯一标签
        labels = torch.zeros(len(self.images))
        for i, (q_idx, db_idx) in enumerate(data['positives']):
            labels[db_idx] = q_idx  # 相同查询的图像共享标签
        return labels.long()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def main():
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScreenHashModel().to(device)
    
    # 加载数据
    data = load_data('copydays')  # 或其他数据集
    dataset = HashDataset(data)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 训练模型
    trained_model = train_model(model, train_loader, val_loader, device)
    
    # 最终评估
    test_mAP, test_μAP = evaluate_model(trained_model, val_loader, device)
    print(f"最终测试结果 - mAP: {test_mAP:.4f}, μAP: {test_μAP:.4f}")

if __name__ == '__main__':
    main()