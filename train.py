import torch
import torch.nn as nn
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from utils import load_data, build_dataloader  # 假设utils.py在相同目录

# MoCo配置参数
class Config:
    dim = 128                 # 特征维度
    K = 65536                 # 队列大小
    m = 0.999                 # 动量系数
    temperature = 0.07        # 温度系数
    batch_size = 128          # 批大小
    num_workers = 8           # 数据加载线程数
    epochs = 100              # 训练轮数
    lr = 0.03                 # 学习率

# 数据增强策略（MoCo v2）
class MoCoAugment:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # 颜色抖动
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),               # 随机灰度化
            transforms.RandomHorizontalFlip(),               # 水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]  # 生成两个增强视图

# MoCo模型定义
class MoCo(nn.Module):
    def __init__(self, base_encoder, config):
        super(MoCo, self).__init__()
        self.config = config
        
        # 初始化编码器
        self.encoder_q = base_encoder(num_classes=config.dim)
        self.encoder_k = base_encoder(num_classes=config.dim)

        # 冻结键编码器的梯度
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        # 创建队列
        self.register_buffer("queue", torch.randn(config.dim, config.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新键编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.config.m + param_q.data * (1. - self.config.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列中的旧数据
        if ptr + batch_size > self.config.K:
            self.queue[:, ptr:] = keys.T[:, :self.config.K-ptr]
            self.queue[:, :(ptr + batch_size - self.config.K)] = keys.T[:, self.config.K-ptr:]
        else:
            self.queue[:, ptr:ptr+batch_size] = keys.T
        ptr = (ptr + batch_size) % self.config.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        # 计算查询特征
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # 计算键特征
        with torch.no_grad():
            self._momentum_update_key_encoder()  # 更新键编码器
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # 计算对比损失
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # 正样本相似度
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # 负样本相似度

        logits = torch.cat([l_pos, l_neg], dim=1) / self.config.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss()(logits, labels)

        # 更新队列
        self._dequeue_and_enqueue(k)

        return loss

# 自定义ResNet编码器
class ResNetEncoder(models.ResNet):
    def __init__(self, block, layers, num_classes=128):
        super(ResNetEncoder, self).__init__(block, layers)
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 修改最后一层

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18(**kwargs):
    return ResNetEncoder(models.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)

# 训练函数
def train_moco(config):
    # 加载数据
    data_dict = load_data('copydays', num_workers=config.num_workers)
    
    # 创建数据加载器
    transform = MoCoAugment()
    dataset = ContrastiveDatasetMoCo(data_dict, transform=transform)
    loader = DataLoader(dataset, batch_size=config.batch_size, 
                       shuffle=True, num_workers=config.num_workers,
                       pin_memory=True, drop_last=True)
    
    # 初始化模型
    model = MoCo(resnet18, config).cuda()
    optimizer = torch.optim.SGD(model.parameters(), config.lr,
                               momentum=0.9, weight_decay=1e-4)
    scaler = GradScaler()
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        
        for im_q, im_k in tqdm(loader, desc=f'Epoch {epoch+1}/{config.epochs}'):
            im_q = im_q.cuda(non_blocking=True)
            im_k = im_k.cuda(non_blocking=True)

            optimizer.zero_grad()
            
            with autocast():
                loss = model(im_q, im_k)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * im_q.size(0)
        
        avg_loss = total_loss / len(loader.dataset)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.encoder_q.state_dict(), 'best_moco.pth')
    
    print(f"Training complete. Best loss: {best_loss:.4f}")

# MoCo专用数据集类
class ContrastiveDatasetMoCo(Dataset):
    def __init__(self, data_dict, transform=None):
        self.all_images = torch.cat([data_dict['query_images'], 
                                   data_dict['db_images']])
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img = self.all_images[idx]
        if self.transform:
            views = self.transform(img)
        else:
            views = [img, img]
        return views[0], views[1]

# 特征提取函数
class FeatureExtractor:
    def __init__(self, ckpt_path='best_moco.pth'):
        self.model = resnet18(num_classes=Config.dim)
        state_dict = torch.load(ckpt_path)
        self.model.load_state_dict(state_dict)
        self.model.eval().cuda()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            feature = self.model(img)
        return feature.squeeze().cpu().numpy()

if __name__ == '__main__':
    config = Config()
    train_moco(config)
    
    # 使用示例
    extractor = FeatureExtractor()
    features = extractor.extract('test_image.jpg')
    print(f"提取特征维度: {features.shape}")