from utils import *
from train_contrastive  import train_contrastive

def evaluate_contrastive(model, data, use_tensor_decomposition=False):
    """
    评估对比学习模型
    参数:
        use_tensor_decomposition: 是否使用张量分解生成哈希
    """
    model.eval()
    
    # 提取所有特征
    with torch.no_grad():
        db_features = [model(img, return_hash=False) for img in data['db_images']]
        query_features = [model(img, return_hash=False) for img in data['query_images']]
    
    # 生成哈希
    if use_tensor_decomposition:
        db_hash = tensor_decomposition_hash(torch.stack(db_features))
        query_hash = tensor_decomposition_hash(torch.stack(query_features))
    else:
        db_hash = model(torch.stack(data['db_images']), return_hash=True)
        query_hash = model(torch.stack(data['query_images']), return_hash=True)
    
    # 计算mAP
    return evaluate_hash(lambda x: x, data, hash_size=64, 
                        distance_func=lambda a,b: (a != b).float().mean(dim=1))
data = load_data('copydays', './data')

# 训练对比模型
model = train_contrastive(data,epochs=1)

# 评估两种方式
print("直接哈希结果:")
evaluate_contrastive(model, data, use_tensor_decomposition=False)

print("\n张量分解哈希结果:")
evaluate_contrastive(model, data, use_tensor_decomposition=True)