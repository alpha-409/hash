import os
import pickle
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def load_data(dataset, data_dir='./data', transform=None, simulate_images=True):
    """
    加载数据集
    
    参数:
        dataset (str): 数据集名称，例如 'copydays'
        data_dir (str): 数据目录
        transform (callable, optional): 图像预处理转换
        simulate_images (bool): 当图像文件不存在时，是否生成模拟图像
        
    返回:
        dict: 包含数据集信息的字典
    """
    # 确保数据目录是绝对路径
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)
    
    print(f"使用数据目录: {data_dir}")
    
    if dataset.lower() == 'copydays':
        return load_copydays(data_dir, transform, simulate_images)
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

def load_copydays(data_dir='./data', transform=None, simulate_images=True):
    """
    加载 Copydays 数据集
    
    参数:
        data_dir (str): 数据目录
        transform (callable, optional): 图像预处理转换
        simulate_images (bool): 当图像文件不存在时，是否生成模拟图像
        
    返回:
        dict: 包含 Copydays 数据集信息的字典
    """
    # 设置默认的图像转换
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    # 加载 ground truth 数据
    gnd_path = os.path.join(data_dir, 'copydays', 'gnd_copydays.pkl')
    
    # 检查 ground truth 文件是否存在
    if not os.path.exists(gnd_path):
        raise FileNotFoundError(f"找不到 ground truth 文件: {gnd_path}")
    
    try:
        with open(gnd_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise Exception(f"加载 ground truth 文件时出错: {e}")
    
    # 检查数据格式
    required_keys = ['gnd', 'imlist', 'qimlist']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"ground truth 数据缺少必要的键: {key}")
    
    gnd = data['gnd']
    imlist = data['imlist']
    qimlist = data['qimlist']
    
    # 图像目录 - 修改为正确的图像目录
    img_dir = os.path.join(data_dir, 'copydays', 'jpg')
    
    # 检查图像目录是否存在
    if not os.path.exists(img_dir):
        print(f"警告: 图像目录 {img_dir} 不存在!")
        if simulate_images:
            print("将使用模拟图像代替。")
        else:
            os.makedirs(img_dir, exist_ok=True)
            print(f"已创建图像目录: {img_dir}")
    
    # 加载查询图像
    query_images = []
    query_paths = []
    missing_query_images = 0
    
    for q_idx, q_name in enumerate(qimlist):
        img_path = os.path.join(img_dir, q_name + '.jpg')
        query_paths.append(img_path)
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                if transform:
                    img = transform(img)
                query_images.append(img)
            except Exception as e:
                print(f"加载图像 {img_path} 时出错: {e}")
                missing_query_images += 1
                if simulate_images:
                    # 创建一个随机图像作为替代
                    img_tensor = torch.randn(3, 224, 224)
                    query_images.append(img_tensor)
        else:
            missing_query_images += 1
            if simulate_images:
                # 创建一个随机图像作为替代
                img_tensor = torch.randn(3, 224, 224)
                query_images.append(img_tensor)
    
    if missing_query_images > 0:
        print(f"警告: {missing_query_images}/{len(qimlist)} 个查询图像文件不存在或无法加载。")
        if simulate_images:
            print(f"已生成 {missing_query_images} 个模拟图像作为替代。")
    
    # 加载数据库图像
    db_images = []
    db_paths = []
    missing_db_images = 0
    
    for db_idx, db_name in enumerate(imlist):
        img_path = os.path.join(img_dir, db_name + '.jpg')
        db_paths.append(img_path)
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                if transform:
                    img = transform(img)
                db_images.append(img)
            except Exception as e:
                print(f"加载图像 {img_path} 时出错: {e}")
                missing_db_images += 1
                if simulate_images:
                    # 创建一个随机图像作为替代
                    img_tensor = torch.randn(3, 224, 224)
                    db_images.append(img_tensor)
        else:
            missing_db_images += 1
            if simulate_images:
                # 创建一个随机图像作为替代
                img_tensor = torch.randn(3, 224, 224)
                db_images.append(img_tensor)
    
    if missing_db_images > 0:
        print(f"警告: {missing_db_images}/{len(imlist)} 个数据库图像文件不存在或无法加载。")
        if simulate_images:
            print(f"已生成 {missing_db_images} 个模拟图像作为替代。")
    
    # 如果成功加载了图像，则转换为张量
    if query_images:
        query_images = torch.stack(query_images) if isinstance(query_images[0], torch.Tensor) else query_images
    else:
        print("警告: 没有加载任何查询图像!")
    
    if db_images:
        db_images = torch.stack(db_images) if isinstance(db_images[0], torch.Tensor) else db_images
    else:
        print("警告: 没有加载任何数据库图像!")
    
    # 构建正样本和负样本对
    positives = []  # 存储(查询索引, 正样本索引)对
    
    for q_idx, variants in enumerate(gnd):
        # 对于每个查询图像，收集所有变形版本作为正样本
        for variant_type in ['strong', 'crops', 'jpegqual']:
            for db_idx in variants[variant_type]:
                positives.append((q_idx, db_idx))
    
    return {
        'query_images': query_images,
        'db_images': db_images,
        'query_paths': query_paths,
        'db_paths': db_paths,
        'gnd': gnd,
        'imlist': imlist,
        'qimlist': qimlist,
        'positives': positives
    }