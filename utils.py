import os
import pickle
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import concurrent.futures
from tqdm import tqdm  # 添加进度条支持
import multiprocessing  # 添加这一行来获取系统CPU核心数
import time  # 添加时间模块

def load_data(dataset, data_dir='./data', transform=None, simulate_images=True, num_workers=None):
    """
    加载数据集
    
    参数:
        dataset (str): 数据集名称，例如 'copydays'
        data_dir (str): 数据目录
        transform (callable, optional): 图像预处理转换
        simulate_images (bool): 当图像文件不存在时，是否生成模拟图像
        num_workers (int, optional): 用于并行加载图像的工作线程数，默认使用系统可用的最大线程数
        
    返回:
        dict: 包含数据集信息的字典
    """
    # 开始计时
    start_time = time.time()
    
    # 如果未指定线程数，则使用系统可用的最大线程数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        print(f"自动使用系统可用的最大线程数: {num_workers}")
    
    # 确保数据目录是绝对路径
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)
    
    print(f"使用数据目录: {data_dir}")
    
    # 加载数据集
    if dataset.lower() == 'copydays':
        result = load_copydays(data_dir, transform, simulate_images, num_workers)
    else:
        raise ValueError(f"不支持的数据集: {dataset}")
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"数据集 '{dataset}' 加载完成，总耗时: {total_time:.2f}秒")
    
    return result

def load_copydays(data_dir='./data', transform=None, simulate_images=True, num_workers=None):
    """
    加载 Copydays 数据集
    
    参数:
        data_dir (str): 数据目录
        transform (callable, optional): 图像预处理转换
        simulate_images (bool): 当图像文件不存在时，是否生成模拟图像
        num_workers (int, optional): 用于并行加载图像的工作线程数，默认使用系统可用的最大线程数
        
    返回:
        dict: 包含 Copydays 数据集信息的字典
    """
    # 开始计时
    start_time = time.time()
    
    # 如果未指定线程数，则使用系统可用的最大线程数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
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
    gnd_start_time = time.time()
    gnd_path = os.path.join(data_dir, 'copydays', 'gnd_copydays.pkl')
    
    # 检查 ground truth 文件是否存在
    if not os.path.exists(gnd_path):
        raise FileNotFoundError(f"找不到 ground truth 文件: {gnd_path}")
    
    try:
        with open(gnd_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise Exception(f"加载 ground truth 文件时出错: {e}")
    
    gnd_time = time.time() - gnd_start_time
    print(f"Ground truth 数据加载耗时: {gnd_time:.2f}秒")
    
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
    
    # 定义图像加载函数
    def load_image(img_info):
        idx, name, is_query = img_info
        img_path = os.path.join(img_dir, name + '.jpg')
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                if transform:
                    img = transform(img)
                return idx, img, img_path, False  # 成功加载
            except Exception as e:
                if not is_query:
                    print(f"加载图像 {img_path} 时出错: {e}")
                if simulate_images:
                    # 创建一个随机图像作为替代
                    img_tensor = torch.randn(3, 224, 224)
                    return idx, img_tensor, img_path, True  # 使用模拟图像
                return idx, None, img_path, True  # 加载失败
        else:
            if simulate_images:
                # 创建一个随机图像作为替代
                img_tensor = torch.randn(3, 224, 224)
                return idx, img_tensor, img_path, True  # 使用模拟图像
            return idx, None, img_path, True  # 图像不存在
    
    # 准备查询图像任务
    query_tasks = [(idx, name, True) for idx, name in enumerate(qimlist)]
    
    # 准备数据库图像任务
    db_tasks = [(idx, name, False) for idx, name in enumerate(imlist)]
    
    # 初始化结果列表
    query_images = [None] * len(qimlist)
    query_paths = [None] * len(qimlist)
    db_images = [None] * len(imlist)
    db_paths = [None] * len(imlist)
    
    missing_query_images = 0
    missing_db_images = 0
    
    # 多线程加载查询图像
    query_start_time = time.time()
    print(f"\n▶ 正在使用 {num_workers} 个线程加载查询图像...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_image, task) for task in query_tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(query_tasks)):
            idx, img, path, is_missing = future.result()
            query_images[idx] = img
            query_paths[idx] = path
            if is_missing:
                missing_query_images += 1
    query_time = time.time() - query_start_time
    print(f"查询图像加载耗时: {query_time:.2f}秒 ({len(query_tasks)}张图像)")
    
    # 多线程加载数据库图像
    db_start_time = time.time()
    print(f"\n▶ 正在使用 {num_workers} 个线程加载数据库图像...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_image, task) for task in db_tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(db_tasks)):
            idx, img, path, is_missing = future.result()
            db_images[idx] = img
            db_paths[idx] = path
            if is_missing:
                missing_db_images += 1
    db_time = time.time() - db_start_time
    print(f"数据库图像加载耗时: {db_time:.2f}秒 ({len(db_tasks)}张图像)")
    
    # 移除None值（加载失败且不模拟的情况）
    query_images = [img for img in query_images if img is not None]
    db_images = [img for img in db_images if img is not None]
    
    if missing_query_images > 0:
        print(f"警告: {missing_query_images}/{len(qimlist)} 个查询图像文件不存在或无法加载。")
        if simulate_images:
            print(f"已生成 {missing_query_images} 个模拟图像作为替代。")
    
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
    positive_start_time = time.time()
    positives = []  # 存储(查询索引, 正样本索引)对
    
    for q_idx, variants in enumerate(gnd):
        # 对于每个查询图像，收集所有变形版本作为正样本
        for variant_type in ['strong', 'crops', 'jpegqual']:
            for db_idx in variants[variant_type]:
                positives.append((q_idx, db_idx))
    
    positive_time = time.time() - positive_start_time
    print(f"正样本对构建耗时: {positive_time:.2f}秒 (共{len(positives)}对)")
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"Copydays 数据集加载完成，总耗时: {total_time:.2f}秒")
    print(f"- 查询图像: {len(query_images)}张")
    print(f"- 数据库图像: {len(db_images)}张")
    print(f"- 正样本对: {len(positives)}对")
    
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
