import pickle
import os

# 指定文件路径
file_path = './data/copydays/gnd_copydays.pkl'  # 请确保文件路径正确

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"文件 {file_path} 不存在！")
else:
    # 读取 pickle 文件
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 打印数据结构信息
        print(f"数据类型: {type(data)}")
        
        # 如果是字典类型，打印键
        if isinstance(data, dict):
            print(f"字典键: {list(data.keys())}")
            
            # 分析每个键的内容
            for key in data.keys():
                print(f"\n{key} 的类型: {type(data[key])}")
                
                if key == 'gnd':
                    print(f"{key} 的长度: {len(data[key])}")
                    print(f"{key} 的前3个元素: {data[key][:3]}")
                elif key == 'imlist' or key == 'qimlist':
                    print(f"{key} 的长度: {len(data[key])}")
                    print(f"{key} 的前5个元素: {data[key][:5]}")
        
    except Exception as e:
        print(f"读取文件时出错: {e}")