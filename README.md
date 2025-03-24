# 图像哈希拷贝检测评估工具

这个工具用于评估不同图像哈希算法在图像拷贝检测任务上的性能。它支持多种哈希方法（ahash、dhash、phash），并提供详细的评估指标。

## 运行环境要求

- Python 3.6+
- CUDA支持（可选，用于GPU加速）
- 必要的Python包（建议使用requirements.txt安装）

## 数据集结构

数据集应按以下结构组织：
```
data/
└── copydays/
    ├── jpg/              # 原始图像
    │   └── *.jpg
    └── gnd_copydays.pkl  # Ground truth数据
```

## 命令行参数

```bash
python run_evaluation.py [选项]
```

支持的选项：
- `--device`: 计算设备选择
  - 可选值：'cpu' 或 'cuda'
  - 默认值：'cuda'
  - 示例：--device cpu

- `--hash-size`: 哈希大小
  - 类型：整数
  - 默认值：8
  - 示例：--hash-size 16

- `--threshold`: 相似度阈值
  - 类型：浮点数
  - 默认值：0.8
  - 取值范围：0.0-1.0
  - 示例：--threshold 0.75

- `--batch-size`: 批处理大小
  - 类型：整数
  - 默认值：32
  - 示例：--batch-size 64

## 使用示例

1. 使用默认参数运行评估：
```bash
python run_evaluation.py
```

2. 在CPU上运行并使用不同的哈希大小：
```bash
python run_evaluation.py --device cpu --hash-size 16
```

3. 调整相似度阈值和批处理大小：
```bash
python run_evaluation.py --threshold 0.75 --batch-size 64
```

## 输出说明

程序运行时会显示以下信息：
1. 数据集统计信息（原始图像数量、查询图像数量等）
2. 每种哈希方法的评估结果，包括：
   - mAP (mean Average Precision)
   - μAP (micro Average Precision)
   - P@k (Precision at k)
   - R@k (Recall at k)，其中k = 1,5,10

评估结果将自动保存到 `results` 目录下，文件名格式为：
```
hash_evaluation_[hash_size]_[device]_[timestamp].json
```

结果文件包含：
- 运行参数配置
- 详细的评估指标
- 每种哈希方法的性能数据

## 注意事项

1. 如果使用CUDA，请确保已正确安装NVIDIA驱动和CUDA工具包
2. 首次运行可能需要一定时间来建立图像索引
3. 结果目录会自动创建，无需手动创建
