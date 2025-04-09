#!/bin/bash

# 定义数据集列表
# datasets=("copydays" "copy2" "scid")
datasets=("copydays" "scid")
# 定义种子列表（1到5）
seeds=(1 2 3 4 5)

# 定义输出日志文件
log_file="main_runs.log"

# 清空日志文件（如果存在）
> $log_file

echo "开始运行实验..." | tee -a $log_file
echo "-----------------------------------" | tee -a $log_file

# 遍历每个数据集
for dataset in "${datasets[@]}"; do
    # 遍历每个种子值
    for seed in "${seeds[@]}"; do
        echo "运行: dataset=${dataset}, seed=${seed}" | tee -a $log_file
        echo "命令: python main.py --dataset ${dataset} --epochs 500 --eval-epoch 1 --seed ${seed}" | tee -a $log_file
        
        # 执行命令并将输出记录到日志文件
        python main.py --dataset ${dataset} --epochs 500 --eval-epoch 1 --seed ${seed} | tee -a $log_file
        
        echo "-----------------------------------" | tee -a $log_file
    done
done

echo "所有实验已完成。" | tee -a $log_file
