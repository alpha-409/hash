#!/bin/bash

# 设置要使用的算法
algorithms="all"

# 设置要执行的 hash_size 列表
hash_sizes=(4 8 16 32 64 128 256 1024)

# 指定输出文件
output_file="output.txt"

# 清空输出文件（如果存在）
> $output_file

# 遍历每个 hash_size 并执行命令
for hash_size in "${hash_sizes[@]}"; do
    echo "Executing for hash_size=${hash_size}..." | tee -a $output_file
    python evaluate_hashes.py --algorithms $algorithms --hash_size $hash_size | tee -a $output_file
done

echo "All tasks completed." | tee -a $output_file
