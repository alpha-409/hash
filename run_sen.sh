#!/bin/bash

# --- 配置 ---

# 要测试的数据集 (可以选择一个进行敏感性分析，以减少运行时间)
DATASET="copydays"
# DATASET="scid" 

# 基础输出目录 (所有敏感性分析结果将存储在此目录的子目录中)
BASE_OUTPUT_DIR="./sensitivity_analysis_output"

# 主日志文件
MAIN_LOG_FILE="${BASE_OUTPUT_DIR}/sensitivity_analysis_main.log"

# 用于敏感性分析的固定种子 (确保比较的公平性)
FIXED_SEED=42

# 训练周期 (可以减少周期数以加速敏感性分析，但结果可能不完全收敛)
EPOCHS=50 
# EPOCHS=20 # 更快的选项

# 评估频率
EVAL_EPOCH=5 
# EVAL_EPOCH=2 # 更快的选项

# 是否使用模拟图像 (如果你的数据集不完整)
SIMULATE_IMAGES_FLAG="--simulate-images" # 或 "--no-simulate-images"

# --- 基准参数 (你的默认或最佳已知参数) ---
BASE_HASH_BITS=64
BASE_CP_RANK=32
BASE_BACKBONE="resnet18"
BASE_PRETRAINED_FLAG="--pretrained" # 或 "--no-pretrained"
BASE_BATCH_SIZE=32
BASE_LR=1e-5
BASE_WD=1e-5
BASE_MARGIN=0.4
BASE_LAMBDA_SIM=1.0 # 通常固定为1.0
BASE_LAMBDA_QUANT=0.5
BASE_LAMBDA_BALANCE=0.1
BASE_NUM_WORKERS=4 # 根据你的机器调整

# --- 要测试的参数范围 ---

# 1. 学习率 (Learning Rate)
lr_list=(1e-6 5e-6 1e-5 5e-5 1e-4) 

# 2. 哈希位数 (Hash Bits)
hash_bits_list=(32 64 128) 

# 3. CP 分解秩 (CP Rank - 可能与 hash_bits 相关)
#    注意：如果测试 hash_bits，可能需要相应调整 cp_rank 或单独测试
cp_rank_list=(16 32 64) # 假设 hash_bits=64

# 4. 量化损失权重 (Lambda Quantization)
lambda_quant_list=(0.01 0.1 0.5 1.0 5.0)

# 5. 平衡损失权重 (Lambda Balance)
lambda_balance_list=(0.01 0.05 0.1 0.5 1.0)

# 6. 相似性损失边距 (Margin)
margin_list=(0.2 0.3 0.4 0.5 0.6)

# 7. 权重衰减 (Weight Decay)
wd_list=(0 1e-6 1e-5 1e-4)

# 8. 批处理大小 (Batch Size)
#    注意：更改 batch_size 可能需要调整学习率
batch_size_list=(16 32 64) 

# --- 脚本执行 ---

# 创建基础输出目录和日志文件
mkdir -p "$BASE_OUTPUT_DIR"
> "$MAIN_LOG_FILE" # 清空旧日志

echo "开始参数敏感性分析..." | tee -a "$MAIN_LOG_FILE"
echo "数据集: $DATASET" | tee -a "$MAIN_LOG_FILE"
echo "固定种子: $FIXED_SEED" | tee -a "$MAIN_LOG_FILE"
echo "训练周期: $EPOCHS" | tee -a "$MAIN_LOG_FILE"
echo "主日志文件: $MAIN_LOG_FILE" | tee -a "$MAIN_LOG_FILE"
echo "-----------------------------------" | tee -a "$MAIN_LOG_FILE"

# --- 函数：执行单次运行 ---
run_experiment() {
    # 参数: 1=参数名, 2=参数值, 3...=具体命令参数
    local param_name=$1
    local param_value=$2
    shift 2 # 移除前两个参数，剩下的是 python 命令参数

    local run_output_dir="${BASE_OUTPUT_DIR}/${DATASET}_${param_name}_${param_value}"
    mkdir -p "$run_output_dir"

    local run_log_file="${run_output_dir}/run.log"

    echo "运行: ${param_name}=${param_value}" | tee -a "$MAIN_LOG_FILE"
    echo "输出目录: ${run_output_dir}" | tee -a "$MAIN_LOG_FILE"
    
    # 构建命令
    local cmd="python main.py \
        --dataset ${DATASET} \
        --epochs ${EPOCHS} \
        --eval-epoch ${EVAL_EPOCH} \
        --seed ${FIXED_SEED} \
        --output-dir ${run_output_dir} \
        --num-workers ${BASE_NUM_WORKERS} \
        ${SIMULATE_IMAGES_FLAG} \
        $@" # 添加传递进来的特定参数

    echo "命令: $cmd" | tee -a "$MAIN_LOG_FILE"

    # 执行命令并将标准输出和错误输出都记录到运行日志和主日志
    # 使用 > "$run_log_file" 2>&1 将标准输出和错误输出重定向到运行日志文件
    # 使用 tee -a "$MAIN_LOG_FILE" 将标准输出附加到主日志文件
    # 使用管道将标准错误通过管道传递给 tee -a "$MAIN_LOG_FILE" 
    # 注意：这种复杂的重定向有时可能在所有 shell 环境中不完全一致，但通常有效
    { $cmd 2>&1 | tee "$run_log_file" ; } | tee -a "$MAIN_LOG_FILE"
    
    # 检查退出码 (可选)
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "警告: 命令执行失败，退出码 ${exit_code} (${param_name}=${param_value})" | tee -a "$MAIN_LOG_FILE"
    fi

    echo "完成: ${param_name}=${param_value}" | tee -a "$MAIN_LOG_FILE"
    echo "-----------------------------------" | tee -a "$MAIN_LOG_FILE"
}


# --- 1. 测试学习率 (lr) ---
echo "===== 测试学习率 (lr) =====" | tee -a "$MAIN_LOG_FILE"
for lr_val in "${lr_list[@]}"; do
    run_experiment "lr" "$lr_val" \
        --hash-bits ${BASE_HASH_BITS} \
        --cp-rank ${BASE_CP_RANK} \
        --backbone ${BASE_BACKBONE} \
        ${BASE_PRETRAINED_FLAG} \
        --batch-size ${BASE_BATCH_SIZE} \
        --lr ${lr_val} \
        --wd ${BASE_WD} \
        --margin ${BASE_MARGIN} \
        --lambda-sim ${BASE_LAMBDA_SIM} \
        --lambda-quant ${BASE_LAMBDA_QUANT} \
        --lambda-balance ${BASE_LAMBDA_BALANCE}
done

# --- 2. 测试哈希位数 (hash_bits) ---
echo "===== 测试哈希位数 (hash_bits) =====" | tee -a "$MAIN_LOG_FILE"
for hb_val in "${hash_bits_list[@]}"; do
    # 注意：如果CP Rank应该与hash_bits相关，你可能需要在这里调整 BASE_CP_RANK
    # 例如: local current_cp_rank=$((hb_val / 2)) # 或者保持固定
    run_experiment "hash_bits" "$hb_val" \
        --hash-bits ${hb_val} \
        --cp-rank ${BASE_CP_RANK} \
        --backbone ${BASE_BACKBONE} \
        ${BASE_PRETRAINED_FLAG} \
        --batch-size ${BASE_BATCH_SIZE} \
        --lr ${BASE_LR} \
        --wd ${BASE_WD} \
        --margin ${BASE_MARGIN} \
        --lambda-sim ${BASE_LAMBDA_SIM} \
        --lambda-quant ${BASE_LAMBDA_QUANT} \
        --lambda-balance ${BASE_LAMBDA_BALANCE}
done

# --- 3. 测试 CP 分解秩 (cp_rank) ---
echo "===== 测试 CP 分解秩 (cp_rank) =====" | tee -a "$MAIN_LOG_FILE"
for cpr_val in "${cp_rank_list[@]}"; do
    run_experiment "cp_rank" "$cpr_val" \
        --hash-bits ${BASE_HASH_BITS} \
        --cp-rank ${cpr_val} \
        --backbone ${BASE_BACKBONE} \
        ${BASE_PRETRAINED_FLAG} \
        --batch-size ${BASE_BATCH_SIZE} \
        --lr ${BASE_LR} \
        --wd ${BASE_WD} \
        --margin ${BASE_MARGIN} \
        --lambda-sim ${BASE_LAMBDA_SIM} \
        --lambda-quant ${BASE_LAMBDA_QUANT} \
        --lambda-balance ${BASE_LAMBDA_BALANCE}
done

# --- 4. 测试量化损失权重 (lambda_quant) ---
echo "===== 测试量化损失权重 (lambda_quant) =====" | tee -a "$MAIN_LOG_FILE"
for lq_val in "${lambda_quant_list[@]}"; do
    run_experiment "lambda_quant" "$lq_val" \
        --hash-bits ${BASE_HASH_BITS} \
        --cp-rank ${BASE_CP_RANK} \
        --backbone ${BASE_BACKBONE} \
        ${BASE_PRETRAINED_FLAG} \
        --batch-size ${BASE_BATCH_SIZE} \
        --lr ${BASE_LR} \
        --wd ${BASE_WD} \
        --margin ${BASE_MARGIN} \
        --lambda-sim ${BASE_LAMBDA_SIM} \
        --lambda-quant ${lq_val} \
        --lambda-balance ${BASE_LAMBDA_BALANCE}
done

# --- 5. 测试平衡损失权重 (lambda_balance) ---
echo "===== 测试平衡损失权重 (lambda_balance) =====" | tee -a "$MAIN_LOG_FILE"
for lb_val in "${lambda_balance_list[@]}"; do
    run_experiment "lambda_balance" "$lb_val" \
        --hash-bits ${BASE_HASH_BITS} \
        --cp-rank ${BASE_CP_RANK} \
        --backbone ${BASE_BACKBONE} \
        ${BASE_PRETRAINED_FLAG} \
        --batch-size ${BASE_BATCH_SIZE} \
        --lr ${BASE_LR} \
        --wd ${BASE_WD} \
        --margin ${BASE_MARGIN} \
        --lambda-sim ${BASE_LAMBDA_SIM} \
        --lambda-quant ${BASE_LAMBDA_QUANT} \
        --lambda-balance ${lb_val}
done

# --- 6. 测试相似性损失边距 (margin) ---
echo "===== 测试相似性损失边距 (margin) =====" | tee -a "$MAIN_LOG_FILE"
for m_val in "${margin_list[@]}"; do
    run_experiment "margin" "$m_val" \
        --hash-bits ${BASE_HASH_BITS} \
        --cp-rank ${BASE_CP_RANK} \
        --backbone ${BASE_BACKBONE} \
        ${BASE_PRETRAINED_FLAG} \
        --batch-size ${BASE_BATCH_SIZE} \
        --lr ${BASE_LR} \
        --wd ${BASE_WD} \
        --margin ${m_val} \
        --lambda-sim ${BASE_LAMBDA_SIM} \
        --lambda-quant ${BASE_LAMBDA_QUANT} \
        --lambda-balance ${BASE_LAMBDA_BALANCE}
done

# --- 7. 测试权重衰减 (wd) ---
echo "===== 测试权重衰减 (wd) =====" | tee -a "$MAIN_LOG_FILE"
for wd_val in "${wd_list[@]}"; do
    run_experiment "wd" "$wd_val" \
        --hash-bits ${BASE_HASH_BITS} \
        --cp-rank ${BASE_CP_RANK} \
        --backbone ${BASE_BACKBONE} \
        ${BASE_PRETRAINED_FLAG} \
        --batch-size ${BASE_BATCH_SIZE} \
        --lr ${BASE_LR} \
        --wd ${wd_val} \
        --margin ${BASE_MARGIN} \
        --lambda-sim ${BASE_LAMBDA_SIM} \
        --lambda-quant ${BASE_LAMBDA_QUANT} \
        --lambda-balance ${BASE_LAMBDA_BALANCE}
done

# --- 8. 测试批处理大小 (batch_size) ---
echo "===== 测试批处理大小 (batch_size) =====" | tee -a "$MAIN_LOG_FILE"
for bs_val in "${batch_size_list[@]}"; do
    # 警告：更改batch_size可能需要调整学习率，这里未自动调整
    run_experiment "batch_size" "$bs_val" \
        --hash-bits ${BASE_HASH_BITS} \
        --cp-rank ${BASE_CP_RANK} \
        --backbone ${BASE_BACKBONE} \
        ${BASE_PRETRAINED_FLAG} \
        --batch-size ${bs_val} \
        --lr ${BASE_LR} \
        --wd ${BASE_WD} \
        --margin ${BASE_MARGIN} \
        --lambda-sim ${BASE_LAMBDA_SIM} \
        --lambda-quant ${BASE_LAMBDA_QUANT} \
        --lambda-balance ${BASE_LAMBDA_BALANCE}
done

# --- (可选) 测试其他参数，如 backbone, pretrained ---
# echo "===== 测试 Backbone =====" | tee -a "$MAIN_LOG_FILE"
# backbones=("resnet18" "resnet34") # 添加你想测试的
# for bb_val in "${backbones[@]}"; do
#     run_experiment "backbone" "$bb_val" \
#         --hash-bits ${BASE_HASH_BITS} \
#         --cp-rank ${BASE_CP_RANK} \
#         --backbone ${bb_val} \
#         ${BASE_PRETRAINED_FLAG} \
#         --batch-size ${BASE_BATCH_SIZE} \
#         --lr ${BASE_LR} \
#         --wd ${BASE_WD} \
#         --margin ${BASE_MARGIN} \
#         --lambda-sim ${BASE_LAMBDA_SIM} \
#         --lambda-quant ${BASE_LAMBDA_QUANT} \
#         --lambda-balance ${BASE_LAMBDA_BALANCE}
# done

# echo "===== 测试 Pretrained =====" | tee -a "$MAIN_LOG_FILE"
# pretrained_flags=("--pretrained" "--no-pretrained")
# pretrained_labels=("True" "False") # 用于目录和日志记录
# for i in ${!pretrained_flags[@]}; do
#     pt_flag=${pretrained_flags[$i]}
#     pt_label=${pretrained_labels[$i]}
#     run_experiment "pretrained" "$pt_label" \
#         --hash-bits ${BASE_HASH_BITS} \
#         --cp-rank ${BASE_CP_RANK} \
#         --backbone ${BASE_BACKBONE} \
#         ${pt_flag} \
#         --batch-size ${BASE_BATCH_SIZE} \
#         --lr ${BASE_LR} \
#         --wd ${BASE_WD} \
#         --margin ${BASE_MARGIN} \
#         --lambda-sim ${BASE_LAMBDA_SIM} \
#         --lambda-quant ${BASE_LAMBDA_QUANT} \
#         --lambda-balance ${BASE_LAMBDA_BALANCE}
# done


echo "所有参数敏感性分析运行已完成。" | tee -a "$MAIN_LOG_FILE"
echo "检查主日志文件: $MAIN_LOG_FILE" | tee -a "$MAIN_LOG_FILE"
echo "检查各个子目录中的详细日志和结果: ${BASE_OUTPUT_DIR}/" | tee -a "$MAIN_LOG_FILE"