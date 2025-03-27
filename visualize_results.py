import matplotlib.pyplot as plt
import numpy as np

# 算法名称
algorithms = [
    'aHash', 'pHash', 'dHash', 'wHash', 'cHash', 'mhHash', 'ResNet50 Hash', 
    'ResNet50 Deep Features', 'Multiscale ResNet50 Hash', 'Multiscale ResNet50 Deep Features', 'ViT Hash', 'ViT Deep Features'
]

# 每个算法对应不同哈希大小的 mAP 和 μAP 值
hash_sizes = [8, 16, 32, 64, 128, 256]
mAP_values = {
    'aHash': [0.6368, 0.6717, 0.6682, 0.6624, 0.6579, 0.6557],
    'pHash': [0.5589, 0.5727, 0.5347, 0.5081, 0.4852, 0.4681],
    'dHash': [0.5561, 0.5721, 0.5514, 0.5170, 0.4899, 0.4731],
    'wHash': [0.5796, 0.6203, 0.6298, 0.6267, 0.6199, 0.6135],
    'cHash': [0.4008, 0.4668, 0.3712, 0.3479, 0.4396, 0.4404],
    'mhHash': [0.4433, 0.4795, 0.4785, 0.4735, 0.4641, 0.4400],
    'ResNet50 Hash': [0.7121, 0.8662, 0.9015, 0.9045, 0.9045, 0.9045],
    'ResNet50 Deep Features': [0.8893, 0.8893, 0.8893, 0.8893, 0.8893, 0.8893],
    'Multiscale ResNet50 Hash': [0.7757, 0.8994, 0.9253, 0.9295, 0.9295, 0.9295],
    'Multiscale ResNet50 Deep Features': [0.9188, 0.9188, 0.9188, 0.9188, 0.9188, 0.9188],
    'ViT Hash': [0.7432, 0.8761, 0.9020, 0.9020, 0.9020, 0.9020],
    'ViT Deep Features': [0.9138, 0.9138, 0.9138, 0.9138, 0.9138, 0.9138],
}

# 设置图表
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制每个算法的 mAP 曲线
for algo in algorithms:
    ax.plot(hash_sizes, mAP_values[algo], label=algo)

# 设置标签和标题
ax.set_xlabel('哈希大小')
ax.set_ylabel('mAP')
ax.set_title('不同哈希算法的mAP性能')

# 设置网格
ax.grid(True)

# 显示图例
ax.legend(title="算法", bbox_to_anchor=(1.05, 1), loc='upper left')

# 保存图形
plt.tight_layout()
plt.savefig('g:/trae/myhash/results/all_algorithms_map.png', dpi=300)
plt.close()

# 绘制两个组别的对比度调整
# 为了区分这两组数据，我们使用不同的线型和更高的对比度
group_1 = ['aHash', 'pHash', 'dHash', 'wHash', 'cHash', 'mhHash']
group_2 = ['ResNet50 Hash', 'ResNet50 Deep Features', 'Multiscale ResNet50 Hash', 'Multiscale ResNet50 Deep Features', 'ViT Hash', 'ViT Deep Features']

# 设置图表
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制组1的mAP值（相同线型，增加对比度）
for algo in group_1:
    ax.plot(hash_sizes, mAP_values[algo], label=algo, linestyle='-', linewidth=2)

# 绘制组2的mAP值（相同线型，增加对比度）
for algo in group_2:
    ax.plot(hash_sizes, mAP_values[algo], label=algo, linestyle='--', linewidth=2)

# 设置标签和标题
ax.set_xlabel('哈希大小')
ax.set_ylabel('mAP')
ax.set_title('不同哈希算法的mAP性能（高对比度）')

# 设置网格
ax.grid(True)

# 显示图例
ax.legend(title="算法", bbox_to_anchor=(1.05, 1), loc='upper left')

# 保存图形
plt.tight_layout()
plt.savefig('g:/trae/myhash/results/grouped_algorithms_map.png', dpi=300)
plt.close()

# 单独绘制传统哈希算法
fig, ax = plt.subplots(figsize=(10, 6))
for algo in group_1:
    ax.plot(hash_sizes, mAP_values[algo], label=algo, linewidth=2)

ax.set_xlabel('哈希大小')
ax.set_ylabel('mAP')
ax.set_title('传统哈希算法的mAP性能')
ax.grid(True)
ax.legend(title="算法")
plt.tight_layout()
plt.savefig('g:/trae/myhash/results/traditional_algorithms_map.png', dpi=300)
plt.close()

# 单独绘制深度学习哈希算法
fig, ax = plt.subplots(figsize=(10, 6))
for algo in group_2:
    ax.plot(hash_sizes, mAP_values[algo], label=algo, linewidth=2)

ax.set_xlabel('哈希大小')
ax.set_ylabel('mAP')
ax.set_title('深度学习哈希算法的mAP性能')
ax.grid(True)
ax.legend(title="算法")
plt.tight_layout()
plt.savefig('g:/trae/myhash/results/deep_learning_algorithms_map.png', dpi=300)
plt.close()

# 绘制柱状图比较最佳性能
best_map = {algo: max(mAP_values[algo]) for algo in algorithms}
best_hash_size = {algo: hash_sizes[np.argmax(mAP_values[algo])] for algo in algorithms}

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(algorithms, [best_map[algo] for algo in algorithms])

# 添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}\n(size={best_hash_size[algorithms[i]]})',
            ha='center', va='bottom', rotation=0)

ax.set_xlabel('算法')
ax.set_ylabel('最佳mAP')
ax.set_title('各算法的最佳mAP性能')
ax.set_ylim(0, 1.0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('g:/trae/myhash/results/best_performance_map.png', dpi=300)
plt.close()

print("可视化分析完成，图表已保存到 g:/trae/myhash/results/ 目录")