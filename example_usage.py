from data import load_copydays_dataset, visualize_results, calculate_metrics
import os

def main():
    # 使用当前工作目录作为base_dir
    base_dir = os.getcwd()
    
    # 加载数据集
    print("Loading Copydays dataset...")
    original_images, query_images, ground_truth = load_copydays_dataset(
        base_dir,
        preprocess=True,
        target_size=(224, 224)  # 示例尺寸
    )
    
    print("\nDataset Statistics:")
    print(f"Number of original images: {len(original_images)}")
    print(f"Number of query images: {len(query_images)}")
    print(f"Number of ground truth mappings: {len(ground_truth)}")
    
    # 示例：显示第一个查询图像及其对应的ground truth图像
    if query_images and ground_truth:
        first_query_id = list(query_images.keys())[0]
        if first_query_id in ground_truth:
            first_gt_id = ground_truth[first_query_id]
            
            # 构造一个模拟的检索结果（这里只是演示）
            retrieved_paths = [
                (original_images[first_gt_id]["path"], 1.0),  # Ground truth match
                # 添加一些其他图像作为对比
                *[(original_images[img_id]["path"], 0.5) 
                  for img_id in list(original_images.keys())[1:5]]
            ]
            
            print("\nVisualizing first query results...")
            visualize_results(
                query_images[first_query_id]["path"],
                retrieved_paths,
                ground_truth_path=original_images[first_gt_id]["path"],
                top_k=5
            )

if __name__ == "__main__":
    main()
