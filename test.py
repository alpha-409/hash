import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import time
import torch
from torchvision import transforms
from scipy.fftpack import dct
from data import load_copydays_dataset, visualize_results

# Determine if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()

class ImageHashDetector:
    def __init__(self, device='auto'):
        """
        Initialize the hash detector with specified device
        
        Parameters:
            device (str): 'cpu', 'cuda', or 'auto' (will use CUDA if available)
        """
        self.device = self._get_device(device)
        print(f"Using device: {self.device}")
        
    def _get_device(self, device):
        """Determine the actual device to use"""
        if device == 'auto':
            return 'cuda' if CUDA_AVAILABLE else 'cpu'
        elif device == 'cuda' and not CUDA_AVAILABLE:
            print("CUDA requested but not available, falling back to CPU")
            return 'cpu'
        return device
    
    def _preprocess_for_hash(self, img_path, size=8, grayscale=True):
        """Load and preprocess image for hashing"""
        img = Image.open(img_path)
        if grayscale:
            img = img.convert('L')
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        return img_array
    
    def _hamming_distance(self, hash1, hash2):
        """Calculate Hamming distance between two hashes"""
        return np.count_nonzero(hash1 != hash2)
    
    def _batch_hamming_distance(self, query_hashes, db_hashes):
        """
        Calculate Hamming distances between query hashes and database hashes
        
        Returns:
            distances: matrix of shape (num_queries, num_db_images)
        """
        if self.device == 'cuda':
            # Convert to PyTorch tensors for CUDA acceleration
            q_tensor = torch.tensor(query_hashes, device='cuda')
            db_tensor = torch.tensor(db_hashes, device='cuda')
            
            # Calculate XOR between each query and all db_hashes
            # Reshape for broadcasting (num_queries, 1, hash_size) x (1, num_db, hash_size)
            xor_result = q_tensor.view(q_tensor.size(0), 1, -1) != db_tensor.view(1, db_tensor.size(0), -1)
            
            # Sum bits that differ (Hamming distance)
            distances = torch.sum(xor_result, dim=2).cpu().numpy()
        else:
            # CPU implementation
            num_queries = len(query_hashes)
            num_db = len(db_hashes)
            distances = np.zeros((num_queries, num_db), dtype=np.int32)
            
            for i in range(num_queries):
                for j in range(num_db):
                    distances[i, j] = self._hamming_distance(query_hashes[i], db_hashes[j])
                    
        return distances
    
    def compute_ahash(self, img_path):
        """
        Compute Average Hash (aHash)
        
        Steps:
        1. Resize image to 8x8
        2. Convert to grayscale
        3. Compute average pixel value
        4. Set bits based on comparison with average
        """
        img_array = self._preprocess_for_hash(img_path)
        avg = np.mean(img_array)
        hash_bits = (img_array >= avg).flatten()
        return hash_bits
    
    def compute_dhash(self, img_path):
        """
        Compute Difference Hash (dHash)
        
        Steps:
        1. Resize image to 9x8
        2. Convert to grayscale
        3. Compute differences between adjacent pixels
        4. Set bits based on comparison
        """
        img = Image.open(img_path).convert('L')
        img = img.resize((9, 8), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        
        # Compute differences (horizontal)
        diff = img_array[:, 1:] > img_array[:, :-1]
        return diff.flatten()
    
    def compute_phash(self, img_path):
        """
        Compute Perceptual Hash (pHash)
        
        Steps:
        1. Resize image to 32x32
        2. Convert to grayscale
        3. Apply DCT transformation
        4. Keep low-frequency components (upper left 8x8)
        5. Compute median of low-frequency values
        6. Set bits based on comparison with median
        """
        img = Image.open(img_path).convert('L')
        img = img.resize((32, 32), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        
        # Apply DCT
        dct_result = dct(dct(img_array.T, norm='ortho').T, norm='ortho')
        
        # Keep low-frequency components (upper left 8x8)
        dct_low = dct_result[:8, :8]
        
        # Compute median (excluding the DC component at 0,0)
        median = np.median(dct_low.flatten()[1:])
        
        # Set bits based on comparison with median
        hash_bits = (dct_low >= median).flatten()
        return hash_bits
    
    def compute_hashes_for_dataset(self, images_dict, hash_method):
        """
        Compute hashes for all images in the dataset
        
        Parameters:
            images_dict: Dictionary of images
            hash_method: 'ahash', 'dhash', or 'phash'
        
        Returns:
            Dictionary mapping image IDs to their hash values
        """
        hash_functions = {
            'ahash': self.compute_ahash,
            'dhash': self.compute_dhash,
            'phash': self.compute_phash
        }
        
        hash_func = hash_functions.get(hash_method)
        if not hash_func:
            raise ValueError(f"Invalid hash method: {hash_method}")
        
        image_hashes = {}
        total = len(images_dict)
        
        start_time = time.time()
        for i, (image_id, image_info) in enumerate(images_dict.items()):
            if (i + 1) % 100 == 0:
                print(f"Processing {i+1}/{total} images...")
            
            image_path = image_info['path']
            image_hashes[image_id] = hash_func(image_path)
        
        elapsed = time.time() - start_time
        print(f"Computed {hash_method} for {total} images in {elapsed:.2f} seconds")
        
        return image_hashes
    
    def search_image_by_hash(self, query_hashes, db_hashes, top_k=10):
        """
        Search for similar images based on hash values
        
        Parameters:
            query_hashes: Dictionary of query image hashes
            db_hashes: Dictionary of database image hashes
            top_k: Number of top results to return
        
        Returns:
            Dictionary mapping query IDs to list of (db_id, distance) tuples
        """
        # Convert dictionaries to lists for batch processing
        query_ids = list(query_hashes.keys())
        db_ids = list(db_hashes.keys())
        
        query_hash_array = np.array([query_hashes[q_id] for q_id in query_ids])
        db_hash_array = np.array([db_hashes[db_id] for db_id in db_ids])
        
        # Calculate distances
        distances = self._batch_hamming_distance(query_hash_array, db_hash_array)
        
        # For each query, find top-k results
        results = {}
        for i, query_id in enumerate(query_ids):
            query_distances = distances[i]
            
            # Get indices of top-k smallest distances
            if top_k < len(db_ids):
                top_indices = np.argpartition(query_distances, top_k)[:top_k]
                # Sort these top-k indices by distance
                top_indices = top_indices[np.argsort(query_distances[top_indices])]
            else:
                top_indices = np.argsort(query_distances)
            
            # Map indices to db_ids and distances
            results[query_id] = [(db_ids[idx], float(query_distances[idx])) for idx in top_indices]
        
        return results

def calculate_map(search_results, ground_truth):
    """
    Calculate Mean Average Precision (mAP)
    
    Parameters:
        search_results: Dictionary mapping query IDs to list of (db_id, distance) tuples
        ground_truth: Dictionary mapping query IDs to their true matches
    
    Returns:
        mAP value
    """
    aps = []
    
    for query_id, results in search_results.items():
        if query_id not in ground_truth:
            continue
            
        true_match = ground_truth[query_id]
        precision_list = []
        
        found = False
        for rank, (result_id, _) in enumerate(results, 1):
            if result_id == true_match:
                precision = 1.0 / rank
                precision_list.append(precision)
                found = True
                break
                
        ap = np.mean(precision_list) if precision_list else 0
        if found:  # Only include queries where the true match was found
            aps.append(ap)
    
    return np.mean(aps) if aps else 0

def calculate_micro_ap(search_results, ground_truth, threshold=10):
    """
    Calculate Micro Average Precision (μAP) at a specific threshold
    
    Parameters:
        search_results: Dictionary mapping query IDs to list of (db_id, distance) tuples
        ground_truth: Dictionary mapping query IDs to their true matches
        threshold: Hamming distance threshold for considering a match
    
    Returns:
        μAP value
    """
    tp = 0  # True positives
    fp = 0  # False positives
    
    for query_id, results in search_results.items():
        if query_id not in ground_truth:
            continue
            
        true_match = ground_truth[query_id]
        
        # Check if the top result is correct and within threshold
        if results and results[0][0] == true_match and results[0][1] <= threshold:
            tp += 1
        else:
            fp += 1
    
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def evaluate_hash_method(detector, original_images, query_images, ground_truth, hash_method, threshold=10):
    """
    Evaluate a hash method on the dataset
    
    Parameters:
        detector: ImageHashDetector instance
        original_images: Dictionary of original images
        query_images: Dictionary of query images
        ground_truth: Dictionary mapping query IDs to their true matches
        hash_method: Hash method to evaluate ('ahash', 'dhash', or 'phash')
        threshold: Hamming distance threshold for μAP
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating {hash_method}...")
    
    # Compute hashes
    start_time = time.time()
    db_hashes = detector.compute_hashes_for_dataset(original_images, hash_method)
    query_hashes = detector.compute_hashes_for_dataset(query_images, hash_method)
    hash_time = time.time() - start_time
    
    # Search for matches
    start_time = time.time()
    search_results = detector.search_image_by_hash(query_hashes, db_hashes)
    search_time = time.time() - start_time
    
    # Calculate metrics
    mAP = calculate_map(search_results, ground_truth)
    microAP = calculate_micro_ap(search_results, ground_truth, threshold)
    
    # Group results by transformation type
    transform_groups = {}
    for query_id, results in search_results.items():
        if query_id in query_images:
            transform = query_images[query_id].get('transform', 'unknown')
            if transform not in transform_groups:
                transform_groups[transform] = {'results': {}, 'ground_truth': {}}
            transform_groups[transform]['results'][query_id] = results
            if query_id in ground_truth:
                transform_groups[transform]['ground_truth'][query_id] = ground_truth[query_id]
    
    # Calculate metrics per transformation
    transform_metrics = {}
    for transform, data in transform_groups.items():
        transform_mAP = calculate_map(data['results'], data['ground_truth'])
        transform_microAP = calculate_micro_ap(data['results'], data['ground_truth'], threshold)
        transform_metrics[transform] = {
            'mAP': transform_mAP,
            'μAP': transform_microAP,
            'count': len(data['results'])
        }
    
    return {
        'hash_method': hash_method,
        'mAP': mAP,
        'μAP': microAP,
        'hash_time': hash_time,
        'search_time': search_time,
        'transform_metrics': transform_metrics,
        'search_results': search_results  # Return results for visualization
    }

def visualize_metrics(eval_results):
    """
    Visualize evaluation metrics for different hash methods
    
    Parameters:
        eval_results: List of evaluation result dictionaries
    """
    methods = [res['hash_method'] for res in eval_results]
    map_values = [res['mAP'] for res in eval_results]
    microap_values = [res['μAP'] for res in eval_results]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, map_values, width, label='mAP')
    ax.bar(x + width/2, microap_values, width, label='μAP')
    
    ax.set_ylabel('Score')
    ax.set_title('Performance of Different Hash Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    
    # Add value labels
    for i, v in enumerate(map_values):
        ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(microap_values):
        ax.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()

def visualize_transform_metrics(eval_results):
    """
    Visualize metrics per transformation type
    
    Parameters:
        eval_results: List of evaluation result dictionaries
    """
    for result in eval_results:
        method = result['hash_method']
        transform_metrics = result['transform_metrics']
        
        transforms = list(transform_metrics.keys())
        map_values = [metrics['mAP'] for metrics in transform_metrics.values()]
        microap_values = [metrics['μAP'] for metrics in transform_metrics.values()]
        counts = [metrics['count'] for metrics in transform_metrics.values()]
        
        x = np.arange(len(transforms))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, map_values, width, label='mAP')
        ax.bar(x + width/2, microap_values, width, label='μAP')
        
        ax.set_ylabel('Score')
        ax.set_title(f'Performance of {method} by Transformation Type')
        ax.set_xticks(x)
        ax.set_xticklabels(transforms)
        ax.legend()
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(i, -0.05, f'n={count}', ha='center')
        
        plt.tight_layout()
        plt.show()

def run_image_hash_evaluation(base_dir, device='auto'):
    """
    Run complete evaluation of hash-based copy detection methods
    
    Parameters:
        base_dir: Base directory of Copydays dataset
        device: 'cpu', 'cuda', or 'auto'
    
    Returns:
        Evaluation results
    """
    print("Loading Copydays dataset...")

    original_images, query_images, ground_truth = load_copydays_dataset(base_dir, preprocess=False)
    
    print(f"Loaded {len(original_images)} original images and {len(query_images)} query images")
    print(f"Ground truth contains {len(ground_truth)} mappings")
    
    # Initialize detector
    detector = ImageHashDetector(device=device)
    
    # Evaluate different hash methods
    eval_results = []
    for hash_method in ['ahash', 'dhash', 'phash']:
        result = evaluate_hash_method(
            detector, original_images, query_images, ground_truth, hash_method
        )
        eval_results.append(result)
        
        print(f"\nResults for {hash_method}:")
        print(f"mAP: {result['mAP']:.4f}")
        print(f"μAP: {result['μAP']:.4f}")
        print(f"Hash computation time: {result['hash_time']:.2f} seconds")
        print(f"Search time: {result['search_time']:.2f} seconds")
    
    # Visualize overall results
    visualize_metrics(eval_results)
    
    # Visualize results per transformation
    visualize_transform_metrics(eval_results)
    
    return eval_results, detector, original_images, query_images, ground_truth

def visualize_sample_results(eval_results, original_images, query_images, ground_truth, num_samples=5):
    """
    Visualize sample search results
    
    Parameters:
        eval_results: List of evaluation result dictionaries
        original_images: Dictionary of original images
        query_images: Dictionary of query images
        ground_truth: Dictionary of ground truth mappings
        num_samples: Number of samples to visualize
    """
    # Get sample query IDs
    query_ids = list(query_images.keys())
    if len(query_ids) > num_samples:
        np.random.seed(42)  # For reproducibility
        sample_ids = np.random.choice(query_ids, num_samples, replace=False)
    else:
        sample_ids = query_ids
    
    for query_id in sample_ids:
        query_path = query_images[query_id]['path']
        transform = query_images[query_id].get('transform', 'unknown')
        
        # Get ground truth if available
        gt_path = None
        if query_id in ground_truth:
            gt_id = ground_truth[query_id]
            if gt_id in original_images:
                gt_path = original_images[gt_id]['path']
        
        # Show results for each hash method
        for result in eval_results:
            method = result['hash_method']
            search_results = result['search_results']
            
            if query_id in search_results:
                retrieved = [(original_images[img_id]['path'], 1.0 - dist/64.0) 
                            for img_id, dist in search_results[query_id][:5]
                            if img_id in original_images]
                
                print(f"\nQuery: {query_id} (Transform: {transform})")
                print(f"Hash method: {method}")
                
                visualize_results(query_path, retrieved, gt_path, top_k=5)

# Main execution
if __name__ == "__main__":
    # Use this function to run the complete evaluation
    # Example: results, detector, orig_imgs, query_imgs, gnd_truth = run_image_hash_evaluation("/path/to/copydays")
    
    # You can then visualize sample results:
    # visualize_sample_results(results, orig_imgs, query_imgs, gnd_truth)
    base_dir = os.getcwd()  # 当前目录
    results, detector, orig_imgs, query_imgs, gnd_truth = run_image_hash_evaluation(base_dir=base_dir)
    print("Image Hash Detection module loaded successfully!")
    print("Use run_image_hash_evaluation() function to run the complete evaluation.")

