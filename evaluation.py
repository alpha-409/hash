# evaluation.py
import torch
import numpy as np
from tqdm import tqdm

def calculate_hamming_distance(B1, B2):
    """
    Calculates Hamming distance between two sets of binary hash codes {-1, 1}.
    """
    L = B1.shape[1]
    dist = 0.5 * (L - torch.matmul(B1, B2.t()))
    return dist

def calculate_metrics(query_hash, db_hash, query_labels, db_labels, gnd_truth, top_k=None):
    """
    Calculates Mean Average Precision (mAP) and Micro-Average Precision (μAP).

    Args:
        query_hash (torch.Tensor): Query hash codes (N_query, L), {-1, 1}.
        db_hash (torch.Tensor): Database hash codes (N_db, L), {-1, 1}.
        query_labels (list/np.array): Original labels/indices for query items (N_query).
        db_labels (list/np.array): Original labels/indices for database items (N_db).
        gnd_truth (list): List where gnd_truth[q_idx] = set of relevant db_indices for query q_idx.
        top_k (int, optional): Calculate metrics@k. If None, use all retrieved items.

    Returns:
        dict: Dictionary containing 'mAP' and 'muAP'.
    """
    num_query = query_hash.shape[0]
    num_db = db_hash.shape[0]
    APs = [] # List to store AP for each query (for mAP)

    # Accumulators for micro-average calculation
    total_sum_precision = 0.0 # Sum of precision@k for every hit across all queries
    total_relevant_items = 0  # Total number of relevant items across all queries

    # Ensure hashes are on CPU for numpy operations if needed, or keep on GPU
    device = query_hash.device
    query_hash = query_hash.to(device)
    db_hash = db_hash.to(device)

    # Calculate all pairwise Hamming distances
    distances = calculate_hamming_distance(query_hash, db_hash) # Shape (N_query, N_db)

    print("Calculating AP/μAP for each query...")
    for i in tqdm(range(num_query)):
        query_idx = query_labels[i] # Original index/label of the i-th query

        # Get true relevant database indices for this query
        try:
            relevant_db_indices = gnd_truth[i]
        except IndexError:
            print(f"Warning: Could not find ground truth for query index {i} (original label {query_idx}). Skipping.")
            continue

        if not isinstance(relevant_db_indices, set):
             relevant_db_indices = set(relevant_db_indices)

        num_relevant = len(relevant_db_indices)

        # If a query has no relevant items, its AP is 0, and it contributes nothing to μAP sum.
        if num_relevant == 0:
            APs.append(0.0)
            continue

        total_relevant_items += num_relevant # Accumulate for μAP denominator

        # Get distances for the current query and sort
        dists = distances[i, :]
        sorted_db_indices_local = torch.argsort(dists) # Indices relative to db_hash (0 to N_db-1)

        # Map sorted indices back to original db labels/indices
        retrieved_db_original_indices = [db_labels[k] for k in sorted_db_indices_local.cpu().numpy()]

        # Limit to top_k if specified
        if top_k is not None and top_k > 0:
            retrieved_db_original_indices = retrieved_db_original_indices[:top_k]

        # Calculate AP for this query and accumulate precision sum for μAP
        hits = 0
        query_sum_precision = 0.0 # Sum of P@k for this query's relevant items

        for k, retrieved_idx in enumerate(retrieved_db_original_indices):
            if retrieved_idx in relevant_db_indices:
                hits += 1
                precision_at_k = hits / (k + 1.0)
                query_sum_precision += precision_at_k # Accumulate P@k for *this query's* hits

        if hits == 0:
            average_precision = 0.0
        else:
            average_precision = query_sum_precision / num_relevant
            total_sum_precision += query_sum_precision # Add this query's total P@k sum to the global sum for μAP

        APs.append(average_precision) # Store AP for mAP calculation

    # Calculate final metrics
    mAP = np.mean(APs) if APs else 0.0
    muAP = total_sum_precision / total_relevant_items if total_relevant_items > 0 else 0.0

    return {'mAP': mAP, 'muAP': muAP}