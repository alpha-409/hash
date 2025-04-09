# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class ImageHashingDataset(Dataset):
    """
    PyTorch Dataset for hashing models using preloaded data.
    Provides images and their original indices for similarity matrix construction.
    Uses the 'database' images as the primary training set.
    """
    def __init__(self, data_dict, dataset_type='db'):
        """
        Args:
            data_dict (dict): The dictionary returned by load_data.
            dataset_type (str): 'db' to use database images, 'query' for query images,
                                 'all' to use both (indices need careful mapping if using 'all').
                                 Default 'db' is typical for training hashing models.
        """
        super().__init__()

        if dataset_type == 'db':
            self.images = data_dict['db_images']
            self.paths = data_dict['db_paths']
            self.original_indices = list(range(len(data_dict['imlist'])))
            self.name_to_idx = {name: i for i, name in enumerate(data_dict['imlist'])}
        elif dataset_type == 'query':
            self.images = data_dict['query_images']
            self.paths = data_dict['query_paths']
            self.original_indices = list(range(len(data_dict['qimlist'])))
            self.name_to_idx = {name: i for i, name in enumerate(data_dict['qimlist'])}
        elif dataset_type == 'all':
            # Combine query and db images carefully, remapping indices
            raise NotImplementedError("Dataset type 'all' requires careful index remapping.")
        else:
             raise ValueError("Invalid dataset_type. Choose 'db', 'query', or implement 'all'.")


        # --- Precompute similarity information for faster batch S matrix creation ---
        self.gnd = data_dict['gnd'] # Ground truth structure
        self.qimlist = data_dict['qimlist']
        self.imlist = data_dict['imlist']
        self.positives_list = data_dict.get('positives', []) # List of (q_idx, db_idx)

        # Build a lookup for positive pairs using *database* indices
        # This is crucial if training primarily on the DB set.
        self.positive_pairs_db_indices = set()
        self.query_to_relevant_db_indices = [set() for _ in range(len(self.qimlist))]

        # Map query names to their indices in qimlist
        qname_to_qidx = {name: i for i, name in enumerate(self.qimlist)}

        if dataset_type == 'db': # If training on DB set
             # Map db image names to their indices in imlist
             dbname_to_dbidx = {name: i for i, name in enumerate(self.imlist)}
             dbidx_to_name = {i: name for name, i in dbname_to_dbidx.items()}

             # Create a set of all db indices that are considered 'relevant' copies of *any* query
             all_relevant_db_indices = set()
             for q_idx, q_gnd in enumerate(self.gnd): # Copydays structure
                 q_relevant_indices = set()
                 for key in ['ok', 'junk', 'strong', 'crops', 'jpegqual']: # Include ok/junk if needed? Usually not for positives.
                      if key in q_gnd:
                          q_relevant_indices.update(q_gnd[key])
                 all_relevant_db_indices.update(q_relevant_indices)
                 self.query_to_relevant_db_indices[q_idx] = q_relevant_indices


             # Build positive pairs WITHIN the database set based on common query origin
             # Find all db images that belong to the same original query
             original_map = {} # Map original query index -> set of db copy indices
             for q_idx, q_gnd in enumerate(self.gnd):
                relevant_indices = set()
                for key in ['ok', 'junk', 'strong', 'crops', 'jpegqual']:
                    if key in q_gnd:
                        relevant_indices.update(q_gnd[key])

                original_map[q_idx] = relevant_indices

             # Add pairs (db_idx1, db_idx2) if they belong to the same original query
             for q_idx, db_indices in original_map.items():
                 db_list = sorted(list(db_indices)) # Ensure consistent order
                 for i in range(len(db_list)):
                     for j in range(i + 1, len(db_list)):
                         # Add pair only if both indices are actually in our dataset subset (db_images)
                         idx1 = db_list[i]
                         idx2 = db_list[j]
                         # Check if these indices correspond to the images we loaded
                         if 0 <= idx1 < len(self.original_indices) and 0 <= idx2 < len(self.original_indices):
                              # Use the *current* indices within self.images if they differ from original
                              # Assuming self.original_indices maps 1:1 for 'db' type
                              self.positive_pairs_db_indices.add(tuple(sorted((idx1, idx2))))


        print(f"Dataset initialized with {len(self.images)} images of type '{dataset_type}'.")
        print(f"Found {len(self.positive_pairs_db_indices)} positive pairs within the DB set for training.")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # Return the original index associated with this image in the *full* list (imlist or qimlist)
        original_idx = self.original_indices[idx]
        return image, original_idx # Return image and its *original* index

# --- Collate Function for Batching ---
def collate_fn(batch, positive_pairs_lookup):
    """
    Custom collate function to create batches and the similarity matrix S.
    Args:
        batch (list): A list of tuples, where each tuple is (image_tensor, original_index).
        positive_pairs_lookup (set): A set containing tuples of sorted original indices for positive pairs.
    Returns:
        tuple: (images_batch, S_matrix, batch_original_indices)
               images_batch: Tensor of images (B, C, H, W).
               S_matrix: Similarity matrix (B, B) with 1 for similar, -1 for dissimilar, 0 for diagonal.
               batch_original_indices: List of original indices for the items in the batch.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    original_indices = [item[1] for item in batch]
    batch_size = len(original_indices)

    S = torch.full((batch_size, batch_size), -1.0, dtype=torch.float32) # Initialize all as dissimilar

    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                S[i, j] = 0.0 # Diagonal is ignored
            else:
                idx1 = original_indices[i]
                idx2 = original_indices[j]
                # Check if the pair (or its reverse) exists in the positive lookup
                if tuple(sorted((idx1, idx2))) in positive_pairs_lookup:
                    S[i, j] = 1.0 # Mark as similar

    return images, S, original_indices