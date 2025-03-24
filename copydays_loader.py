import pickle
import numpy as np
from typing import Dict, List, Any
import os

class CopydaysDataset:
    def __init__(self, pkl_path: str):
        """Initialize the Copydays dataset loader.
        
        Args:
            pkl_path: Path to the gnd_copydays.pkl file
        """
        self.data = self._load_data(pkl_path)
        self.gnd = self.data['gnd']
        self.image_list = self.data['imlist']
        self.query_list = self.data['qimlist']
        
    def _load_data(self, pkl_path: str) -> Dict[str, Any]:
        """Load the dataset from pickle file."""
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Dataset file not found: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    def get_query_images(self) -> List[str]:
        """Get list of query image names."""
        return self.query_list
    
    def get_database_images(self) -> List[str]:
        """Get list of all database image names."""
        return self.image_list
    
    def get_ground_truth(self, query_idx: int) -> Dict[str, List[int]]:
        """Get ground truth matches for a query image.
        
        Args:
            query_idx: Index of the query image
            
        Returns:
            Dictionary containing:
                - 'strong': List of indices for strong matches
                - 'crops': List of indices for cropped versions
                - 'jpegqual': List of indices for JPEG quality variants
        """
        if query_idx >= len(self.gnd):
            raise IndexError(f"Query index {query_idx} out of range")
        return self.gnd[query_idx]
    
    def get_image_path(self, image_name: str) -> str:
        """Get the full path for an image.
        
        Args:
            image_name: Name of the image from imlist or qimlist
            
        Returns:
            Full path to the image file
        """
        return os.path.join('copydays', 'jpg', f"{image_name}.jpg")

# Example usage
if __name__ == "__main__":
    dataset = CopydaysDataset("copydays/gnd_copydays.pkl")
    
    # Print dataset statistics
    print("Copydays Dataset Summary:")
    print(f"Number of query images: {len(dataset.get_query_images())}")
    print(f"Number of database images: {len(dataset.get_database_images())}")
    
    # Example: Get ground truth for first query
    print("\nGround truth for first query image:")
    gnd = dataset.get_ground_truth(0)
    print("Strong matches:", gnd['strong'])
    print("Cropped versions:", gnd['crops'])
    print("JPEG quality variants:", gnd['jpegqual'])
    
    # Example: Get some image paths
    print("\nExample image paths:")
    if len(dataset.get_query_images()) > 0:
        query_img = dataset.get_query_images()[0]
        print(f"First query image: {dataset.get_image_path(query_img)}")
    
    if len(dataset.get_database_images()) > 0:
        db_img = dataset.get_database_images()[0]
        print(f"First database image: {dataset.get_image_path(db_img)}")
