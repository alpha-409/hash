from .load_scid import load_scid
from .ImageAttacker import ImageAttacker
from .convert_to_jpg import process_directory as convert_bmp_to_jpg
from .convert_gnd import convert_to_gnd
from .process_scid import process_scid_dataset

__all__ = [
    'load_scid',           # Load processed SCID dataset
    'ImageAttacker',       # Generate attacked versions of images
    'convert_bmp_to_jpg',  # Convert BMP images to JPG format
    'convert_to_gnd',      # Generate ground truth JSON file
    'process_scid_dataset' # Process SCID dataset end-to-end
]

# Module version
__version__ = '1.0.0'

# Module description
__description__ = """
SCID Dataset Processing Module

This module provides tools for processing and using the SCID (Screen Content Image Database) dataset:

1. Dataset Preparation:
   - Convert BMP images to JPG format
   - Generate attacked versions (blur, brightness, crops, JPEG compression)
   - Create ground truth JSON file

2. Dataset Loading:
   - Load processed images and ground truth
   - Support for query and database images
   - Parallel loading with progress tracking

3. Attack Generation:
   - Gaussian blur (3 levels)
   - Brightness adjustment (2 levels)
   - Center crops (9 ratios)
   - JPEG compression (9 qualities)

Usage example:
    from SCID import process_scid_dataset, load_scid
    
    # Process dataset
    process_scid_dataset('path/to/scid')
    
    # Load processed dataset
    data = load_scid()
    
    # Access data
    query_images = data['query_images']
    db_images = data['db_images']
    positives = data['positives']
"""
