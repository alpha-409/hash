# SCID Dataset Processing Module

This module provides tools for processing and using the SCID (Screen Content Image Database) dataset. It handles the complete pipeline from raw BMP images to a processed dataset with various image attacks.

## Features

1. **Dataset Preparation**
   - Convert BMP images to JPG format
   - Generate attacked versions of images
   - Create ground truth JSON file

2. **Attack Types**
   - **Strong Attacks**
     - Gaussian blur (radius: 3, 5, 7)
     - Brightness adjustment (50%, 150%)
   - **Crop Attacks**
     - Center crops (ratios: 10%, 15%, 20%, 30%, 40%, 50%, 60%, 70%, 80%)
   - **JPEG Quality**
     - Compression levels (quality: 75, 50, 30, 20, 15, 10, 8, 5, 3)

3. **Efficient Processing**
   - Parallel image processing
   - Progress tracking
   - Error handling and reporting

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd path/to/SCID
   ```

2. Install requirements:
   ```bash
   pip install pillow tqdm numpy
   ```

## Directory Structure

```
SCID/
├── bmp/                  # Original BMP images
├── jpg/                  # Converted JPG images
├── attacked_images/      # Generated attack versions
│   ├── strong_attack/
│   ├── crops_attack/
│   └── jpegqual_attack/
└── gnd_scid.json        # Ground truth file
```

## Usage

### 1. End-to-End Processing

Process the entire dataset from BMP to attacked versions:

```python
from SCID import process_scid_dataset

# Process dataset
process_scid_dataset('path/to/scid', verify=True)
```

### 2. Individual Components

#### a. Convert BMP to JPG
```python
from SCID import convert_bmp_to_jpg

convert_bmp_to_jpg(
    src_dir='path/to/bmp',
    dst_dir='path/to/jpg',
    quality=95
)
```

#### b. Generate Attacks
```python
from SCID import ImageAttacker

attacker = ImageAttacker('path/to/output')
attacker.process_directory('path/to/jpg')
```

#### c. Create Ground Truth
```python
from SCID import convert_to_gnd

convert_to_gnd(output_dir='path/to/scid')
```

#### d. Load Dataset
```python
from SCID import load_scid

data = load_scid()
query_images = data['query_images']  # Query images tensor
db_images = data['db_images']        # Database images tensor
positives = data['positives']        # Positive pairs
```

## Command Line Interface

Each component can be run from the command line:

1. Process entire dataset:
```bash
python process_scid.py -d /path/to/scid --verify
```

2. Convert BMP to JPG:
```bash
python convert_to_jpg.py -s /path/to/bmp -d /path/to/jpg
```

3. Generate attacks:
```bash
python ImageAttacker.py -i /path/to/jpg -o /path/to/attacked_images
```

4. Create ground truth:
```bash
python convert_gnd.py -o /path/to/output --verify
```

## Output Format

### Ground Truth JSON Structure
```json
{
  "gnd": [
    {
      "strong": ["SCI01_blur_3.jpg", ...],
      "crops": ["SCI01_crop_10.jpg", ...],
      "jpegqual": ["SCI01_jpeg_75.jpg", ...]
    },
    ...
  ],
  "imlist": ["attacked_images/...", ...],
  "qimlist": ["jpg/SCI01.jpg", ...]
}
```

### Loaded Dataset Format
```python
{
    'query_images': torch.Tensor,  # Shape: [N, C, H, W]
    'db_images': torch.Tensor,     # Shape: [M, C, H, W]
    'query_paths': list,           # Original paths
    'db_paths': list,             # Attack paths
    'gnd': list,                  # Ground truth data
    'imlist': list,               # Database image list
    'qimlist': list,              # Query image list
    'positives': list             # (query_idx, db_idx) pairs
}
```

## Notes

1. Make sure the BMP directory contains all original SCID images before processing.
2. The module creates all necessary directories if they don't exist.
3. Use the verify option to check file existence during ground truth generation.
4. Progress bars show processing status for long operations.
5. Error logs are provided for failed operations.

## Requirements

- Python 3.6+
- Pillow
- tqdm
- numpy
- torch (for loading dataset)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
