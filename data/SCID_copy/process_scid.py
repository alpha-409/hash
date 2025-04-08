import os
from pathlib import Path
import argparse
from convert_to_jpg import process_directory as convert_to_jpg
from ImageAttacker import ImageAttacker
from convert_gnd import convert_to_gnd

def process_scid_dataset(base_dir, verify=True):
    """
    Process the SCID dataset end-to-end:
    1. Convert BMP to JPG
    2. Generate attacked versions
    3. Create ground truth file
    
    Args:
        base_dir (str): Base directory containing SCID dataset
        verify (bool): Whether to verify file existence in ground truth
    """
    base_path = Path(base_dir)
    
    # Define directories
    bmp_dir = base_path / "bmp"
    jpg_dir = base_path / "jpg"
    attack_dir = base_path / "attacked_images"
    
    if not bmp_dir.exists():
        raise FileNotFoundError(f"BMP directory not found: {bmp_dir}")
    
    print("\n=== SCID Dataset Processing ===")
    
    # Step 1: Convert BMP to JPG
    print("\n1. Converting BMP images to JPG format...")
    convert_to_jpg(
        src_dir=str(bmp_dir),
        dst_dir=str(jpg_dir),
        quality=95
    )
    
    # Step 2: Generate attacked versions
    print("\n2. Generating attacked versions...")
    attacker = ImageAttacker(str(attack_dir))
    attacker.process_directory(str(jpg_dir))
    
    # Step 3: Create ground truth file
    print("\n3. Generating ground truth file...")
    convert_to_gnd(
        output_dir=str(base_path),
        simulate=not verify
    )
    
    print("\n=== Processing Complete ===")
    print(f"Original JPG images: {jpg_dir}")
    print(f"Attacked images: {attack_dir}")
    print(f"Ground truth file: {base_path}/gnd_scid.json")

def main():
    parser = argparse.ArgumentParser(description="Process SCID dataset end-to-end")
    parser.add_argument(
        "--dir",
        "-d",
        required=True,
        help="Base directory containing SCID dataset"
    )
    parser.add_argument(
        "--verify",
        "-v",
        action="store_true",
        help="Verify file existence in ground truth"
    )
    args = parser.parse_args()
    
    try:
        process_scid_dataset(args.dir, args.verify)
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
