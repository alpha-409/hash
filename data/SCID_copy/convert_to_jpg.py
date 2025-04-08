import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import multiprocessing

def convert_bmp_to_jpg(bmp_path, jpg_path, quality=95):
    """Convert a single BMP image to JPG format"""
    try:
        with Image.open(bmp_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Save as JPG
            img.save(jpg_path, 'JPEG', quality=quality)
        return True, bmp_path
    except Exception as e:
        return False, f"{bmp_path}: {str(e)}"

def process_directory(src_dir, dst_dir, num_workers=None, quality=95):
    """
    Convert all BMP images in source directory to JPG format
    
    Args:
        src_dir (str): Source directory containing BMP files
        dst_dir (str): Destination directory for JPG files
        num_workers (int): Number of worker threads (default: CPU count)
        quality (int): JPEG quality (0-100)
    """
    # Convert paths to Path objects
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Create destination directory if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of BMP files
    bmp_files = list(src_path.glob("*.bmp"))
    
    if not bmp_files:
        print(f"No BMP files found in {src_dir}")
        return
    
    # Set number of workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    print(f"\nConverting {len(bmp_files)} BMP files to JPG format...")
    print(f"Using {num_workers} worker threads")
    
    # Prepare conversion tasks
    conversion_tasks = []
    for bmp_file in bmp_files:
        jpg_file = dst_path / f"{bmp_file.stem}.jpg"
        conversion_tasks.append((bmp_file, jpg_file))
    
    # Process files in parallel
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(convert_bmp_to_jpg, bmp_file, jpg_file, quality)
            for bmp_file, jpg_file in conversion_tasks
        ]
        
        # Process results with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Converting images"
        ):
            success, result = future.result()
            if not success:
                failures.append(result)
    
    # Report results
    print(f"\nConversion complete!")
    print(f"Successfully converted: {len(bmp_files) - len(failures)} files")
    
    if failures:
        print(f"\nFailed conversions ({len(failures)}):")
        for failure in failures:
            print(f"- {failure}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert BMP images to JPG format")
    parser.add_argument("--src", "-s", required=True, help="Source directory containing BMP files")
    parser.add_argument("--dst", "-d", required=True, help="Destination directory for JPG files")
    parser.add_argument("--quality", "-q", type=int, default=95, help="JPEG quality (0-100)")
    parser.add_argument("--workers", "-w", type=int, help="Number of worker threads")
    args = parser.parse_args()
    
    # Validate JPEG quality
    if not 0 <= args.quality <= 100:
        parser.error("JPEG quality must be between 0 and 100")
    
    process_directory(args.src, args.dst, args.workers, args.quality)

if __name__ == "__main__":
    main()
