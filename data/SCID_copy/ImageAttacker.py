import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
from tqdm import tqdm

class ImageAttacker:
    """Class to generate attacked versions of images"""
    
    def __init__(self, output_dir="attacked_images"):
        """
        Initialize ImageAttacker
        
        Args:
            output_dir (str): Base directory for attacked images
        """
        self.output_dir = Path(output_dir)
        
        # Create attack-specific directories
        self.attack_dirs = {
            "strong": self.output_dir / "strong_attack",
            "crops": self.output_dir / "crops_attack",
            "jpegqual": self.output_dir / "jpegqual_attack"
        }
        
        for dir_path in self.attack_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def apply_crop(self, image, ratio):
        """Apply center crop with given ratio"""
        width, height = image.size
        crop_width = int(width * ratio / 100)
        crop_height = int(height * ratio / 100)
        
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        return image.crop((left, top, right, bottom))
    
    def apply_blur(self, image, radius):
        """Apply Gaussian blur with given radius"""
        return image.filter(ImageFilter.GaussianBlur(radius))
    
    def apply_brightness(self, image, factor):
        """Apply brightness adjustment with given factor (percentage)"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor / 100)
    
    def apply_jpeg_compression(self, image, quality):
        """Apply JPEG compression with given quality"""
        # Save to temporary file and reload to apply actual JPEG compression
        temp_path = self.output_dir / "_temp.jpg"
        image.save(temp_path, "JPEG", quality=quality)
        compressed = Image.open(temp_path)
        temp_path.unlink()  # Delete temporary file
        return compressed
    
    def apply_all_attacks(self, image_path):
        """
        Apply all attacks to an image and save results
        
        Args:
            image_path (str): Path to original image
        """
        # Load image
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get base name without extension
                base_name = Path(image_path).stem
                
                # Strong attacks
                # 1. Blur attacks
                for radius in [3, 5, 7]:
                    attacked = self.apply_blur(img, radius)
                    out_path = self.attack_dirs["strong"] / f"{base_name}_blur_{radius}.jpg"
                    attacked.save(out_path, "JPEG", quality=95)
                
                # 2. Brightness attacks
                for factor in [50, 150]:  # 50% darker and 150% brighter
                    attacked = self.apply_brightness(img, factor)
                    out_path = self.attack_dirs["strong"] / f"{base_name}_brightness_{factor}.jpg"
                    attacked.save(out_path, "JPEG", quality=95)
                
                # Crop attacks
                crop_ratios = list(range(10, 81, 10)) + [15]  # 10,15,20,30,40,50,60,70,80
                for ratio in crop_ratios:
                    attacked = self.apply_crop(img, ratio)
                    out_path = self.attack_dirs["crops"] / f"{base_name}_crop_{ratio}.jpg"
                    attacked.save(out_path, "JPEG", quality=95)
                
                # JPEG quality attacks
                qualities = [75, 50, 30, 20, 15, 10, 8, 5, 3]
                for quality in qualities:
                    attacked = self.apply_jpeg_compression(img, quality)
                    out_path = self.attack_dirs["jpegqual"] / f"{base_name}_jpeg_{quality}.jpg"
                    attacked.save(out_path, "JPEG", quality=quality)
                
                return True
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False
    
    def process_directory(self, input_dir):
        """
        Process all images in a directory
        
        Args:
            input_dir (str): Directory containing original images
        """
        input_path = Path(input_dir)
        image_files = list(input_path.glob("*.jpg"))
        
        if not image_files:
            print(f"No JPG files found in {input_dir}")
            return
        
        print(f"\nProcessing {len(image_files)} images...")
        success_count = 0
        
        # Process images with progress bar
        for img_path in tqdm(image_files, desc="Generating attacks"):
            if self.apply_all_attacks(img_path):
                success_count += 1
        
        print(f"\nAttack generation complete!")
        print(f"Successfully processed: {success_count}/{len(image_files)} images")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate attacked versions of images")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing original images")
    parser.add_argument("--output", "-o", default="attacked_images", help="Output directory for attacked images")
    args = parser.parse_args()
    
    attacker = ImageAttacker(args.output)
    attacker.process_directory(args.input)

if __name__ == "__main__":
    main()
