import os
import cv2
import json
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import argparse
from pathlib import Path
import random


class ImageAttacker:
    """Class to apply various attacks on images and generate JSON metadata."""
    
    def __init__(self, output_dir=None):
        """Initialize the ImageAttacker.
        
        Args:
            output_dir: Directory to save attacked images. If None, save in the same directory as source.
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Track all processed images for JSON generation
        self.processed_images = {}
        self.query_images = []
        self.all_image_list = []
        
    def apply_jpeg_compression(self, image_path, quality_factors=None):
        """Apply JPEG compression with different quality factors.
        
        Args:
            image_path: Path to the original image
            quality_factors: List of JPEG quality factors (0-100)
        
        Returns:
            List of paths to attacked images
        """
        if quality_factors is None:
            quality_factors = [75, 50, 30, 20, 15, 10, 8, 5, 3]
        
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]
        
        output_paths = []
        jpeg_list = []
        
        # Open the image
        img = Image.open(image_path)
        
        # Apply JPEG compression with different quality factors
        for qf in quality_factors:
            if self.output_dir:
                output_path = os.path.join(self.output_dir, f"jpegqual_{qf}_{base_name}.jpg")
            else:
                output_path = os.path.join(os.path.dirname(image_path), f"jpegqual_{qf}_{base_name}.jpg")
                
            img.save(output_path, "JPEG", quality=qf)
            output_paths.append(output_path)
            
            # Add to image list
            output_filename = os.path.basename(output_path)
            self.all_image_list.append(output_filename)
            jpeg_list.append(output_filename)
            
            print(f"Created JPEG compressed image with quality {qf}: {output_path}")
        
        return output_paths, jpeg_list
    
    def apply_cropping(self, image_path, crop_percentages=None):
        """Apply cropping with different percentages of the image surface removed.
        
        Args:
            image_path: Path to the original image
            crop_percentages: List of percentages of the image to crop
        
        Returns:
            List of paths to attacked images
        """
        if crop_percentages is None:
            crop_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]
        
        output_paths = []
        crop_list = []
        
        # Open the image
        img = Image.open(image_path)
        width, height = img.size
        
        # Apply cropping with different percentages
        for percentage in crop_percentages:
            # Calculate the size of the cropped region
            crop_ratio = 1.0 - (percentage / 100.0)
            crop_width = int(width * crop_ratio)
            crop_height = int(height * crop_ratio)
            
            # Calculate the random position for cropping
            left = np.random.randint(0, width - crop_width + 1)
            top = np.random.randint(0, height - crop_height + 1)
            right = left + crop_width
            bottom = top + crop_height
            
            # Crop the image
            cropped_img = img.crop((left, top, right, bottom))
            
            if self.output_dir:
                output_path = os.path.join(self.output_dir, f"crops_{percentage}_{base_name}.jpg")
            else:
                output_path = os.path.join(os.path.dirname(image_path), f"crops_{percentage}_{base_name}.jpg")
                
            cropped_img.save(output_path, "JPEG", quality=95)
            output_paths.append(output_path)
            
            # Add to image list
            output_filename = os.path.basename(output_path)
            self.all_image_list.append(output_filename)
            crop_list.append(output_filename)
            
            print(f"Created cropped image with {percentage}% removed: {output_path}")
        
        return output_paths, crop_list
    
    def apply_strong_attacks(self, image_path, num_attacks=5):
        """Apply various strong attacks to an image.
        
        Args:
            image_path: Path to the original image
            num_attacks: Number of strong attacks to apply
        
        Returns:
            List of paths to attacked images
        """
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]
        
        output_paths = []
        strong_list = []
        
        # Open the image
        img = Image.open(image_path)
        
        # List of strong attack functions
        attack_functions = [
            self._apply_blur,
            self._apply_noise,
            self._apply_brightness,
            self._apply_contrast,
            self._apply_paint
        ]
        
        # Apply a random subset of attacks
        selected_attacks = random.sample(attack_functions, min(num_attacks, len(attack_functions)))
        
        attack_counter = 1
        for attack_func in selected_attacks:
            if self.output_dir:
                output_path = os.path.join(self.output_dir, f"strong_{attack_counter}_{base_name}.jpg")
            else:
                output_path = os.path.join(os.path.dirname(image_path), f"strong_{attack_counter}_{base_name}.jpg")
                
            attacked_img = attack_func(img)
            attacked_img.save(output_path, "JPEG", quality=95)
            output_paths.append(output_path)
            
            # Add to image list
            output_filename = os.path.basename(output_path)
            self.all_image_list.append(output_filename)
            strong_list.append(output_filename)
            
            print(f"Created strong attack image {attack_counter}: {output_path}")
            attack_counter += 1
        
        return output_paths, strong_list
    
    def _apply_blur(self, img):
        """Apply Gaussian blur with random radius."""
        radius = random.choice([2, 3, 5, 7])
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _apply_noise(self, img):
        """Apply random noise to the image."""
        img_array = np.array(img)
        h, w, c = img_array.shape
        
        # Number of pixels to modify (5-15% of the image)
        noise_level = random.randint(5, 15)
        num_pixels = int((h * w * noise_level) / 100)
        
        # Randomly modify pixels
        for _ in range(num_pixels):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            
            # Add random value to the pixel
            for i in range(c):
                value = img_array[y, x, i] + np.random.randint(-50, 50)
                img_array[y, x, i] = max(0, min(255, value))
        
        return Image.fromarray(img_array)
    
    def _apply_brightness(self, img):
        """Apply brightness adjustment with random factor."""
        factor = random.choice([0.5, 0.7, 1.3, 1.5])
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def _apply_contrast(self, img):
        """Apply contrast adjustment with random factor."""
        factor = random.choice([0.5, 0.7, 1.3, 1.5])
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def _apply_paint(self, img):
        """Simulate paint effect by adding random colored strokes."""
        stroke_size = random.choice([3, 5, 8, 10])
        painted_img = img.copy()
        draw = ImageDraw.Draw(painted_img)
        
        # Add random colored strokes
        width, height = img.size
        num_strokes = random.randint(30, 70)
        
        for _ in range(num_strokes):
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            x2 = min(width, x1 + np.random.randint(-50, 50))
            y2 = min(height, y1 + np.random.randint(-50, 50))
            
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
            
            draw.line((x1, y1, x2, y2), fill=color, width=stroke_size)
        
        return painted_img
    
    def process_image(self, image_path):
        """Process a single image with all attacks.
        
        Args:
            image_path: Path to the original image
        
        Returns:
            Dictionary with the processing results
        """
        base_name = os.path.basename(image_path)
        
        # Add original image to query list
        self.query_images.append(base_name)
        self.all_image_list.append(base_name)
        
        # Apply attacks
        _, jpeg_list = self.apply_jpeg_compression(image_path)
        _, crop_list = self.apply_cropping(image_path)
        _, strong_list = self.apply_strong_attacks(image_path)
        
        # Store results for JSON generation
        self.processed_images[base_name] = {
            "jpegqual": jpeg_list,
            "crops": crop_list,
            "strong": strong_list
        }
        
        return self.processed_images[base_name]
    
    def generate_json(self, output_path):
        """Generate JSON file with metadata.
        
        Args:
            output_path: Path to save the JSON file
        """
        # Create the JSON structure
        data = {
            "gnd": self.processed_images,
            "imlist": self.all_image_list,
            "qimlist": self.query_images
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Generated JSON metadata file: {output_path}")


def main():
    """Main function to parse arguments and run the image attacker."""
    parser = argparse.ArgumentParser(description="Apply various attacks to images and generate JSON metadata")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory (if not specified, save in same directory as source)")
    parser.add_argument("--json", "-j", default="dataset_metadata.json", help="Output JSON file path")
    args = parser.parse_args()
    
    # Create the image attacker
    attacker = ImageAttacker(output_dir=args.output)
    
    # Process input (file or directory)
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        attacker.process_image(str(input_path))
    elif input_path.is_dir():
        # Directory
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            image_files.extend(list(input_path.glob(ext)))
        
        for file_path in image_files:
            attacker.process_image(str(file_path))
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    # Generate JSON metadata
    attacker.generate_json(args.json)


if __name__ == "__main__":
    main()

