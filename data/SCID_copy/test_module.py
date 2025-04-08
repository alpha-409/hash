import os
import shutil
import unittest
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from .load_scid import load_scid
from .ImageAttacker import ImageAttacker
from .convert_to_jpg import process_directory as convert_bmp_to_jpg
from .convert_gnd import convert_to_gnd
from .process_scid import process_scid_dataset

class TestSCIDModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create test environment with sample images"""
        cls.test_dir = Path("test_data")
        cls.test_dir.mkdir(exist_ok=True)
        
        # Create test directories
        cls.bmp_dir = cls.test_dir / "bmp"
        cls.jpg_dir = cls.test_dir / "jpg"
        cls.attack_dir = cls.test_dir / "attacked_images"
        
        cls.bmp_dir.mkdir(exist_ok=True)
        cls.jpg_dir.mkdir(exist_ok=True)
        
        # Create sample BMP images
        cls.create_sample_images()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def create_sample_images(cls):
        """Create sample test images"""
        # Create two sample images with different patterns
        for i in range(1, 3):
            # Create a sample image with some pattern
            img = Image.new('RGB', (256, 256))
            pixels = img.load()
            
            # Create different patterns for each image
            for x in range(256):
                for y in range(256):
                    if i == 1:
                        # First image: gradient pattern
                        pixels[x, y] = ((x + y) % 256, x % 256, y % 256)
                    else:
                        # Second image: checkerboard pattern
                        pixels[x, y] = (
                            ((x + y) // 32 % 2) * 255,
                            (x // 32 % 2) * 255,
                            (y // 32 % 2) * 255
                        )
            
            # Save as BMP
            bmp_path = cls.bmp_dir / f"SCI{i:02d}.bmp"
            img.save(bmp_path, 'BMP')
    
    def test_bmp_to_jpg_conversion(self):
        """Test BMP to JPG conversion"""
        convert_bmp_to_jpg(
            src_dir=str(self.bmp_dir),
            dst_dir=str(self.jpg_dir),
            quality=95
        )
        
        # Check if JPG files were created
        jpg_files = list(self.jpg_dir.glob("*.jpg"))
        self.assertEqual(len(jpg_files), 2)
    
    def test_attack_generation(self):
        """Test attack generation"""
        attacker = ImageAttacker(str(self.attack_dir))
        attacker.process_directory(str(self.jpg_dir))
        
        # Check if attack directories were created
        attack_types = ['strong_attack', 'crops_attack', 'jpegqual_attack']
        for attack_type in attack_types:
            self.assertTrue((self.attack_dir / attack_type).exists())
        
        # Check if attacked images were created
        strong_attacks = len(list((self.attack_dir / 'strong_attack').glob("*.jpg")))
        crop_attacks = len(list((self.attack_dir / 'crops_attack').glob("*.jpg")))
        jpeg_attacks = len(list((self.attack_dir / 'jpegqual_attack').glob("*.jpg")))
        
        # Each image should have:
        # - 5 strong attacks (3 blur + 2 brightness)
        # - 9 crop attacks
        # - 9 jpeg quality attacks
        self.assertEqual(strong_attacks, 10)  # 5 attacks × 2 images
        self.assertEqual(crop_attacks, 18)    # 9 crops × 2 images
        self.assertEqual(jpeg_attacks, 18)    # 9 qualities × 2 images
    
    def test_ground_truth_generation(self):
        """Test ground truth file generation"""
        convert_to_gnd(str(self.test_dir))
        
        # Check if ground truth file was created
        gnd_path = self.test_dir / "gnd_scid.json"
        self.assertTrue(gnd_path.exists())
        
        # Basic validation of dataset loading
        data = load_scid(str(self.test_dir))
        self.assertIn('query_images', data)
        self.assertIn('db_images', data)
        self.assertIn('positives', data)
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing"""
        # Clean up previous test artifacts
        if self.jpg_dir.exists():
            shutil.rmtree(self.jpg_dir)
        if self.attack_dir.exists():
            shutil.rmtree(self.attack_dir)
        if (self.test_dir / "gnd_scid.json").exists():
            (self.test_dir / "gnd_scid.json").unlink()
        
        # Run end-to-end processing
        process_scid_dataset(str(self.test_dir))
        
        # Verify results
        self.assertTrue(self.jpg_dir.exists())
        self.assertTrue(self.attack_dir.exists())
        self.assertTrue((self.test_dir / "gnd_scid.json").exists())
        
        # Load and verify dataset
        data = load_scid(str(self.test_dir))
        self.assertIsInstance(data['query_images'], torch.Tensor)
        self.assertIsInstance(data['db_images'], torch.Tensor)
        self.assertEqual(len(data['query_images']), 2)  # 2 test images

if __name__ == '__main__':
    unittest.main(verbosity=2)
