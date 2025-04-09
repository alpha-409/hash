import os
# import cv2 # Still unused, consider removing
import json
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import argparse
from pathlib import Path
import random
import pickle # <-- Import the pickle module

class ImageAttacker:
    """Class to apply various attacks on images and generate JSON/PKL metadata."""

    def __init__(self, output_dir=None):
        """Initialize the ImageAttacker.

        Args:
            output_dir: Directory to save attacked images. If None, save in the same directory as source.
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Track intermediate results: mapping original filename to attacked filenames
        self.processed_images_filenames = {}
        # List of original image filenames (maintains order)
        self.query_images = []
        # Complete list of all image filenames (original + attacked)
        self.all_image_list = []

    # --- Attack methods (apply_jpeg_compression, apply_cropping, apply_strong_attacks) ---
    # No changes needed in the attack methods themselves
    # ... (Keep the existing attack methods as they are from the previous version) ...
    def apply_jpeg_compression(self, image_path, quality_factors=None):
        """Apply JPEG compression with different quality factors.

        Args:
            image_path: Path to the original image
            quality_factors: List of JPEG quality factors (0-100)

        Returns:
            Tuple: (List of paths to attacked images, List of base filenames of attacked images)
        """
        if quality_factors is None:
            # Reduced list for faster testing, use your original list if needed
            # quality_factors = [75, 50, 30, 20, 15, 10, 8, 5, 3]
            quality_factors = [75, 50, 20, 10, 5]


        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]

        output_paths = []
        jpeg_list_filenames = [] # Store filenames

        # Open the image
        try:
            img = Image.open(image_path).convert("RGB") # Ensure RGB for JPEG saving
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return [], []


        # Apply JPEG compression with different quality factors
        for qf in quality_factors:
            if self.output_dir:
                output_path = os.path.join(self.output_dir, f"jpegqual_{qf}_{base_name}.jpg")
            else:
                output_path = os.path.join(os.path.dirname(image_path), f"jpegqual_{qf}_{base_name}.jpg")

            try:
                img.save(output_path, "JPEG", quality=qf)
                output_paths.append(output_path)

                # Add to image list (filename only)
                output_filename = os.path.basename(output_path)
                if output_filename not in self.all_image_list: # Avoid duplicates if reprocessing occurs
                    self.all_image_list.append(output_filename)
                jpeg_list_filenames.append(output_filename)

                # print(f"Created JPEG compressed image with quality {qf}: {output_path}")
            except Exception as e:
                print(f"Error saving JPEG (quality {qf}) for {base_name}: {e}")


        # print(f"JPEG Compression applied to {img_name}")
        return output_paths, jpeg_list_filenames

    def apply_cropping(self, image_path, crop_percentages=None):
        """Apply cropping with different percentages of the image surface removed.

        Args:
            image_path: Path to the original image
            crop_percentages: List of percentages of the image to crop

        Returns:
            Tuple: (List of paths to attacked images, List of base filenames of attacked images)
        """
        if crop_percentages is None:
            # Reduced list for faster testing
            # crop_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            crop_percentages = [10, 30, 50, 70, 90]

        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]

        output_paths = []
        crop_list_filenames = [] # Store filenames

        # Open the image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return [], []

        width, height = img.size

        # Apply cropping with different percentages
        for percentage in crop_percentages:
            if percentage <= 0 or percentage >= 100:
                continue # Skip invalid percentages

            # Calculate the size of the *remaining* region
            crop_ratio = 1.0 - (percentage / 100.0)
            # Ensure cropped dimensions are at least 1 pixel
            crop_width = max(1, int(width * crop_ratio))
            crop_height = max(1, int(height * crop_ratio))

            # Calculate the random position for cropping (top-left corner)
            # Ensure the random range is valid (non-negative)
            max_left = max(0, width - crop_width)
            max_top = max(0, height - crop_height)
            left = np.random.randint(0, max_left + 1)
            top = np.random.randint(0, max_top + 1)
            right = left + crop_width
            bottom = top + crop_height

            # Crop the image
            try:
                cropped_img = img.crop((left, top, right, bottom))

                if self.output_dir:
                    output_path = os.path.join(self.output_dir, f"crops_{percentage}_{base_name}.jpg")
                else:
                    output_path = os.path.join(os.path.dirname(image_path), f"crops_{percentage}_{base_name}.jpg")

                cropped_img.save(output_path, "JPEG", quality=95) # Save as JPEG
                output_paths.append(output_path)

                # Add to image list (filename only)
                output_filename = os.path.basename(output_path)
                if output_filename not in self.all_image_list: # Avoid duplicates
                    self.all_image_list.append(output_filename)
                crop_list_filenames.append(output_filename)

                # print(f"Created cropped image with {percentage}% removed: {output_path}")
            except Exception as e:
                print(f"Error cropping/saving (percentage {percentage}) for {base_name}: {e}")

        # print(f"Cropping applied to {img_name}")
        return output_paths, crop_list_filenames

    def apply_strong_attacks(self, image_path, num_attacks=5):
        """Apply various strong attacks to an image.

        Args:
            image_path: Path to the original image
            num_attacks: Number of strong attacks to apply

        Returns:
            Tuple: (List of paths to attacked images, List of base filenames of attacked images)
        """
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]

        output_paths = []
        strong_list_filenames = [] # Store filenames

        # Open the image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return [], []

        # List of strong attack functions
        attack_functions = [
            self._apply_blur,
            self._apply_noise,
            self._apply_brightness,
            self._apply_contrast,
            self._apply_paint
        ]

        # Ensure num_attacks is not more than available functions
        num_attacks = min(num_attacks, len(attack_functions))

        # Apply a random subset of attacks
        if num_attacks > 0:
            selected_attacks = random.sample(attack_functions, num_attacks)
        else:
            selected_attacks = [] # Handle case where num_attacks could be 0


        attack_counter = 1
        for attack_func in selected_attacks:
            if self.output_dir:
                output_path = os.path.join(self.output_dir, f"strong_{attack_counter}_{base_name}.jpg")
            else:
                output_path = os.path.join(os.path.dirname(image_path), f"strong_{attack_counter}_{base_name}.jpg")

            try:
                attacked_img = attack_func(img.copy()) # Pass a copy to avoid modifying original for next attack
                attacked_img.save(output_path, "JPEG", quality=95) # Save as JPEG
                output_paths.append(output_path)

                # Add to image list (filename only)
                output_filename = os.path.basename(output_path)
                if output_filename not in self.all_image_list: # Avoid duplicates
                    self.all_image_list.append(output_filename)
                strong_list_filenames.append(output_filename)

                # print(f"Created strong attack image {attack_counter}: {output_path}")
                attack_counter += 1
            except Exception as e:
                print(f"Error applying/saving strong attack {attack_counter} ({attack_func.__name__}) for {base_name}: {e}")

        # print(f"Strong attacks applied to {img_name}")
        return output_paths, strong_list_filenames

    # --- Helper attack methods (_apply_blur, _apply_noise, etc.) ---
    # No changes needed here
    # ... (Keep the existing helper methods) ...
    def _apply_blur(self, img):
        """Apply Gaussian blur with random radius."""
        radius = random.choice([2, 3, 5, 7])
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def _apply_noise(self, img):
        """Apply random noise to the image."""
        try:
            img_array = np.array(img)
            if len(img_array.shape) != 3 or img_array.shape[2] < 3:
                 # print(f"Warning: Image for noise is not RGB/RGBA ({img_array.shape}). Skipping noise.")
                 return img # Return original if not suitable

            h, w, c = img_array.shape[:3] # Handle RGBA safely

            # Number of pixels to modify (5-15% of the image)
            noise_level = random.randint(5, 15)
            num_pixels = int((h * w * noise_level) / 100)

            # Generate random coordinates and noise values efficiently
            xs = np.random.randint(0, w, num_pixels)
            ys = np.random.randint(0, h, num_pixels)
            # Generate noise for appropriate number of channels (usually 3 for RGB)
            noise = np.random.randint(-50, 51, size=(num_pixels, img_array.shape[2]))

            # Apply noise using vectorized operations where possible
            # Ensure indices access valid channels
            target_channels = img_array.shape[2]
            pixels = img_array[ys, xs, :target_channels].astype(np.int16) # Read original pixel values
            noisy_pixels = pixels + noise
            img_array[ys, xs, :target_channels] = np.clip(noisy_pixels, 0, 255).astype(np.uint8) # Clip and assign back

            return Image.fromarray(img_array)
        except Exception as e:
             print(f"Error applying noise: {e}")
             return img # Return original on error


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
        try:
            stroke_size = random.choice([3, 5, 8, 10])
            painted_img = img.copy()
            draw = ImageDraw.Draw(painted_img)

            width, height = img.size
            num_strokes = random.randint(30, 70)

            for _ in range(num_strokes):
                x1 = np.random.randint(0, width)
                y1 = np.random.randint(0, height)
                # Ensure endpoints stay within bounds
                x2 = np.clip(x1 + np.random.randint(-50, 51), 0, width-1)
                y2 = np.clip(y1 + np.random.randint(-50, 51), 0, height-1)

                color = (
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                )
                # Use tuple for color if image is RGB, add alpha if RGBA
                # fill_color = color if img.mode == "RGB" else (*color, 255)
                fill_color = color # PIL handles RGB/RGBA okay with 3-tuple usually

                # Only draw if line has some length
                if x1 != x2 or y1 != y2:
                    draw.line((x1, y1, x2, y2), fill=fill_color, width=stroke_size)

            return painted_img
        except Exception as e:
            print(f"Error applying paint effect: {e}")
            return img # Return original on error

    # --- Processing and Metadata Generation ---

    def process_image(self, image_path):
        """Process a single image with all attacks.

        Args:
            image_path: Path to the original image

        Returns:
            Boolean: True if processing was successful (at least partially), False otherwise.
        """
        print(f"Processing: {image_path}")
        if not os.path.exists(image_path):
            print(f"Error: Input image not found at {image_path}")
            return False

        base_name = os.path.basename(image_path)

        # Add original image to query list and the main list *first*
        # Ensure it's not added multiple times if process_image is called again
        if base_name not in self.query_images:
             self.query_images.append(base_name)
        if base_name not in self.all_image_list:
            # Add original image at the beginning of its block
            self.all_image_list.append(base_name)
        else:
            # If original image is already in all_image_list, it means we might be reprocessing.
            print(f"Warning: Original image {base_name} seems to be processed more than once.")
            # We'll continue, but be aware index mapping might need care if duplicates are intended.


        # Apply attacks and get lists of attacked filenames
        _, jpeg_list = self.apply_jpeg_compression(image_path)
        _, crop_list = self.apply_cropping(image_path)
        _, strong_list = self.apply_strong_attacks(image_path)

        # Store intermediate results using filenames
        # Overwrite if reprocessing occurs, using the latest attack results
        self.processed_images_filenames[base_name] = {
            "jpegqual": jpeg_list,
            "crops": crop_list,
            "strong": strong_list
        }

        # print(f"Finished processing: {base_name}")
        return True


    def generate_metadata_files(self, json_output_path):
        """Generate JSON and PKL files with metadata, using indices for gnd.

        Args:
            json_output_path: Path to save the JSON file. PKL file will be derived.
        """
        print("\nGenerating metadata files (JSON and PKL)...")

        # Deduplicate all_image_list while preserving order (important for index stability)
        # Using dict.fromkeys for Python 3.7+ order preservation
        unique_ordered_list = list(dict.fromkeys(self.all_image_list))
        if len(unique_ordered_list) != len(self.all_image_list):
            print(f"Warning: Duplicates found in image list. Using unique list of {len(unique_ordered_list)} items for indexing.")
            self.all_image_list = unique_ordered_list

        # 1. Create the filename-to-index mapping from the final (unique) list
        filename_to_index = {name: idx for idx, name in enumerate(self.all_image_list)}

        # 2. Build the index-based gnd structure
        gnd_indexed = []
        missing_attacked_files = 0
        # Iterate through query_images to maintain the correct order corresponding to qimlist
        for query_filename in self.query_images:
            if query_filename not in self.processed_images_filenames:
                print(f"Warning: No processed data found for query image {query_filename}. Appending empty entry to gnd.")
                gnd_indexed.append({}) # Append an empty dict for this query
                continue
            # Check if the query image itself is in the index map (it should be!)
            if query_filename not in filename_to_index:
                 print(f"Critical Warning: Query image '{query_filename}' not found in the final all_image_list index map. Skipping in gnd.")
                 gnd_indexed.append({}) # Append empty as a fallback
                 continue


            attack_results_filenames = self.processed_images_filenames[query_filename]
            attack_indices_dict = {}

            for attack_type, attacked_filenames in attack_results_filenames.items():
                indices_list = []
                for fname in attacked_filenames:
                    if fname in filename_to_index:
                        indices_list.append(filename_to_index[fname])
                    else:
                        # This indicates an attacked file was generated but somehow didn't make it
                        # into the final unique all_image_list, or wasn't added initially.
                        print(f"Warning: Attacked filename '{fname}' (from query '{query_filename}') not found in the final index map. Skipping this file.")
                        missing_attacked_files += 1
                attack_indices_dict[attack_type] = indices_list

            gnd_indexed.append(attack_indices_dict)

        if missing_attacked_files > 0:
             print(f"Total missing/skipped attacked files during index mapping: {missing_attacked_files}")


        # 3. Create the final metadata dictionary
        metadata = {
            # gnd is now a list, order matches qimlist
            "gnd": gnd_indexed,
            # imlist contains all unique filenames (original + attacked)
            "imlist": self.all_image_list,
            # qimlist contains original filenames
            "qimlist": self.query_images
        }

        # --- Save as JSON ---
        try:
            with open(json_output_path, 'w') as f:
                json.dump(metadata, f, indent=2) # Use indent for readability
            print(f"Generated JSON metadata file: {json_output_path}")
        except Exception as e:
            print(f"Error writing JSON file to {json_output_path}: {e}")

        # --- Save as PKL ---
        # Derive PKL path from JSON path
        pkl_output_path = Path(json_output_path).with_suffix('.pkl')
        try:
            with open(pkl_output_path, 'wb') as f: # Use 'wb' for binary write mode
                pickle.dump(metadata, f) # Dump the whole dictionary
            print(f"Generated PKL metadata file: {pkl_output_path}")
        except Exception as e:
            print(f"Error writing PKL file to {pkl_output_path}: {e}")


def main():
    """Main function to parse arguments and run the image attacker."""
    parser = argparse.ArgumentParser(description="Apply attacks to images and generate JSON/PKL metadata (index-based gnd)")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory containing images")
    parser.add_argument("--output", "-o", default=None, help="Output directory for attacked images (default: relative dir 'output_attacks')")
    # Keep --json argument name, but it now dictates both json and pkl filenames
    parser.add_argument("--json", "-j", default="dataset_metadata_indexed.json", help="Output JSON file path (PKL file will have the same base name)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = args.output

    # Determine input type and set default output dir if needed
    if input_path.is_file():
        if output_dir is None:
            output_dir = input_path.resolve().parent / "output_attacks" # Use resolve for better path handling
        image_paths = [input_path]
        print(f"Processing single file: {input_path}")
    elif input_path.is_dir():
        if output_dir is None:
            # Place default output dir relative to the script's CWD or the input dir?
            # Let's keep it relative to input dir for consistency.
            output_dir = input_path.resolve() / "output_attacks"
        image_paths = []
        print(f"Scanning directory: {input_path}")
        # More comprehensive list of common image extensions
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.gif', '*.webp']:
            image_paths.extend(list(input_path.glob(ext)))
        if not image_paths:
             print(f"No supported images found in directory: {input_path}")
             return
        print(f"Found {len(image_paths)} images.")
        # Sort image paths for potentially more consistent ordering if needed
        image_paths.sort()
    else:
        print(f"Error: Input '{args.input}' is not a valid file or directory")
        return

    # Create the image attacker with the determined output directory
    attacker = ImageAttacker(output_dir=str(output_dir)) # Ensure output_dir is string

    # Process images
    processed_count = 0
    for file_path in image_paths:
        if attacker.process_image(str(file_path)):
             processed_count += 1

    if processed_count > 0:
        # Generate JSON & PKL metadata files
        # Pass the designated JSON path; PKL path will be derived inside the method.
        attacker.generate_metadata_files(args.json)
    else:
        print("No images were successfully processed. Metadata files not generated.")


if __name__ == "__main__":
    main()

# Example Usage:
# python your_script_name.py -i images/ -o attacked_images/ -j metadata/dataset_info.json
# (This will create attacked_images/, metadata/dataset_info.json, and metadata/dataset_info.pkl)

# python your_script_name.py -i my_image.png
# (This will create ./output_attacks/, ./dataset_metadata_indexed.json, and ./dataset_metadata_indexed.pkl)