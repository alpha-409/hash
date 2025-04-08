import os
from ImageAttacker import ImageAttacker

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "attacked_images")
    
    # Initialize the attacker
    attacker = ImageAttacker(output_dir=output_dir)
    
    # Get input directory containing jpg images
    input_dir = os.path.join(os.path.dirname(__file__), "jpg")
    
    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            print(f"\nProcessing {filename}...")
            
            # Apply all attacks to the image
            attacker.apply_all_attacks(image_path)

if __name__ == "__main__":
    main()
