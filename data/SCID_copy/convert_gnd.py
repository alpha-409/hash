import os
import json
from pathlib import Path

def verify_image_exists(img_path, simulate=True):
    """Verify if an image file exists"""
    if not simulate:
        return Path(img_path).exists()
    return True

def convert_to_gnd(output_dir=None, simulate=True):
    """
    Convert SCID images and attacks to ground truth JSON file
    
    Args:
        output_dir (str): Directory to save the JSON file (default: current dir)
        simulate (bool): If True, don't verify image existence
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize data structure
    data = {
        "gnd": [],
        "imlist": [],
        "qimlist": []
    }
    
    # Define attack parameters
    attack_params = {
        "strong": [
            ("blur", [3, 5, 7]),
            ("brightness", [50, 150])
        ],
        "crops": list(range(10, 81, 10)) + [15],  # 10,15,20,30,40,50,60,70,80
        "jpegqual": [75, 50, 30, 20, 15, 10, 8, 5, 3]
    }
    
    missing_files = []
    
    # Generate for each SCID image (SCI01 to SCI40)
    for i in range(1, 41):
        img_name = f"SCI{i:02d}"
        query_path = f"jpg/{img_name}.jpg"
        
        # Verify query image
        if not verify_image_exists(query_path, simulate):
            missing_files.append(query_path)
        
        # Add to query list
        data["qimlist"].append(query_path)
        
        # Initialize attacked versions entry
        gnd_entry = {
            "strong": [],
            "crops": [],
            "jpegqual": []
        }
        
        # Generate strong attacks
        for attack_type, params in attack_params["strong"]:
            for param in params:
                img_file = f"{img_name}_{attack_type}_{param}.jpg"
                attack_path = f"attacked_images/strong_attack/{img_file}"
                
                if not verify_image_exists(attack_path, simulate):
                    missing_files.append(attack_path)
                
                gnd_entry["strong"].append(img_file)
                data["imlist"].append(attack_path)
        
        # Generate crop attacks
        for crop_ratio in sorted(attack_params["crops"]):
            img_file = f"{img_name}_crop_{crop_ratio}.jpg"
            attack_path = f"attacked_images/crops_attack/{img_file}"
            
            if not verify_image_exists(attack_path, simulate):
                missing_files.append(attack_path)
            
            gnd_entry["crops"].append(img_file)
            data["imlist"].append(attack_path)
        
        # Generate JPEG quality attacks
        for quality in sorted(attack_params["jpegqual"], reverse=True):
            img_file = f"{img_name}_jpeg_{quality}.jpg"
            attack_path = f"attacked_images/jpegqual_attack/{img_file}"
            
            if not verify_image_exists(attack_path, simulate):
                missing_files.append(attack_path)
            
            gnd_entry["jpegqual"].append(img_file)
            data["imlist"].append(attack_path)
        
        data["gnd"].append(gnd_entry)
    
    # Save to JSON file
    output_file = os.path.join(output_dir, "gnd_scid.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nGround truth file generated: {output_file}")
    print(f"Total query images: {len(data['qimlist'])}")
    print(f"Total attacked images: {len(data['imlist'])}")
    print(f"Total ground truth entries: {len(data['gnd'])}")
    
    if not simulate:
        print("\nMissing files:")
        for file in missing_files:
            print(f"- {file}")
    
    # Verify structure consistency
    exp_attacks_per_image = (
        len(attack_params["crops"]) + 
        len(attack_params["jpegqual"]) + 
        sum(len(params) for _, params in attack_params["strong"])
    )
    exp_total_attacks = exp_attacks_per_image * len(data["qimlist"])
    
    if len(data["imlist"]) != exp_total_attacks:
        print(f"\nWarning: Expected {exp_total_attacks} attacked images, but got {len(data['imlist'])}")

    return data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert SCID dataset to ground truth JSON")
    parser.add_argument("--output", "-o", help="Output directory for JSON file")
    parser.add_argument("--verify", "-v", action="store_true", help="Verify image existence")
    args = parser.parse_args()
    
    convert_to_gnd(args.output, not args.verify)
