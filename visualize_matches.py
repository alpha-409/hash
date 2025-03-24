import matplotlib.pyplot as plt
from PIL import Image
from copydays_loader import CopydaysDataset
import os

def show_query_and_matches(dataset, query_idx=0):
    """Display a query image and its matches."""
    # Get the query image
    query_name = dataset.get_query_images()[query_idx]
    query_path = dataset.get_image_path(query_name)
    
    # Get ground truth matches
    gnd = dataset.get_ground_truth(query_idx)
    
    # Get image list
    img_list = dataset.get_database_images()
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Show query image
    plt.subplot(141)
    if os.path.exists(query_path):
        query_img = Image.open(query_path)
        plt.imshow(query_img)
        plt.title('Query Image\n' + query_name)
    else:
        plt.text(0.5, 0.5, 'Image not found', ha='center', va='center')
    plt.axis('off')
    
    # Show one example from each category
    categories = [
        ('Strong Match', gnd['strong']),
        ('Cropped Version', gnd['crops']),
        ('JPEG Variant', gnd['jpegqual'])
    ]
    
    for idx, (title, matches) in enumerate(categories, start=2):
        plt.subplot(1, 4, idx)
        if matches and len(matches) > 0:
            match_name = img_list[matches[0]]
            match_path = dataset.get_image_path(match_name)
            if os.path.exists(match_path):
                match_img = Image.open(match_path)
                plt.imshow(match_img)
                plt.title(f'{title}\n{match_name}')
            else:
                plt.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        else:
            plt.text(0.5, 0.5, 'No matches', ha='center', va='center')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('copydays_example.png')
    plt.close()

if __name__ == "__main__":
    dataset = CopydaysDataset("copydays/gnd_copydays.pkl")
    show_query_and_matches(dataset)
    print("Visualization saved as 'copydays_example.png'")
    
    # Print additional dataset information
    print("\nDataset Information:")
    print(f"Total number of query images: {len(dataset.get_query_images())}")
    
    # Print example of first ground truth matches
    gnd = dataset.get_ground_truth(0)
    print(f"\nFirst query image matches:")
    print(f"- Number of strong matches: {len(gnd['strong'])}")
    print(f"- Number of cropped versions: {len(gnd['crops'])}")
    print(f"- Number of JPEG quality variants: {len(gnd['jpegqual'])}")
