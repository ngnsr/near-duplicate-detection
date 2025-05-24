import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

def generate_synthetic_dataset(input_dir, output_dir, num_images=50, variations_per_image=4):
    """
    Generate a synthetic dataset with variations of input images.
    
    Args:
        input_dir (str): Path to folder with original images.
        output_dir (str): Path to save synthetic dataset.
        num_images (int): Number of original images to process (default: 50).
        variations_per_image (int): Number of variations per image (default: 4).
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List of image files (supporting .jpg and .png)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
    images = images[:min(num_images, len(images))]  # Limit to num_images
    
    # Ground truth list for CSV
    ground_truth = []
    
    for idx, img_name in enumerate(images):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        
        base_name = os.path.splitext(img_name)[0]
        
        # Save original image
        original_path = os.path.join(output_dir, f"{base_name}_orig.jpg")
        cv2.imwrite(original_path, img)
        original_size = os.path.getsize(original_path) / 1024  # Size in KB
        
        # Variation 1: JPEG compression (quality 50)
        compressed_path = os.path.join(output_dir, f"{base_name}_compressed.jpg")
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_pil.save(compressed_path, "JPEG", quality=50, optimize=True)
        compressed_size = os.path.getsize(compressed_path) / 1024
        ground_truth.append([original_path, compressed_path, 1])  # 1 = near-duplicate
        
        # Variation 2: Add Gaussian noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        noisy_img = cv2.add(img, noise)
        noisy_path = os.path.join(output_dir, f"{base_name}_noisy.jpg")
        cv2.imwrite(noisy_path, noisy_img)
        noisy_size = os.path.getsize(noisy_path) / 1024
        ground_truth.append([original_path, noisy_path, 1])
        
        # Variation 3: Resize (80% of original size)
        resized_img = cv2.resize(img, (int(img.shape[1] * 0.8), int(img.shape[0] * 0.8)))
        resized_path = os.path.join(output_dir, f"{base_name}_resized.jpg")
        cv2.imwrite(resized_path, resized_img)
        resized_size = os.path.getsize(resized_path) / 1024
        ground_truth.append([original_path, resized_path, 1])
        
        # Variation 4: Brightness adjustment (increase by 50)
        bright_img = cv2.convertScaleAbs(img, beta=50)
        bright_path = os.path.join(output_dir, f"{base_name}_bright.jpg")
        cv2.imwrite(bright_path, bright_img)
        bright_size = os.path.getsize(bright_path) / 1024
        ground_truth.append([original_path, bright_path, 1])
        
        # Log file sizes
        print(f"Generated variations for {img_name}:")
        print(f"  Original: {original_size:.2f} KB")
        print(f"  Compressed: {compressed_size:.2f} KB")
        print(f"  Noisy: {noisy_size:.2f} KB")
        print(f"  Resized: {resized_size:.2f} KB")
        print(f"  Bright: {bright_size:.2f} KB")
    
    # Save ground truth to CSV
    ground_truth_df = pd.DataFrame(ground_truth, columns=['image_1', 'image_2', 'is_duplicate'])
    ground_truth_path = os.path.join(output_dir, 'ground_truth.csv')
    ground_truth_df.to_csv(ground_truth_path, index=False)
    print(f"Ground truth saved to {ground_truth_path}")
    
    return ground_truth_path

if __name__ == "__main__":
    input_dir = "./original_images"  # Folder with original images
    output_dir = "./synthetic_dataset"  # Folder to save synthetic dataset
    generate_synthetic_dataset(input_dir, output_dir, num_images=50, variations_per_image=4)