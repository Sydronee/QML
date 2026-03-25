import os
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm
import string

# Reuse your existing mapping
ALPHABET = string.digits + string.ascii_letters
CHAR_MAPPING = {char: idx for idx, char in enumerate(ALPHABET)}

def label_to_int(char):
    return CHAR_MAPPING.get(char, 0)

def process_dataset(data_dir="Large_Captcha_Dataset", output_file="processed_data.pt"):
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    if not image_paths:
        image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    
    all_images = []
    all_labels = []
    
    print(f"Found {len(image_paths)} images. Starting pre-processing...")

    # We use a simplified version of your contour logic here
    for img_path in tqdm(image_paths):
        filename = os.path.basename(img_path).split('.')[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # --- Your Slicing Logic ---
        # Note: I'm using a simplified equal-width slice here for speed, 
        # but you can drop your full _contour_segments code here instead!
        h, w = img.shape
        chunk_w = w // 5
        for i in range(5):
            char = filename[i] if i < len(filename) else 'A'
            segment = img[:, i*chunk_w:(i+1)*chunk_w]
            segment = cv2.resize(segment, (32, 32))
            
            # Convert to float32 and normalize once
            all_images.append(segment.astype(np.float32) / 255.0)
            all_labels.append(label_to_int(char))

    if not all_images:
        print("No valid image segments found. Nothing to save.")
        return

    # Combine and shuffle so batches get a mixed character distribution
    combined = list(zip(all_images, all_labels))
    np.random.shuffle(combined)
    all_images, all_labels = zip(*combined)

    # Convert lists to massive tensors
    print("Saving to disk... this might take a minute.")
    data_dict = {
        'images': torch.tensor(np.array(all_images)).unsqueeze(1), # Shape: [N, 1, 32, 32]
        'labels': torch.tensor(all_labels, dtype=torch.long)        # Shape: [N]
    }
    
    torch.save(data_dict, output_file)
    print(f"Done! Created {output_file}")

if __name__ == "__main__":
    process_dataset()