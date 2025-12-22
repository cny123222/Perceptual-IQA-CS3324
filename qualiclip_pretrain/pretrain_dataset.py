"""
Dataset for QualiCLIP Self-Supervised Pretraining

Loads clean images and generates overlapping random crops for contrastive learning.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import json
import random


class QualityAwarePretrainDataset(Dataset):
    """
    Dataset for quality-aware self-supervised pretraining.
    
    For each image, generates two random overlapping crops that will be:
    1. Degraded with the same distortion type and levels
    2. Used for consistency loss (same source, same degradation → similar features)
    
    Args:
        image_paths (list): List of paths to clean images
        crop_size (int): Size of random crops (default: 224)
        base_size (int): Resize images to this size before cropping (default: 512)
        overlap_ratio (float): Minimum overlap between two crops (default: 0.5)
        transform (callable): Optional transform to apply after cropping
    """
    
    def __init__(self, image_paths, crop_size=224, base_size=512, 
                 overlap_ratio=0.5, transform=None):
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.base_size = base_size
        self.overlap_ratio = overlap_ratio
        self.transform = transform
        
        # Base transforms: resize and convert to RGB
        self.base_transform = T.Compose([
            T.Resize((base_size, base_size)),
            T.Lambda(lambda img: img.convert('RGB'))
        ])
        
        print(f"Pretrain Dataset: {len(self.image_paths)} images")
        print(f"  Crop size: {crop_size}x{crop_size}")
        print(f"  Base size: {base_size}x{base_size}")
        print(f"  Overlap ratio: {overlap_ratio}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            crop1 (PIL.Image): First random crop
            crop2 (PIL.Image): Second random crop (overlapping with crop1)
            image_idx (int): Index of the source image
        """
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return first image as fallback
            image = Image.open(self.image_paths[0]).convert('RGB')
        
        # Resize to base size
        image = self.base_transform(image)
        
        # Generate two overlapping random crops
        crop1, crop2 = self._generate_overlapping_crops(image)
        
        # Apply additional transforms if provided
        if self.transform is not None:
            crop1 = self.transform(crop1)
            crop2 = self.transform(crop2)
        
        return crop1, crop2, idx
    
    def _generate_overlapping_crops(self, image):
        """
        Generate two random crops with guaranteed overlap.
        
        Strategy:
        1. Generate first crop at random location
        2. Generate second crop near the first one to ensure overlap
        """
        img_w, img_h = image.size
        crop_size = self.crop_size
        
        # First crop: fully random
        max_x = img_w - crop_size
        max_y = img_h - crop_size
        
        x1 = random.randint(0, max_x)
        y1 = random.randint(0, max_y)
        
        crop1 = TF.crop(image, y1, x1, crop_size, crop_size)
        
        # Second crop: ensure overlap with first crop
        # Overlap ratio determines how close the crops should be
        max_offset = int(crop_size * (1 - self.overlap_ratio))
        
        # Random offset from first crop
        x_offset = random.randint(-max_offset, max_offset)
        y_offset = random.randint(-max_offset, max_offset)
        
        x2 = max(0, min(max_x, x1 + x_offset))
        y2 = max(0, min(max_y, y1 + y_offset))
        
        crop2 = TF.crop(image, y2, x2, crop_size, crop_size)
        
        return crop1, crop2


def load_koniq_train_images(koniq_path):
    """
    Load KonIQ-10k training set image paths.
    
    Args:
        koniq_path (str): Path to koniq-10k directory
        
    Returns:
        list: List of image paths
    """
    # Load split file
    train_split_file = os.path.join(koniq_path, 'koniq_train.json')
    
    if not os.path.exists(train_split_file):
        print(f"Warning: {train_split_file} not found. Creating from scratch...")
        # If split file doesn't exist, use all images
        img_dir = os.path.join(koniq_path, '512x384')
        if not os.path.exists(img_dir):
            img_dir = koniq_path  # Try root directory
        
        image_files = [f for f in os.listdir(img_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths = [os.path.join(img_dir, f) for f in image_files]
    else:
        # Load from JSON
        with open(train_split_file, 'r') as f:
            data = json.load(f)
        
        # Extract image paths
        if isinstance(data, list):
            # Format: [{"image": "path/to/img.jpg", "score": X}, ...]
            image_paths = []
            for item in data:
                img_name = os.path.basename(item['image'])
                # Try different possible locations
                possible_paths = [
                    os.path.join(koniq_path, 'train', img_name),  # Actual structure: train/xxx.jpg
                    os.path.join(koniq_path, '512x384', img_name),
                    os.path.join(koniq_path, 'images', img_name),
                    os.path.join(koniq_path, img_name)
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        image_paths.append(path)
                        break
        else:
            raise ValueError(f"Unexpected format in {train_split_file}")
    
    # Filter out non-existent paths
    image_paths = [p for p in image_paths if os.path.exists(p)]
    
    print(f"Loaded {len(image_paths)} training images from KonIQ-10k")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {koniq_path}. Please check the path.")
    
    return image_paths


def load_image_folder(folder_path, extensions=('.jpg', '.jpeg', '.png')):
    """
    Load all images from a folder.
    
    Args:
        folder_path (str): Path to folder containing images
        extensions (tuple): Valid image file extensions
        
    Returns:
        list: List of image paths
    """
    image_paths = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    
    print(f"Loaded {len(image_paths)} images from {folder_path}")
    return image_paths


def test_dataset():
    """Test the dataset with a sample directory"""
    import matplotlib.pyplot as plt
    
    print("Testing Pretrain Dataset...")
    
    # For testing, create dummy image paths or use a sample directory
    koniq_path = '/root/Perceptual-IQA-CS3324/koniq-10k'
    
    if os.path.exists(koniq_path):
        try:
            image_paths = load_koniq_train_images(koniq_path)
            
            # Create dataset
            dataset = QualityAwarePretrainDataset(
                image_paths,
                crop_size=224,
                base_size=512,
                overlap_ratio=0.5
            )
            
            # Test loading
            crop1, crop2, idx = dataset[0]
            
            print(f"✓ Dataset test passed")
            print(f"  Crop 1 size: {crop1.size}")
            print(f"  Crop 2 size: {crop2.size}")
            print(f"  Image index: {idx}")
            
            # Visualize (optional)
            # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # axes[0].imshow(crop1)
            # axes[0].set_title("Crop 1")
            # axes[1].imshow(crop2)
            # axes[1].set_title("Crop 2")
            # plt.savefig('test_crops.png')
            # print("  Saved visualization to test_crops.png")
            
        except Exception as e:
            print(f"Warning: Could not test with KonIQ data: {e}")
            print("This is expected if KonIQ dataset is not available yet")
    else:
        print(f"KonIQ path not found: {koniq_path}")
        print("This is expected if KonIQ dataset is not available yet")


if __name__ == '__main__':
    test_dataset()

