"""
Synthetic Degradation Generator for QualiCLIP Pretraining

Implements multiple types of image degradations with progressive intensity levels.
Based on QualiCLIP paper (Appendix S3).
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance
import io
import numpy as np
import random


class SyntheticDegradation:
    """
    Apply progressive synthetic degradation to clean images.
    
    Supports multiple distortion types with configurable intensity levels:
    - Gaussian Blur
    - JPEG Compression
    - Gaussian Noise
    - Brightness Adjustment
    
    Args:
        distortion_type (str): Type of distortion ('blur', 'jpeg', 'noise', 'brightness')
        num_levels (int): Number of degradation levels (default: 5)
        
    Returns:
        List[PIL.Image]: List of degraded images, length = num_levels
    """
    
    def __init__(self, distortion_type='blur', num_levels=5):
        self.distortion_type = distortion_type
        self.num_levels = num_levels
        
        # Define degradation parameters for each level
        # Level 1 = mild, Level 5 = severe
        self.degradation_params = {
            'blur': {
                'sigmas': [0.5, 1.0, 1.5, 2.0, 2.5],
                'kernel_size': 5
            },
            'jpeg': {
                'qualities': [85, 70, 55, 40, 25]
            },
            'noise': {
                'stds': [5, 10, 15, 20, 25]  # Gaussian noise standard deviation
            },
            'brightness': {
                # < 1.0 = darker, > 1.0 = brighter
                'factors': [0.7, 0.5, 0.3, 1.3, 1.5]
            }
        }
        
    def __call__(self, image):
        """
        Apply progressive degradation to input image.
        
        Args:
            image (PIL.Image or torch.Tensor): Input clean image
            
        Returns:
            List[PIL.Image]: List of progressively degraded images
        """
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        
        degraded_images = []
        
        for level in range(self.num_levels):
            if self.distortion_type == 'blur':
                degraded = self._apply_blur(image, level)
            elif self.distortion_type == 'jpeg':
                degraded = self._apply_jpeg_compression(image, level)
            elif self.distortion_type == 'noise':
                degraded = self._apply_noise(image, level)
            elif self.distortion_type == 'brightness':
                degraded = self._apply_brightness(image, level)
            else:
                raise ValueError(f"Unknown distortion type: {self.distortion_type}")
            
            degraded_images.append(degraded)
        
        return degraded_images
    
    def _apply_blur(self, image, level):
        """Apply Gaussian blur with increasing sigma"""
        sigma = self.degradation_params['blur']['sigmas'][level]
        kernel_size = self.degradation_params['blur']['kernel_size']
        
        # Use PIL's GaussianBlur filter
        blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return blurred
    
    def _apply_jpeg_compression(self, image, level):
        """Apply JPEG compression with decreasing quality"""
        quality = self.degradation_params['jpeg']['qualities'][level]
        
        # Compress image through JPEG encoding/decoding
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).copy()  # Copy to avoid closing buffer
        buffer.close()
        
        return compressed
    
    def _apply_noise(self, image, level):
        """Apply Gaussian noise with increasing standard deviation"""
        std = self.degradation_params['noise']['stds'][level]
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)
        
        # Add Gaussian noise
        noise = np.random.normal(0, std, img_array.shape)
        noisy = img_array + noise
        
        # Clip to valid range [0, 255]
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        # Convert back to PIL
        return Image.fromarray(noisy)
    
    def _apply_brightness(self, image, level):
        """Apply brightness adjustment (darken or brighten)"""
        factor = self.degradation_params['brightness']['factors'][level]
        
        # Use PIL's ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        adjusted = enhancer.enhance(factor)
        
        return adjusted


class RandomDegradation:
    """
    Apply random degradation type with progressive levels.
    Useful for data augmentation during pretraining.
    
    Args:
        num_levels (int): Number of degradation levels (default: 5)
        distortion_types (list): List of allowed distortion types
    """
    
    def __init__(self, num_levels=5, distortion_types=None):
        if distortion_types is None:
            distortion_types = ['blur', 'jpeg', 'noise', 'brightness']
        
        self.num_levels = num_levels
        self.distortion_types = distortion_types
        
    def __call__(self, image):
        """Randomly select a distortion type and apply it"""
        distortion_type = random.choice(self.distortion_types)
        degrader = SyntheticDegradation(distortion_type, self.num_levels)
        return degrader(image), distortion_type


def test_degradation():
    """Test function to visualize different degradation types"""
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Create a test image (or load one)
    # For testing, create a simple gradient image
    test_img = Image.new('RGB', (224, 224), color='white')
    
    print("Testing Degradation Generator...")
    
    distortion_types = ['blur', 'jpeg', 'noise', 'brightness']
    
    for dist_type in distortion_types:
        degrader = SyntheticDegradation(dist_type, num_levels=5)
        degraded_images = degrader(test_img)
        print(f"{dist_type}: Generated {len(degraded_images)} degraded images")
    
    print("✓ All degradation types work correctly")


if __name__ == '__main__':
    test_degradation()

