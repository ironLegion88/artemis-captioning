"""
Image Preprocessing Module for ArtEmis Caption Generation

This module handles image loading, transformation, and preprocessing for the
ArtEmis dataset. Implements efficient on-the-fly loading to minimize memory usage.

Key Features:
- Resize images to 128x128 (matches constants.IMAGE_SIZE)
- Normalize with ImageNet statistics (transfer learning compatibility)
- On-the-fly loading (memory efficient for 5000 images)
- RGB conversion (handles grayscale images)
- Data augmentation support (optional)

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np

from utils.constants import (
    IMAGE_SIZE,
    WIKIART_DIR,
    DEVICE
)


class ImagePreprocessor:
    """
    Handles image preprocessing for the ArtEmis dataset.
    
    Features:
    - Resizes images to target size (default: 128x128)
    - Normalizes using ImageNet mean/std (for transfer learning)
    - Converts grayscale to RGB
    - Supports both training and validation transforms
    - On-the-fly loading to save memory
    
    Attributes:
        image_size (Tuple[int, int]): Target image dimensions (H, W)
        train_transform (transforms.Compose): Training transformations
        val_transform (transforms.Compose): Validation transformations
    """
    
    # ImageNet normalization statistics (standard for pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self, 
        image_size: Tuple[int, int] = IMAGE_SIZE,
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            image_size: Target image size as (height, width). Default from constants.
            normalize: Whether to normalize with ImageNet statistics. Default True.
            augment: Whether to apply data augmentation during training. Default False.
        """
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        
        # Build transformation pipelines
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()
    
    def _build_train_transform(self) -> transforms.Compose:
        """
        Build transformation pipeline for training images.
        
        Includes optional data augmentation if self.augment is True:
        - Random horizontal flip (50% chance)
        - Random rotation (±10 degrees)
        - Color jitter (brightness, contrast, saturation)
        
        Returns:
            Composed transformation pipeline for training
        """
        transform_list = []
        
        # Resize to target size
        transform_list.append(transforms.Resize(self.image_size))
        
        # Data augmentation (optional)
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            ])
        
        # Convert to tensor (scales to [0, 1])
        transform_list.append(transforms.ToTensor())
        
        # Normalize with ImageNet statistics
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _build_val_transform(self) -> transforms.Compose:
        """
        Build transformation pipeline for validation/test images.
        
        No augmentation, only:
        - Resize to target size
        - Convert to tensor
        - Normalize (if enabled)
        
        Returns:
            Composed transformation pipeline for validation/testing
        """
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD
                )
            )
        
        return transforms.Compose(transform_list)
    
    def load_image(
        self, 
        image_path: Union[str, Path],
        is_train: bool = False
    ) -> torch.Tensor:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file (absolute or relative)
            is_train: If True, apply training transforms (with augmentation).
                     If False, apply validation transforms. Default False.
        
        Returns:
            Preprocessed image as torch.Tensor with shape (C, H, W)
            - C = 3 (RGB channels)
            - H, W = self.image_size
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            PIL.UnidentifiedImageError: If file is not a valid image
        """
        # Convert to Path object
        image_path = Path(image_path)
        
        # Check if file exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply appropriate transform
        transform = self.train_transform if is_train else self.val_transform
        image_tensor = transform(image)
        
        return image_tensor
    
    def load_batch(
        self,
        image_paths: list,
        is_train: bool = False
    ) -> torch.Tensor:
        """
        Load and preprocess a batch of images.
        
        Args:
            image_paths: List of paths to image files
            is_train: If True, apply training transforms. Default False.
        
        Returns:
            Batch of preprocessed images as torch.Tensor with shape (B, C, H, W)
            - B = batch size (len(image_paths))
            - C = 3 (RGB channels)
            - H, W = self.image_size
        
        Raises:
            FileNotFoundError: If any image file doesn't exist
        """
        # Load all images in the batch
        image_tensors = []
        for image_path in image_paths:
            image_tensor = self.load_image(image_path, is_train=is_train)
            image_tensors.append(image_tensor)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(image_tensors, dim=0)
        
        return batch_tensor
    
    def denormalize(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization to get back to [0, 1] range.
        
        Useful for visualization after preprocessing.
        
        Args:
            image_tensor: Normalized tensor with shape (C, H, W) or (B, C, H, W)
        
        Returns:
            Denormalized tensor in [0, 1] range
        """
        if not self.normalize:
            return image_tensor
        
        # Convert to numpy for easier manipulation
        is_batch = image_tensor.ndim == 4
        
        if is_batch:
            # Shape: (B, C, H, W)
            mean = torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        else:
            # Shape: (C, H, W)
            mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        
        # Denormalize: x = x_norm * std + mean
        denormalized = image_tensor * std + mean
        
        # Clip to [0, 1] range
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized
    
    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """
        Convert a normalized tensor back to PIL Image for visualization.
        
        Args:
            image_tensor: Tensor with shape (C, H, W), possibly normalized
        
        Returns:
            PIL Image in RGB mode
        """
        # Denormalize if needed
        denormalized = self.denormalize(image_tensor)
        
        # Convert to numpy array (H, W, C)
        image_np = denormalized.permute(1, 2, 0).cpu().numpy()
        
        # Scale to [0, 255] and convert to uint8
        image_np = (image_np * 255).astype(np.uint8)
        
        # Create PIL Image
        pil_image = Image.fromarray(image_np, mode='RGB')
        
        return pil_image
    
    def get_image_path(self, style: str, filename: str) -> Path:
        """
        Construct full path to an image in the WikiArt dataset.
        
        Args:
            style: Art style directory name (e.g., "Impressionism")
            filename: Image filename (e.g., "123.jpg")
        
        Returns:
            Full path to the image file
        """
        return WIKIART_DIR / style / filename
    
    def validate_image(self, image_path: Union[str, Path]) -> bool:
        """
        Check if an image file is valid and can be loaded.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            True if image is valid, False otherwise
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return False
            
            # Try to open and verify
            with Image.open(image_path) as img:
                img.verify()
            
            return True
            
        except Exception:
            return False


def test_preprocessing():
    """
    Test the image preprocessing pipeline with sample images.
    """
    print("=" * 70)
    print("TESTING IMAGE PREPROCESSING")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        image_size=IMAGE_SIZE,
        normalize=True,
        augment=False
    )
    
    print(f"\n✓ ImagePreprocessor initialized")
    print(f"  - Image size: {preprocessor.image_size}")
    print(f"  - Normalize: {preprocessor.normalize}")
    print(f"  - Augment: {preprocessor.augment}")
    
    # Find a sample image
    sample_styles = ["Impressionism", "Realism", "Expressionism"]
    sample_image_path = None
    
    for style in sample_styles:
        style_dir = WIKIART_DIR / style
        if style_dir.exists():
            images = list(style_dir.glob("*.jpg"))
            if images:
                sample_image_path = images[0]
                break
    
    if sample_image_path is None:
        print("\n⚠ No sample images found. Please ensure WikiArt dataset is downloaded.")
        return
    
    print(f"\n✓ Found sample image: {sample_image_path.name}")
    print(f"  - Style: {sample_image_path.parent.name}")
    
    # Test single image loading
    try:
        image_tensor = preprocessor.load_image(sample_image_path, is_train=False)
        print(f"\n✓ Image loaded successfully")
        print(f"  - Tensor shape: {image_tensor.shape}")
        print(f"  - Tensor dtype: {image_tensor.dtype}")
        print(f"  - Value range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        
        # Test batch loading
        batch_paths = [sample_image_path] * 4  # Duplicate for batch test
        batch_tensor = preprocessor.load_batch(batch_paths, is_train=False)
        print(f"\n✓ Batch loading successful")
        print(f"  - Batch shape: {batch_tensor.shape}")
        
        # Test denormalization
        denorm_tensor = preprocessor.denormalize(image_tensor)
        print(f"\n✓ Denormalization successful")
        print(f"  - Denorm range: [{denorm_tensor.min():.3f}, {denorm_tensor.max():.3f}]")
        
        # Test PIL conversion
        pil_image = preprocessor.tensor_to_pil(image_tensor)
        print(f"\n✓ Tensor to PIL conversion successful")
        print(f"  - PIL size: {pil_image.size}")
        print(f"  - PIL mode: {pil_image.mode}")
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_preprocessing()
