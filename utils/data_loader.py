"""
PyTorch DataLoader for ArtEmis Caption Generation

This module provides PyTorch Dataset and DataLoader classes for efficiently
loading images and captions during training, validation, and testing.

Key Features:
- Efficient batch loading with multiprocessing
- On-the-fly image preprocessing
- Caption tokenization and padding
- Memory-efficient design for CPU training
- Support for data augmentation

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np

from utils.constants import (
    SPLITS_DIR,
    WIKIART_DIR,
    PROCESSED_IMAGES_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGE_SIZE,
    MAX_CAPTION_LENGTH
)
from utils.image_preprocessing import ImagePreprocessor
from utils.text_preprocessing import TextPreprocessor


class ArtEmisDataset(Dataset):
    """
    PyTorch Dataset for ArtEmis image-caption pairs.
    
    Loads images and captions on-the-fly to minimize memory usage.
    Applies image preprocessing and caption tokenization.
    
    Args:
        split_file: Path to split JSON file (train.json, val.json, or test.json)
        text_preprocessor: TextPreprocessor instance with built vocabulary
        image_preprocessor: ImagePreprocessor instance for image transforms
        is_train: If True, apply training transforms (augmentation)
        max_captions_per_image: Maximum number of captions to use per image (None = all)
    
    Returns:
        Tuple of (image_tensor, caption_tokens, caption_length, painting_name)
    """
    
    def __init__(
        self,
        split_file: Path,
        text_preprocessor: TextPreprocessor,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        is_train: bool = False,
        max_captions_per_image: Optional[int] = None,
        use_preprocessed: bool = True
    ):
        """Initialize the dataset."""
        self.split_file = Path(split_file)
        self.text_preprocessor = text_preprocessor
        self.is_train = is_train
        self.max_captions_per_image = max_captions_per_image
        
        # Check if preprocessed images are available
        self.use_preprocessed = use_preprocessed and PROCESSED_IMAGES_DIR.exists() and any(PROCESSED_IMAGES_DIR.iterdir())
        
        # Create image preprocessor with skip_resize if using preprocessed images
        if image_preprocessor is not None:
            self.image_preprocessor = image_preprocessor
        else:
            self.image_preprocessor = ImagePreprocessor(skip_resize=self.use_preprocessed)
        
        # Load split data
        with open(self.split_file, 'r', encoding='utf-8') as f:
            self.split_data = json.load(f)
        
        # Build index of all image-caption pairs
        self.samples = []
        for painting_data in self.split_data['paintings']:
            painting_name = painting_data['painting']
            style = painting_data['style']
            captions = painting_data['captions']
            
            # Limit captions per image if specified
            if self.max_captions_per_image:
                captions = captions[:self.max_captions_per_image]
            
            # Create a sample for each caption
            for caption in captions:
                self.samples.append({
                    'painting': painting_name,
                    'style': style,
                    'caption': caption
                })
        
        print(f"  - Loaded {len(self.samples)} image-caption pairs from {self.split_file.name}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """
        Get a single image-caption pair.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of:
            - image_tensor: Preprocessed image tensor (C, H, W)
            - caption_tokens: Tokenized caption as tensor (max_length,)
            - caption_length: Actual caption length before padding
            - painting_name: Name of the painting
        """
        sample = self.samples[idx]
        painting_name = sample['painting']
        style = sample['style']
        caption_text = sample['caption']
        
        # Construct image path - use preprocessed images if available
        if self.use_preprocessed:
            image_path = PROCESSED_IMAGES_DIR / style / f"{painting_name}.jpg"
        else:
            image_path = WIKIART_DIR / style / f"{painting_name}.jpg"
        
        # Load and preprocess image
        try:
            image_tensor = self.image_preprocessor.load_image(
                image_path,
                is_train=self.is_train
            )
        except Exception as e:
            # Fallback: create black image if loading fails
            print(f"Warning: Failed to load {image_path}: {e}")
            image_tensor = torch.zeros(3, *IMAGE_SIZE)
        
        # Tokenize caption
        caption_tokens = self.text_preprocessor.encode(
            caption_text,
            add_special_tokens=True,
            pad=True
        )
        
        # Calculate actual caption length (excluding padding)
        caption_length = 0
        for token in caption_tokens:
            if token == self.text_preprocessor.PAD_IDX:
                break
            caption_length += 1
        
        # Convert to tensors
        caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)
        
        return image_tensor, caption_tensor, caption_length, painting_name
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_samples': len(self.samples),
            'num_paintings': len(set(s['painting'] for s in self.samples)),
            'num_styles': len(set(s['style'] for s in self.samples)),
            'split': self.split_data['split'],
            'avg_captions_per_painting': len(self.samples) / len(set(s['painting'] for s in self.samples))
        }
        return stats


def collate_fn(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Dictionary with batched tensors:
        - images: (batch_size, C, H, W)
        - captions: (batch_size, max_length)
        - lengths: (batch_size,)
        - painting_names: List of painting names
    """
    # Separate components
    images, captions, lengths, painting_names = zip(*batch)
    
    # Stack images and captions
    images = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return {
        'images': images,
        'captions': captions,
        'lengths': lengths,
        'painting_names': list(painting_names)
    }


def create_data_loaders(
    text_preprocessor: TextPreprocessor,
    image_preprocessor: Optional[ImagePreprocessor] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    max_captions_per_image: Optional[int] = None,
    splits: List[str] = ['train', 'val', 'test']
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for all splits.
    
    Args:
        text_preprocessor: TextPreprocessor with built vocabulary
        image_preprocessor: ImagePreprocessor for image transforms
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        max_captions_per_image: Maximum captions per image (None = all)
        splits: List of splits to create loaders for
    
    Returns:
        Dictionary mapping split name to DataLoader
    """
    if image_preprocessor is None:
        image_preprocessor = ImagePreprocessor()
    
    print("\n" + "=" * 70)
    print("CREATING DATA LOADERS")
    print("=" * 70)
    
    data_loaders = {}
    
    for split_name in splits:
        split_file = SPLITS_DIR / f"{split_name}.json"
        
        if not split_file.exists():
            print(f"\n⚠ Warning: {split_file} not found, skipping {split_name}")
            continue
        
        print(f"\n{split_name.upper()} DataLoader:")
        
        # Create dataset
        is_train = (split_name == 'train')
        dataset = ArtEmisDataset(
            split_file=split_file,
            text_preprocessor=text_preprocessor,
            image_preprocessor=image_preprocessor,
            is_train=is_train,
            max_captions_per_image=max_captions_per_image
        )
        
        # Create dataloader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,  # Shuffle training data only
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,  # CPU training, no benefit from pinned memory
            drop_last=is_train  # Drop last incomplete batch in training
        )
        
        # Display statistics
        stats = dataset.get_statistics()
        print(f"  - Samples: {stats['num_samples']:,}")
        print(f"  - Paintings: {stats['num_paintings']:,}")
        print(f"  - Batches: {len(data_loader):,}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Shuffle: {is_train}")
        
        data_loaders[split_name] = data_loader
    
    return data_loaders


# Alias for backward compatibility
create_dataloaders = create_data_loaders


def test_data_loader():
    """
    Test the DataLoader with a small batch.
    """
    print("=" * 70)
    print("TESTING DATA LOADER")
    print("=" * 70)
    
    # Load vocabulary
    from utils.constants import PROCESSED_DIR
    
    vocab_path = PROCESSED_DIR / "vocabulary.json"
    if not vocab_path.exists():
        print(f"\n❌ Error: {vocab_path} not found")
        print("   Please run utils/text_preprocessing.py first")
        return
    
    print(f"\n✓ Loading vocabulary from: {vocab_path}")
    text_preprocessor = TextPreprocessor()
    text_preprocessor.load_vocabulary(vocab_path)
    
    # Create image preprocessor
    image_preprocessor = ImagePreprocessor(
        image_size=IMAGE_SIZE,
        normalize=True,
        augment=False
    )
    
    # Create data loaders
    data_loaders = create_data_loaders(
        text_preprocessor=text_preprocessor,
        image_preprocessor=image_preprocessor,
        batch_size=4,  # Small batch for testing
        num_workers=0,  # No multiprocessing for testing
        max_captions_per_image=None,
        splits=['train', 'val']
    )
    
    # Test training loader
    print("\n" + "=" * 70)
    print("TESTING BATCH LOADING")
    print("=" * 70)
    
    if 'train' in data_loaders:
        train_loader = data_loaders['train']
        
        # Get first batch
        batch = next(iter(train_loader))
        
        print(f"\nBatch contents:")
        print(f"  - Images shape: {batch['images'].shape}")
        print(f"  - Captions shape: {batch['captions'].shape}")
        print(f"  - Lengths shape: {batch['lengths'].shape}")
        print(f"  - Painting names: {len(batch['painting_names'])}")
        
        print(f"\nImage statistics:")
        print(f"  - Min: {batch['images'].min():.3f}")
        print(f"  - Max: {batch['images'].max():.3f}")
        print(f"  - Mean: {batch['images'].mean():.3f}")
        print(f"  - Std: {batch['images'].std():.3f}")
        
        print(f"\nCaption lengths:")
        print(f"  - Min: {batch['lengths'].min()}")
        print(f"  - Max: {batch['lengths'].max()}")
        print(f"  - Mean: {batch['lengths'].float().mean():.1f}")
        
        # Decode first caption
        first_caption_tokens = batch['captions'][0].tolist()
        decoded_caption = text_preprocessor.decode(first_caption_tokens)
        
        print(f"\nFirst caption (decoded):")
        print(f"  '{decoded_caption}'")
        print(f"\nFirst caption (tokens):")
        print(f"  {first_caption_tokens[:20]}...")
        
        print("\n" + "=" * 70)
        print("✅ DATA LOADER TEST PASSED")
        print("=" * 70)
        
    else:
        print("\n⚠ No train loader created, skipping batch test")


if __name__ == "__main__":
    test_data_loader()
