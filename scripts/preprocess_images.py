"""
Image Preprocessing Script
==========================
Pre-resize all images to 128x128 and save to data/processed/images/
This eliminates on-the-fly resizing during training for faster epochs.

Usage:
    python scripts/preprocess_images.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.constants import (
    WIKIART_DIR,
    PROCESSED_IMAGES_DIR,
    IMAGE_SIZE,
    PROCESSED_DATA_DIR
)


def resize_and_save_image(args):
    """Resize a single image and save to processed folder."""
    src_path, dst_path, size = args
    
    try:
        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already processed
        if dst_path.exists():
            return True, src_path, "skipped"
        
        # Load, resize, and save
        with Image.open(src_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize using high-quality resampling
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            
            # Save as JPEG with good quality
            img_resized.save(dst_path, 'JPEG', quality=95)
        
        return True, src_path, "processed"
    
    except Exception as e:
        return False, src_path, str(e)


def main():
    print("=" * 70)
    print("IMAGE PREPROCESSING: Resize to 128x128")
    print("=" * 70)
    
    # Load selected images from splits
    splits_dir = PROCESSED_DATA_DIR / "splits"
    all_paintings = set()
    
    for split_file in ["train.json", "val.json", "test.json"]:
        split_path = splits_dir / split_file
        if split_path.exists():
            with open(split_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for painting in data['paintings']:
                    all_paintings.add((painting['style'], painting['painting']))
    
    print(f"\nFound {len(all_paintings)} unique paintings to process")
    print(f"Target size: {IMAGE_SIZE}")
    print(f"Source: {WIKIART_DIR}")
    print(f"Destination: {PROCESSED_IMAGES_DIR}")
    
    # Create output directory
    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare tasks
    tasks = []
    for style, painting in all_paintings:
        src_path = WIKIART_DIR / style / f"{painting}.jpg"
        dst_path = PROCESSED_IMAGES_DIR / style / f"{painting}.jpg"
        
        if src_path.exists():
            tasks.append((src_path, dst_path, IMAGE_SIZE))
        else:
            print(f"Warning: Source not found: {src_path}")
    
    print(f"\nProcessing {len(tasks)} images...")
    print("-" * 70)
    
    # Process images with thread pool
    processed = 0
    skipped = 0
    errors = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(resize_and_save_image, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Resizing images") as pbar:
            for future in as_completed(futures):
                success, path, status = future.result()
                if success:
                    if status == "processed":
                        processed += 1
                    else:
                        skipped += 1
                else:
                    errors += 1
                    print(f"\nError: {path.name}: {status}")
                pbar.update(1)
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Processed: {processed}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Errors: {errors}")
    
    # Calculate space saved
    if processed > 0:
        sample_files = list(PROCESSED_IMAGES_DIR.rglob("*.jpg"))[:10]
        if sample_files:
            avg_size = sum(f.stat().st_size for f in sample_files) / len(sample_files)
            total_size_mb = (avg_size * len(tasks)) / (1024 * 1024)
            print(f"\nEstimated processed images size: {total_size_mb:.1f} MB")
    
    print(f"\nImages saved to: {PROCESSED_IMAGES_DIR}")
    print("\nNext: Update data_loader.py to use preprocessed images")


if __name__ == "__main__":
    main()
