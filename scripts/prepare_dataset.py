"""
Prepare Dataset for Specific Size
==================================
Preprocess images for different training configurations.

Usage:
    python scripts/prepare_dataset.py --num-images 5000   # For second laptop
    python scripts/prepare_dataset.py --num-images 15000  # For Google Colab
    python scripts/prepare_dataset.py --num-images 15000 --output-dir colab_data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

def update_constants(num_images):
    """Temporarily update NUM_IMAGES_SUBSET in constants."""
    constants_path = Path("utils/constants.py")
    
    with open(constants_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace NUM_IMAGES_SUBSET
    import re
    new_content = re.sub(
        r'NUM_IMAGES_SUBSET = \d+',
        f'NUM_IMAGES_SUBSET = {num_images}',
        content
    )
    
    with open(constants_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"  Updated NUM_IMAGES_SUBSET to {num_images}")
    return content  # Return original for restoration


def run_preprocessing():
    """Run the preprocessing pipeline."""
    import subprocess
    
    scripts = [
        ("Analyzing dataset...", "scripts/analyze_dataset.py"),
        ("Creating splits...", "scripts/create_splits.py"),
        ("Resizing images...", "scripts/preprocess_images.py"),
    ]
    
    for desc, script in scripts:
        print(f"\n{desc}")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed: {script}")


def copy_to_output(output_dir):
    """Copy processed data to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy data/processed
    src = Path("data/processed")
    dst = output_path / "data" / "processed"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    
    # Copy data/embeddings if exists
    emb_src = Path("data/embeddings")
    if emb_src.exists():
        emb_dst = output_path / "data" / "embeddings"
        if emb_dst.exists():
            shutil.rmtree(emb_dst)
        shutil.copytree(emb_src, emb_dst)
    
    print(f"\n✓ Data copied to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for specific size")
    parser.add_argument("--num-images", type=int, required=True,
                       help="Number of images to include (e.g., 5000, 15000)")
    parser.add_argument("--output-dir", type=str,
                       help="Optional: Copy processed data to this directory")
    parser.add_argument("--skip-if-exists", action="store_true",
                       help="Skip preprocessing if correct number of images already exist")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"DATASET PREPARATION: {args.num_images:,} images")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we already have the right number
    selected_file = Path("data/processed/selected_images.json")
    if args.skip_if_exists and selected_file.exists():
        with open(selected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        current_count = data.get('total_selected', 0)
        if current_count == args.num_images:
            print(f"\n✓ Already have {current_count} images preprocessed. Skipping.")
            if args.output_dir:
                copy_to_output(args.output_dir)
            return
    
    # Update constants
    print("\nUpdating constants...")
    original_content = update_constants(args.num_images)
    
    try:
        # Run preprocessing
        run_preprocessing()
        
        # Verify
        with open(selected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        actual_count = data.get('total_selected', 0)
        print(f"\n✓ Preprocessed {actual_count:,} images")
        
        # Copy if output dir specified
        if args.output_dir:
            copy_to_output(args.output_dir)
        
    finally:
        # Restore original constants (optional - keep the new value)
        pass
    
    # Summary
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    
    # Show sizes
    images_dir = Path("data/processed/images")
    total_size = sum(f.stat().st_size for f in images_dir.rglob('*.jpg')) / (1024 * 1024)
    print(f"Images: {args.num_images:,}")
    print(f"Total size: {total_size:.1f} MB")
    
    if args.output_dir:
        print(f"Output: {args.output_dir}/")
    else:
        print("Output: data/processed/")


if __name__ == "__main__":
    main()
