"""
Prepare Data Package for Transfer
==================================
Creates zip packages for second laptop and Google Colab.

Usage:
    python scripts/prepare_data_package.py --target laptop2
    python scripts/prepare_data_package.py --target colab
    python scripts/prepare_data_package.py --target both
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def verify_data():
    """Verify all required data exists."""
    print("Verifying data...")
    
    required = {
        'data/processed/vocabulary.json': 'Vocabulary file',
        'data/processed/splits/train.json': 'Train split',
        'data/processed/splits/val.json': 'Val split',
        'data/processed/splits/test.json': 'Test split',
        'data/processed/images': 'Preprocessed images',
        'data/processed/captions': 'Caption files',
    }
    
    optional = {
        'data/embeddings/glove_embeddings.npy': 'GloVe embeddings',
        'data/embeddings/word2vec_embeddings.npy': 'Word2Vec embeddings',
        'data/embeddings/tfidf_embeddings.npy': 'TF-IDF embeddings',
    }
    
    all_ok = True
    
    for path, desc in required.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                count = sum(1 for _ in Path(path).rglob('*') if _.is_file())
                print(f"  ✓ {desc}: {count} files")
            else:
                size = os.path.getsize(path) / 1024
                print(f"  ✓ {desc}: {size:.1f} KB")
        else:
            print(f"  ✗ {desc}: MISSING ({path})")
            all_ok = False
    
    print("\nOptional files:")
    for path, desc in optional.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {desc}: {size:.1f} MB")
        else:
            print(f"  - {desc}: Not found (optional)")
    
    return all_ok


def get_folder_size(path):
    """Get total size of folder in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def create_laptop2_package():
    """Create package for second laptop."""
    print("\n" + "=" * 60)
    print("Creating Second Laptop Package")
    print("=" * 60)
    
    output_dir = Path("packages")
    output_dir.mkdir(exist_ok=True)
    
    staging_dir = output_dir / "laptop2_staging"
    
    # Clean up staging
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    
    # Create staging directory structure
    (staging_dir / "data" / "processed").mkdir(parents=True)
    (staging_dir / "data" / "embeddings").mkdir(parents=True)
    
    # Copy data files
    items_to_copy = [
        ("data/processed/images", "data/processed/images"),
        ("data/processed/splits", "data/processed/splits"),
        ("data/processed/captions", "data/processed/captions"),
        ("data/processed/vocabulary.json", "data/processed/vocabulary.json"),
    ]
    
    # Also copy embeddings if available
    if os.path.exists("data/embeddings"):
        items_to_copy.append(("data/embeddings", "data/embeddings"))
    
    for src, dst in items_to_copy:
        src_path = Path(src)
        dst_path = staging_dir / dst
        
        if src_path.exists():
            print(f"  Copying {src}...")
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
    
    # Calculate size
    total_size = get_folder_size(staging_dir)
    print(f"\nStaging size: {total_size:.1f} MB")
    
    # Create zip
    zip_path = output_dir / "laptop2_data"
    print(f"Creating zip: {zip_path}.zip")
    shutil.make_archive(str(zip_path), 'zip', staging_dir)
    
    zip_size = os.path.getsize(f"{zip_path}.zip") / (1024 * 1024)
    print(f"✓ Created: {zip_path}.zip ({zip_size:.1f} MB)")
    
    # Cleanup staging
    shutil.rmtree(staging_dir)
    
    print(f"\n📦 Package ready: packages/laptop2_data.zip")
    print("Transfer this to second laptop and extract to artemis-captioning/data/")
    
    return str(zip_path) + ".zip"


def create_colab_package():
    """Create package for Google Colab."""
    print("\n" + "=" * 60)
    print("Creating Google Colab Package")
    print("=" * 60)
    
    output_dir = Path("packages")
    output_dir.mkdir(exist_ok=True)
    
    staging_dir = output_dir / "colab_staging"
    
    # Clean up staging
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    
    staging_dir.mkdir(parents=True)
    
    # Copy data files
    items_to_copy = [
        ("data/processed/images", "data/processed/images"),
        ("data/processed/splits", "data/processed/splits"),
        ("data/processed/captions", "data/processed/captions"),
        ("data/processed/vocabulary.json", "data/processed/vocabulary.json"),
        ("utils", "utils"),
        ("models", "models"),
        ("train.py", "train.py"),
    ]
    
    # Also copy embeddings if available
    if os.path.exists("data/embeddings"):
        items_to_copy.append(("data/embeddings", "data/embeddings"))
    
    for src, dst in items_to_copy:
        src_path = Path(src)
        dst_path = staging_dir / dst
        
        if src_path.exists():
            print(f"  Copying {src}...")
            if src_path.is_dir():
                # Skip __pycache__
                def ignore_pycache(d, files):
                    return ['__pycache__'] if '__pycache__' in files else []
                shutil.copytree(src_path, dst_path, ignore=ignore_pycache, dirs_exist_ok=True)
            else:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
    
    # Calculate size
    total_size = get_folder_size(staging_dir)
    print(f"\nStaging size: {total_size:.1f} MB")
    
    # Create zip
    zip_path = output_dir / "colab_package"
    print(f"Creating zip: {zip_path}.zip")
    shutil.make_archive(str(zip_path), 'zip', staging_dir)
    
    zip_size = os.path.getsize(f"{zip_path}.zip") / (1024 * 1024)
    print(f"✓ Created: {zip_path}.zip ({zip_size:.1f} MB)")
    
    # Cleanup staging
    shutil.rmtree(staging_dir)
    
    print(f"\n📦 Package ready: packages/colab_package.zip")
    print("Upload to Google Drive → artemis-captioning/ and extract")
    
    return str(zip_path) + ".zip"


def main():
    parser = argparse.ArgumentParser(description="Prepare data packages for transfer")
    parser.add_argument("--target", type=str, choices=['laptop2', 'colab', 'both'],
                       default='both', help="Target platform(s)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATA PACKAGE PREPARATION")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verify data first
    if not verify_data():
        print("\n❌ Missing required files! Run preprocessing first.")
        return 1
    
    packages = []
    
    if args.target in ['laptop2', 'both']:
        pkg = create_laptop2_package()
        packages.append(('Second Laptop', pkg))
    
    if args.target in ['colab', 'both']:
        pkg = create_colab_package()
        packages.append(('Google Colab', pkg))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, path in packages:
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name}: {path} ({size:.1f} MB)")
    
    print("\n✅ All packages created in packages/ folder!")
    print("\nNext steps:")
    print("  1. Transfer packages to respective platforms")
    print("  2. Extract and start training")
    print("  3. See docs/PARALLEL_TRAINING.md for detailed instructions")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
