"""
Create Colab-compatible split files that match actual filesystem encoding.

This script scans the actual preprocessed images directory and creates new
JSON split files where painting names exactly match the filenames on disk.
This fixes UTF-8 encoding mismatches between Windows and Linux (Colab).

The original splits are preserved, and new ones are created in:
data/processed/splits_colab/

Usage:
    python scripts/create_colab_splits.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import unicodedata


def normalize_name(name: str) -> str:
    """Normalize a name for comparison (lowercase, remove extension, normalize unicode)."""
    # Remove .jpg extension if present
    if name.endswith('.jpg'):
        name = name[:-4]
    # Normalize unicode and lowercase
    return unicodedata.normalize('NFC', name.lower())


def create_filename_mapping(images_dir: Path) -> dict:
    """
    Create a mapping from normalized names to actual filesystem names.
    
    Returns:
        Dict mapping normalized_name -> (actual_filename, style)
    """
    mapping = {}
    
    for style_dir in images_dir.iterdir():
        if not style_dir.is_dir():
            continue
        
        style = style_dir.name
        for img_file in style_dir.glob('*.jpg'):
            # Get the actual filename as it exists on disk
            actual_name = img_file.stem  # filename without extension
            normalized = normalize_name(actual_name)
            
            # Store both the actual name and its style
            mapping[normalized] = {
                'actual_name': actual_name,
                'style': style
            }
    
    return mapping


def fix_split_file(original_path: Path, output_path: Path, filename_mapping: dict) -> dict:
    """
    Create a fixed split file with filesystem-matched names.
    
    Returns:
        Statistics about the conversion
    """
    with open(original_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        'total_paintings': 0,
        'matched': 0,
        'unmatched': 0,
        'unmatched_names': []
    }
    
    fixed_paintings = []
    
    for painting_data in data['paintings']:
        stats['total_paintings'] += 1
        original_name = painting_data['painting']
        original_style = painting_data['style']
        
        # Try to find matching file
        normalized = normalize_name(original_name)
        
        if normalized in filename_mapping:
            # Found a match - use the actual filesystem name
            match = filename_mapping[normalized]
            fixed_painting = painting_data.copy()
            fixed_painting['painting'] = match['actual_name']
            fixed_painting['style'] = match['style']  # Use actual style folder name too
            fixed_paintings.append(fixed_painting)
            stats['matched'] += 1
        else:
            # Try fuzzy matching - sometimes it's just minor differences
            found = False
            for norm_key, match in filename_mapping.items():
                # Check if the base part matches (ignore special chars)
                if norm_key.replace('-', '').replace('_', '') == normalized.replace('-', '').replace('_', ''):
                    fixed_painting = painting_data.copy()
                    fixed_painting['painting'] = match['actual_name']
                    fixed_painting['style'] = match['style']
                    fixed_paintings.append(fixed_painting)
                    stats['matched'] += 1
                    found = True
                    break
            
            if not found:
                # No match found - keep original but log it
                fixed_paintings.append(painting_data)
                stats['unmatched'] += 1
                if len(stats['unmatched_names']) < 20:  # Limit logged names
                    stats['unmatched_names'].append(f"{original_style}/{original_name}")
    
    # Create output data
    output_data = {
        'split': data['split'],
        'num_paintings': len(fixed_paintings),
        'num_captions': sum(len(p['captions']) for p in fixed_paintings),
        'paintings': fixed_paintings
    }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write with UTF-8 encoding
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return stats


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    images_dir = base_dir / 'data' / 'processed' / 'images'
    splits_dir = base_dir / 'data' / 'processed' / 'splits'
    output_dir = base_dir / 'data' / 'processed' / 'splits_colab'
    
    print("=" * 70)
    print("Creating Colab-compatible split files")
    print("=" * 70)
    
    # Check directories exist
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        return
    
    if not splits_dir.exists():
        print(f"ERROR: Splits directory not found: {splits_dir}")
        return
    
    # Step 1: Scan filesystem to create mapping
    print("\n1. Scanning preprocessed images directory...")
    filename_mapping = create_filename_mapping(images_dir)
    print(f"   Found {len(filename_mapping)} unique images on filesystem")
    
    # Show sample of what we found
    print("\n   Sample filesystem names:")
    for i, (norm, info) in enumerate(list(filename_mapping.items())[:5]):
        print(f"   - {info['style']}/{info['actual_name']}")
    
    # Step 2: Process each split file
    print("\n2. Processing split files...")
    
    for split_name in ['train', 'val', 'test']:
        original_path = splits_dir / f'{split_name}.json'
        output_path = output_dir / f'{split_name}.json'
        
        if not original_path.exists():
            print(f"   WARNING: {split_name}.json not found, skipping")
            continue
        
        print(f"\n   Processing {split_name}.json...")
        stats = fix_split_file(original_path, output_path, filename_mapping)
        
        print(f"   - Total paintings: {stats['total_paintings']}")
        print(f"   - Matched to filesystem: {stats['matched']}")
        print(f"   - Unmatched (kept original): {stats['unmatched']}")
        
        if stats['unmatched_names']:
            print(f"   - Sample unmatched names:")
            for name in stats['unmatched_names'][:5]:
                print(f"     * {name}")
    
    # Step 3: Summary
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nColab-compatible splits saved to: {output_dir}")
    print("\nTo use on Colab:")
    print("1. Upload the splits_colab folder to your Google Drive")
    print("2. In Colab, copy splits_colab/*.json to data/processed/splits/")
    print("   OR modify data_loader.py to use splits_colab instead")
    print("\nOriginal splits preserved in: {splits_dir}")


if __name__ == '__main__':
    main()
