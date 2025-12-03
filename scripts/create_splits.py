"""
Create Train/Validation/Test Splits for ArtEmis Dataset

This script splits the selected 5,000 images into train/validation/test sets
with stratified sampling by art style to ensure balanced representation.

Split Ratios:
- Training: 80% (4,000 images)
- Validation: 10% (500 images)  
- Test: 10% (500 images)

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import (
    PROCESSED_DIR,
    SPLITS_DIR,
    ARTEMIS_CSV,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED
)


def load_selected_images(json_path: Path) -> Dict:
    """
    Load the selected images JSON file.
    
    Args:
        json_path: Path to selected_images.json
    
    Returns:
        Dictionary containing selected images data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_artemis_captions(csv_path: Path, selected_paintings: set) -> pd.DataFrame:
    """
    Load ArtEmis captions for selected paintings.
    
    Args:
        csv_path: Path to artemis_dataset_release_v0.csv
        selected_paintings: Set of selected painting names
    
    Returns:
        DataFrame with captions for selected paintings
    """
    print(f"\nLoading captions from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter to selected paintings
    df_selected = df[df['painting'].isin(selected_paintings)].copy()
    
    print(f"  - Total captions in dataset: {len(df):,}")
    print(f"  - Captions for selected paintings: {len(df_selected):,}")
    
    return df_selected


def stratified_split_by_style(
    paintings_list: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split paintings into train/val/test sets with stratified sampling by art style.
    
    Ensures each split has proportional representation of all art styles.
    
    Args:
        paintings_list: List of painting dictionaries with 'painting' and 'style' keys
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_paintings, val_paintings, test_paintings) as lists of painting names
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Group paintings by art style
    style_to_paintings = defaultdict(list)
    for item in paintings_list:
        style_to_paintings[item['style']].append(item['painting'])
    
    print(f"\nPerforming stratified split by art style:")
    print(f"  - Train: {train_ratio:.1%}")
    print(f"  - Val: {val_ratio:.1%}")
    print(f"  - Test: {test_ratio:.1%}")
    print(f"  - Random seed: {random_seed}")
    
    train_paintings = []
    val_paintings = []
    test_paintings = []
    
    # Split each style proportionally
    for style, paintings in sorted(style_to_paintings.items()):
        # Shuffle paintings within style
        random.shuffle(paintings)
        
        n_total = len(paintings)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Remaining go to test (handles rounding)
        
        # Split
        style_train = paintings[:n_train]
        style_val = paintings[n_train:n_train + n_val]
        style_test = paintings[n_train + n_val:]
        
        train_paintings.extend(style_train)
        val_paintings.extend(style_val)
        test_paintings.extend(style_test)
        
        print(f"  - {style:30s}: {len(style_train):4d} / {len(style_val):3d} / {len(style_test):3d}")
    
    # Shuffle final splits
    random.shuffle(train_paintings)
    random.shuffle(val_paintings)
    random.shuffle(test_paintings)
    
    return train_paintings, val_paintings, test_paintings


def create_split_files(
    train_paintings: List[str],
    val_paintings: List[str],
    test_paintings: List[str],
    captions_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create train/val/test JSON files with painting names and their captions.
    
    Each split file contains:
    - List of paintings in the split
    - Captions for each painting
    - Statistics about the split
    
    Args:
        train_paintings: List of training painting names
        val_paintings: List of validation painting names
        test_paintings: List of test painting names
        captions_df: DataFrame with all captions
        output_dir: Directory to save split files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': train_paintings,
        'val': val_paintings,
        'test': test_paintings
    }
    
    print(f"\nCreating split files in: {output_dir}")
    
    for split_name, painting_list in splits.items():
        # Filter captions for this split
        split_df = captions_df[captions_df['painting'].isin(painting_list)].copy()
        
        # Group by painting
        split_data = {
            'split': split_name,
            'num_paintings': len(painting_list),
            'num_captions': len(split_df),
            'paintings': []
        }
        
        # Add each painting with its captions
        for painting_name in painting_list:
            painting_df = split_df[split_df['painting'] == painting_name]
            
            painting_data = {
                'painting': painting_name,
                'style': painting_df['art_style'].iloc[0],
                'num_captions': len(painting_df),
                'captions': painting_df['utterance'].tolist(),
                'emotions': painting_df['emotion'].tolist()
            }
            
            split_data['paintings'].append(painting_data)
        
        # Add statistics
        split_data['statistics'] = {
            'num_paintings': len(painting_list),
            'num_captions': len(split_df),
            'avg_captions_per_painting': len(split_df) / len(painting_list),
            'styles': split_df['art_style'].value_counts().to_dict(),
            'emotions': split_df['emotion'].value_counts().to_dict()
        }
        
        # Save to file
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ {split_name:5s}: {len(painting_list):4d} paintings, {len(split_df):5d} captions -> {output_file.name}")


def analyze_splits(splits_dir: Path) -> None:
    """
    Analyze and display statistics about the created splits.
    
    Args:
        splits_dir: Directory containing split JSON files
    """
    print("\n" + "=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)
    
    for split_name in ['train', 'val', 'test']:
        split_file = splits_dir / f"{split_name}.json"
        
        with open(split_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = data['statistics']
        
        print(f"\n{split_name.upper()} SET:")
        print(f"  - Paintings: {stats['num_paintings']:,}")
        print(f"  - Captions: {stats['num_captions']:,}")
        print(f"  - Avg captions/painting: {stats['avg_captions_per_painting']:.2f}")
        
        print(f"  - Top 5 styles:")
        sorted_styles = sorted(stats['styles'].items(), key=lambda x: x[1], reverse=True)
        for style, count in sorted_styles[:5]:
            pct = 100 * count / stats['num_paintings']
            print(f"    • {style:30s}: {count:3d} ({pct:5.1f}%)")
        
        print(f"  - Top 3 emotions:")
        sorted_emotions = sorted(stats['emotions'].items(), key=lambda x: x[1], reverse=True)
        for emotion, count in sorted_emotions[:3]:
            pct = 100 * count / stats['num_captions']
            print(f"    • {emotion:20s}: {count:5d} ({pct:5.1f}%)")


def main():
    """
    Main function to create train/val/test splits.
    """
    print("=" * 70)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 70)
    
    # Load selected images
    selected_images_path = PROCESSED_DIR / "selected_images.json"
    
    if not selected_images_path.exists():
        print(f"\n❌ Error: {selected_images_path} not found")
        print("   Please run scripts/analyze_dataset.py first")
        return
    
    print(f"\n✓ Loading selected images from: {selected_images_path}")
    selected_data = load_selected_images(selected_images_path)
    
    paintings_list = selected_data['selected_paintings']
    print(f"  - Total paintings: {len(paintings_list)}")
    print(f"  - Art styles: {len(set(p['style'] for p in paintings_list))}")
    
    # Get set of selected painting names
    selected_paintings = {item['painting'] for item in paintings_list}
    
    # Load captions
    captions_df = load_artemis_captions(ARTEMIS_CSV, selected_paintings)
    
    # Create stratified splits
    train_paintings, val_paintings, test_paintings = stratified_split_by_style(
        paintings_list,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED
    )
    
    print(f"\nSplit sizes:")
    print(f"  - Train: {len(train_paintings):,} paintings ({100*len(train_paintings)/len(paintings_list):.1f}%)")
    print(f"  - Val:   {len(val_paintings):,} paintings ({100*len(val_paintings)/len(paintings_list):.1f}%)")
    print(f"  - Test:  {len(test_paintings):,} paintings ({100*len(test_paintings)/len(paintings_list):.1f}%)")
    
    # Create split files
    create_split_files(
        train_paintings,
        val_paintings,
        test_paintings,
        captions_df,
        SPLITS_DIR
    )
    
    # Analyze splits
    analyze_splits(SPLITS_DIR)
    
    print("\n" + "=" * 70)
    print("✅ SPLITS CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {SPLITS_DIR / 'train.json'}")
    print(f"  - {SPLITS_DIR / 'val.json'}")
    print(f"  - {SPLITS_DIR / 'test.json'}")
    print("\nNext steps:")
    print("  1. Review the split files")
    print("  2. Create PyTorch DataLoader (utils/data_loader.py)")
    print("  3. Begin model training")
    print("=" * 70)


if __name__ == "__main__":
    main()
