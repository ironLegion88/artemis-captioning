"""
Dataset Analysis and Subset Selection Script

This script:
1. Loads the ArtEmis dataset CSV
2. Analyzes dataset statistics
3. Performs stratified sampling to select 5,000 images
4. Validates that selected images exist and are readable
5. Saves subset metadata to JSON

Author: ArtEmis Caption Generation Project
Date: December 3, 2025
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import json
from collections import Counter
from PIL import Image
from tqdm import tqdm

# Import constants
from utils.constants import (
    ARTEMIS_CSV, WIKIART_DIR, PROCESSED_DATA_DIR,
    NUM_IMAGES_SUBSET, MIN_CAPTIONS_PER_IMAGE, RANDOM_SEED
)


def load_artemis_dataset():
    """Load the ArtEmis dataset CSV"""
    print("\n" + "=" * 70)
    print("LOADING ARTEMIS DATASET")
    print("=" * 70)
    
    print(f"\nReading CSV from: {ARTEMIS_CSV}")
    
    # Read CSV (it's large, so this might take a moment)
    df = pd.read_csv(ARTEMIS_CSV)
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Total annotations: {len(df):,}")
    print(f"  - Columns: {list(df.columns)}")
    
    return df


def analyze_dataset(df):
    """Compute and display dataset statistics"""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    stats = {
        'total_annotations': len(df),
        'unique_paintings': df['painting'].nunique(),
        'unique_art_styles': df['art_style'].nunique(),
        'unique_emotions': df['emotion'].nunique(),
        'avg_annotations_per_painting': len(df) / df['painting'].nunique()
    }
    
    print(f"\nOverall Statistics:")
    print(f"  - Total annotations: {stats['total_annotations']:,}")
    print(f"  - Unique paintings: {stats['unique_paintings']:,}")
    print(f"  - Unique art styles: {stats['unique_art_styles']}")
    print(f"  - Unique emotions: {stats['unique_emotions']}")
    print(f"  - Avg annotations/painting: {stats['avg_annotations_per_painting']:.2f}")
    
    # Art style distribution
    print(f"\nArt Style Distribution:")
    style_counts = df['art_style'].value_counts()
    for style, count in style_counts.head(10).items():
        print(f"  - {style:30s}: {count:6,} annotations")
    
    # Emotion distribution
    print(f"\nEmotion Distribution:")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  - {emotion:20s}: {count:6,} ({count/len(df)*100:.1f}%)")
    
    # Paintings per style (for stratification)
    print(f"\nPaintings per Art Style:")
    paintings_per_style = df.groupby('art_style')['painting'].nunique().sort_values(ascending=False)
    for style, count in paintings_per_style.head(10).items():
        print(f"  - {style:30s}: {count:5,} paintings")
    
    return stats, paintings_per_style


def select_subset(df, num_images=NUM_IMAGES_SUBSET, min_captions=MIN_CAPTIONS_PER_IMAGE):
    """
    Perform stratified sampling to select a subset of images
    
    Args:
        df: ArtEmis dataframe
        num_images: Target number of images to select
        min_captions: Minimum captions required per image
        
    Returns:
        List of selected painting names
    """
    print("\n" + "=" * 70)
    print(f"SELECTING {num_images:,} IMAGE SUBSET (STRATIFIED SAMPLING)")
    print("=" * 70)
    
    np.random.seed(RANDOM_SEED)
    
    # Group by painting and count captions
    painting_caption_counts = df.groupby('painting').size()
    
    # Filter paintings with minimum required captions
    valid_paintings = painting_caption_counts[painting_caption_counts >= min_captions].index.tolist()
    
    print(f"\nFiltering criteria:")
    print(f"  - Minimum captions per image: {min_captions}")
    print(f"  - Paintings meeting criteria: {len(valid_paintings):,}")
    
    # Get art style for each painting
    painting_styles = df.groupby('painting')['art_style'].first()
    
    # Count paintings per style (only valid paintings)
    paintings_per_style = df[df['painting'].isin(valid_paintings)].groupby('art_style')['painting'].nunique()
    total_valid = paintings_per_style.sum()
    
    print(f"\nStratified sampling:")
    print(f"  - Total valid paintings: {total_valid:,}")
    print(f"  - Target subset size: {num_images:,}")
    
    # Select paintings proportionally from each style
    selected_paintings = []
    
    for style in paintings_per_style.index:
        # Calculate proportional number for this style
        proportion = paintings_per_style[style] / total_valid
        n_to_select = int(proportion * num_images)
        
        # Get all valid paintings from this style
        style_paintings = df[(df['art_style'] == style) & (df['painting'].isin(valid_paintings))]['painting'].unique()
        
        # Sample
        if n_to_select > 0 and len(style_paintings) > 0:
            n_actual = min(n_to_select, len(style_paintings))
            selected = np.random.choice(style_paintings, size=n_actual, replace=False)
            selected_paintings.extend(selected.tolist())
            
            print(f"  - {style:30s}: {n_actual:4,} / {len(style_paintings):5,} paintings")
    
    # If we're short, randomly select more from remaining valid paintings
    if len(selected_paintings) < num_images:
        remaining = [p for p in valid_paintings if p not in selected_paintings]
        additional_needed = num_images - len(selected_paintings)
        if len(remaining) >= additional_needed:
            additional = np.random.choice(remaining, size=additional_needed, replace=False)
            selected_paintings.extend(additional.tolist())
            print(f"\n  Added {additional_needed:,} more paintings to reach target")
    
    print(f"\n✓ Total paintings selected: {len(selected_paintings):,}")
    
    return selected_paintings


def validate_images(selected_paintings, df):
    """
    Validate that selected images exist and are readable
    
    Args:
        selected_paintings: List of painting names
        df: ArtEmis dataframe
        
    Returns:
        List of validated painting info dictionaries
    """
    print("\n" + "=" * 70)
    print("VALIDATING IMAGES")
    print("=" * 70)
    
    validated_paintings = []
    corrupted_count = 0
    missing_count = 0
    
    print(f"\nChecking {len(selected_paintings):,} images...")
    
    for painting in tqdm(selected_paintings, desc="Validating images"):
        # Get art style for this painting
        style = df[df['painting'] == painting]['art_style'].iloc[0]
        
        # Construct image path
        image_path = WIKIART_DIR / style / f"{painting}.jpg"
        
        try:
            # Try to open and verify image
            with Image.open(image_path) as img:
                img.verify()
            
            # If successful, add to validated list
            validated_paintings.append({
                'painting': painting,
                'style': style,
                'path': str(image_path.relative_to(PROJECT_ROOT))
            })
            
        except FileNotFoundError:
            missing_count += 1
            if missing_count <= 5:  # Show first 5 missing files
                print(f"  ⚠ Missing: {image_path}")
                
        except Exception as e:
            corrupted_count += 1
            if corrupted_count <= 5:  # Show first 5 corrupted files
                print(f"  ⚠ Corrupted: {painting} - {e}")
    
    print(f"\n✓ Validation complete:")
    print(f"  - Valid images: {len(validated_paintings):,}")
    print(f"  - Missing images: {missing_count}")
    print(f"  - Corrupted images: {corrupted_count}")
    
    return validated_paintings


def save_results(validated_paintings, stats, df):
    """
    Save selected subset and statistics to JSON
    
    Args:
        validated_paintings: List of validated painting info
        stats: Dataset statistics dictionary
        df: ArtEmis dataframe
    """
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Create processed data directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare output data
    output = {
        'metadata': {
            'created_date': '2025-12-03',
            'total_available_images': stats['unique_paintings'],
            'total_annotations': stats['total_annotations'],
            'selection_criteria': {
                'target_size': NUM_IMAGES_SUBSET,
                'min_captions_per_image': MIN_CAPTIONS_PER_IMAGE,
                'sampling_method': 'stratified_by_art_style',
                'random_seed': RANDOM_SEED
            }
        },
        'statistics': stats,
        'selected_paintings': validated_paintings,
        'total_selected': len(validated_paintings)
    }
    
    # Add style distribution of selected subset
    style_dist = {}
    for p in validated_paintings:
        style = p['style']
        style_dist[style] = style_dist.get(style, 0) + 1
    output['subset_style_distribution'] = style_dist
    
    # Add caption count distribution
    caption_counts = []
    for p in validated_paintings:
        n_captions = len(df[df['painting'] == p['painting']])
        caption_counts.append(n_captions)
    
    output['caption_statistics'] = {
        'min_captions': min(caption_counts),
        'max_captions': max(caption_counts),
        'mean_captions': sum(caption_counts) / len(caption_counts),
        'total_captions_in_subset': sum(caption_counts)
    }
    
    # Save to JSON
    output_path = PROCESSED_DATA_DIR / 'selected_images.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"\nSelected Subset Summary:")
    print(f"  - Total images: {len(validated_paintings):,}")
    print(f"  - Total captions: {output['caption_statistics']['total_captions_in_subset']:,}")
    print(f"  - Captions per image: {output['caption_statistics']['mean_captions']:.1f} (avg)")
    print(f"  - Art styles: {len(style_dist)}")
    
    print(f"\nStyle distribution in subset:")
    for style, count in sorted(style_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {style:30s}: {count:4,} images ({count/len(validated_paintings)*100:.1f}%)")
    
    return output_path


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("ARTEMIS DATASET ANALYSIS AND SUBSET SELECTION")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Target Subset Size: {NUM_IMAGES_SUBSET:,} images")
    print(f"Minimum Captions per Image: {MIN_CAPTIONS_PER_IMAGE}")
    
    # Step 1: Load dataset
    df = load_artemis_dataset()
    
    # Step 2: Analyze dataset
    stats, paintings_per_style = analyze_dataset(df)
    
    # Step 3: Select subset (stratified sampling)
    selected_paintings = select_subset(df)
    
    # Step 4: Validate images
    validated_paintings = validate_images(selected_paintings, df)
    
    # Step 5: Save results
    output_path = save_results(validated_paintings, stats, df)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nGenerated file:")
    print(f"  - {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Review the selected_images.json file")
    print(f"  2. Run scripts/create_splits.py to create train/val/test splits")
    print(f"  3. Build vocabulary with utils/text_preprocessing.py")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
