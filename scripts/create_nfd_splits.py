"""
Create Colab-compatible split files using NFD normalization.

The issue: Google Drive/Colab uses NFD (decomposed) Unicode normalization,
while Windows uses NFC (precomposed). This causes filename mismatches.

Example:
  - NFC: joaqu\xc3\xa3 (ã as single character)
  - NFD: joaqua\xcc\x83 (a + combining tilde)

This script creates new JSON files with NFD-normalized painting names.
Run this on Windows, then upload the resulting files to Colab.
"""

import json
import unicodedata
from pathlib import Path


def normalize_to_nfd(text: str) -> str:
    """Convert text to NFD (decomposed) Unicode normalization."""
    return unicodedata.normalize('NFD', text)


def create_nfd_splits():
    """Create NFD-normalized split files for Colab."""
    base_dir = Path(__file__).parent.parent
    splits_dir = base_dir / 'data' / 'processed' / 'splits'
    output_dir = base_dir / 'data' / 'processed' / 'splits_nfd'
    
    print("=" * 70)
    print("Creating NFD-normalized split files for Colab")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name in ['train', 'val', 'test']:
        input_path = splits_dir / f'{split_name}.json'
        output_path = output_dir / f'{split_name}.json'
        
        if not input_path.exists():
            print(f"  WARNING: {split_name}.json not found, skipping")
            continue
        
        print(f"\nProcessing {split_name}.json...")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert all painting names and styles to NFD
        converted = 0
        for painting in data['paintings']:
            old_name = painting['painting']
            old_style = painting['style']
            
            new_name = normalize_to_nfd(old_name)
            new_style = normalize_to_nfd(old_style)
            
            if new_name != old_name or new_style != old_style:
                converted += 1
            
            painting['painting'] = new_name
            painting['style'] = new_style
        
        # Write with UTF-8 encoding, ensure_ascii=False to preserve Unicode
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  - Paintings: {len(data['paintings'])}")
        print(f"  - Converted to NFD: {converted}")
        print(f"  - Saved to: {output_path}")
    
    # Show example of the conversion
    print("\n" + "=" * 70)
    print("EXAMPLE CONVERSION:")
    print("=" * 70)
    
    test_name = "joaquín-sorolla_test"
    nfc = unicodedata.normalize('NFC', test_name)
    nfd = unicodedata.normalize('NFD', test_name)
    
    print(f"  Original: {test_name}")
    print(f"  NFC bytes: {nfc.encode('utf-8')}")
    print(f"  NFD bytes: {nfd.encode('utf-8')}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nNFD-normalized splits saved to: {output_dir}")
    print("\nUpload the 'splits_nfd' folder contents to Colab and copy to:")
    print("  /content/artemis/data/processed/splits/")


if __name__ == '__main__':
    create_nfd_splits()
