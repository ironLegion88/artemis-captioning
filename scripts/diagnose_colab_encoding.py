# Run this in Colab to diagnose the filename encoding issue
# Paste this into a cell and run it

import os
import json
from pathlib import Path

# Check what's in the JSON
splits_dir = Path('/content/artemis/data/processed/splits')
images_dir = Path('/content/artemis/data/processed/images')

print("=" * 70)
print("DIAGNOSING FILENAME ENCODING ISSUE")
print("=" * 70)

# 1. Sample filenames from JSON
print("\n1. SAMPLE NAMES FROM JSON (train.json):")
with open(splits_dir / 'train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find some with special characters
special_char_paintings = []
for p in data['paintings']:
    name = p['painting']
    if any(ord(c) > 127 for c in name):
        special_char_paintings.append((p['style'], name))
        if len(special_char_paintings) >= 5:
            break

for style, name in special_char_paintings:
    print(f"  JSON: {style}/{name}.jpg")
    # Check if file exists
    path = images_dir / style / f"{name}.jpg"
    print(f"  Exists: {path.exists()}")

# 2. Sample actual filenames from disk
print("\n2. SAMPLE ACTUAL FILENAMES ON DISK:")
for style_dir in sorted(images_dir.iterdir())[:3]:
    if style_dir.is_dir():
        files = list(style_dir.glob('*.jpg'))[:3]
        for f in files:
            name = f.name
            # Show hex of any special chars
            if any(ord(c) > 127 for c in name):
                print(f"  DISK: {style_dir.name}/{name}")
                print(f"        Bytes: {name.encode('utf-8')[:50]}")

# 3. Find mismatches for one problematic artist
print("\n3. CHECKING JOAQUÍN SOROLLA FILES:")
target_style = "Impressionism"
target_prefix = "joaqu"

# What's in JSON?
json_names = []
for p in data['paintings']:
    if p['style'] == target_style and target_prefix in p['painting'].lower():
        json_names.append(p['painting'])

print(f"  Found {len(json_names)} in JSON with '{target_prefix}'")
if json_names:
    print(f"  Example: {json_names[0]}")
    print(f"  Bytes: {json_names[0].encode('utf-8')}")

# What's on disk?
disk_names = []
style_path = images_dir / target_style
if style_path.exists():
    for f in style_path.glob('*.jpg'):
        if target_prefix in f.stem.lower():
            disk_names.append(f.stem)

print(f"  Found {len(disk_names)} on disk with '{target_prefix}'")
if disk_names:
    print(f"  Example: {disk_names[0]}")
    print(f"  Bytes: {disk_names[0].encode('utf-8')}")

# 4. Compare byte representations
print("\n4. BYTE COMPARISON:")
if json_names and disk_names:
    j = json_names[0]
    d = disk_names[0]
    print(f"  JSON name bytes: {j.encode('utf-8')}")
    print(f"  DISK name bytes: {d.encode('utf-8')}")
    print(f"  Are they equal? {j == d}")
