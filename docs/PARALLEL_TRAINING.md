# Parallel Training Setup Guide

This guide helps you set up training on multiple machines for parallel hyperparameter experimentation.

---

## 🖥️ SECOND LAPTOP SETUP (Intel Ultra 5 125H)

### Step 1: Clone the Repository
```bash
git clone https://github.com/ironLegion88/artemis-captioning.git
cd artemis-captioning
```

### Step 2: Install UV and Create Environment
```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment
uv venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Step 3: Copy Required Data Files
Copy these folders/files from the main laptop via USB, cloud, or network:

```
📁 data/
├── 📁 processed/
│   ├── 📁 images/           (~57 MB - 5000 pre-resized 128x128 images)
│   │   ├── 📁 Abstract_Expressionism/
│   │   ├── 📁 Baroque/
│   │   └── ... (27 style folders)
│   ├── 📁 splits/
│   │   ├── train.json       (4000 images)
│   │   ├── val.json         (500 images)
│   │   └── test.json        (500 images)
│   ├── 📁 captions/         (caption JSON files per image)
│   └── vocabulary.json      (10000 word vocabulary)
└── 📁 embeddings/           (optional, ~30 MB)
    ├── glove_embeddings.npy
    ├── word2vec_embeddings.npy
    └── tfidf_embeddings.npy
```

**Quick Copy Command (Windows):**
```cmd
:: On main laptop - create zip of required data
cd artemis-captioning
powershell Compress-Archive -Path "data\processed", "data\embeddings" -DestinationPath "artemis_data.zip"
:: Transfer artemis_data.zip to second laptop, then extract to artemis-captioning/data/
```

### Step 4: Verify Setup
```bash
# Activate environment
.venv\Scripts\activate

# Verify files
python -c "from pathlib import Path; imgs=list(Path('data/processed/images').rglob('*.jpg')); print(f'Images: {len(imgs)}')"
# Expected output: Images: 5000

# Verify vocabulary
python -c "import json; v=json.load(open('data/processed/vocabulary.json')); print(f'Vocab size: {v[\"vocab_size\"]}')"
# Expected output: Vocab size: 10000

# Test import
python -c "from utils.data_loader import create_dataloaders; print('✓ All imports OK')"
```

### Step 5: Start Training
```bash
# Train all 3 configurations sequentially (recommended)
python scripts/train_second_laptop.py --all

# OR train specific configuration
python scripts/train_second_laptop.py --config 1   # CNN+LSTM Standard (30 epochs)
python scripts/train_second_laptop.py --config 2   # ViT Compact (30 epochs)
python scripts/train_second_laptop.py --config 3   # CNN+LSTM High LR (25 epochs)
```

**Expected Training Time on Intel Ultra 5 125H:**
- Config 1: ~4-5 hours (30 epochs × 5000 images)
- Config 2: ~3-4 hours (30 epochs × 5000 images, ViT faster)
- Config 3: ~3-4 hours (25 epochs × 5000 images)
- **Total: ~10-13 hours for all 3 models**

---

## ☁️ GOOGLE COLAB SETUP (T4 GPU)

### Step 1: Upload Data to Google Drive

On your main laptop, create a zip with required data:
```bash
cd artemis-captioning

# Create data package for Colab (~100 MB total)
powershell Compress-Archive -Path "data\processed", "data\embeddings", "utils", "models", "train.py" -DestinationPath "colab_package.zip"
```

Upload `colab_package.zip` to Google Drive and extract it to:
```
Google Drive/
└── artemis-captioning/
    ├── data/
    │   ├── processed/
    │   │   ├── images/      (5000 preprocessed images)
    │   │   ├── splits/
    │   │   ├── captions/
    │   │   └── vocabulary.json
    │   └── embeddings/
    ├── utils/
    ├── models/
    └── train.py
```

### Step 2: Open Colab Notebook
1. Go to https://colab.research.google.com
2. Upload `notebooks/Colab_Multi_Model_Training.ipynb` or open from Drive
3. **Enable GPU**: Runtime → Change runtime type → T4 GPU

### Step 3: Run the Notebook
1. Run all setup cells (GPU check, mount Drive, copy data)
2. Select which model to train (1, 2, or 3)
3. Run training cell

**Colab Configurations:**
| Config | Model | Images | Epochs | Est. Time |
|--------|-------|--------|--------|-----------|
| colab_cnn_large | CNN+LSTM (512 embed, 1024 hidden) | 15000 | 50 | ~3-4 hours |
| colab_vit_standard | ViT (256 embed, 6 layers) | 15000 | 50 | ~2-3 hours |
| colab_cnn_glove | CNN+LSTM + GloVe | 15000 | 40 | ~2-3 hours |

### Step 4: Save Results
Results are automatically saved to Google Drive:
- `outputs/colab_{name}/training_history.json`
- `checkpoints/colab_{name}/best_model.pth`

---

## 📋 QUICK REFERENCE

### What to Copy Where

| Destination | Required Files | Size |
|-------------|---------------|------|
| Second Laptop | `data/processed/images/`, `data/processed/splits/`, `data/processed/captions/`, `data/processed/vocabulary.json`, `data/embeddings/` | ~90 MB |
| Google Colab | Same as above + `utils/`, `models/`, `train.py` | ~100 MB |

### Training Scripts

| Location | Script | Configs | Images | Epochs |
|----------|--------|---------|--------|--------|
| Main Laptop | `scripts/train_variations.py` | 5 configs | 3000 | 15-25 |
| Second Laptop | `scripts/train_second_laptop.py` | 3 configs | 5000 | 25-30 |
| Google Colab | `notebooks/Colab_Multi_Model_Training.ipynb` | 3 configs | 15000 | 40-50 |

### Verify Data is Complete
```bash
# Run this on any machine after copying data
python -c "
from pathlib import Path
import json

# Check images
imgs = list(Path('data/processed/images').rglob('*.jpg'))
print(f'✓ Images: {len(imgs)}')

# Check splits
for split in ['train', 'val', 'test']:
    data = json.load(open(f'data/processed/splits/{split}.json'))
    print(f'✓ {split}: {len(data)} samples')

# Check vocabulary
vocab = json.load(open('data/processed/vocabulary.json'))
print(f'✓ Vocabulary: {vocab[\"vocab_size\"]} words')

print('\\n✅ All data verified!')
"
```

---

## 📊 COLLECTING RESULTS

After training completes on all machines, copy back:

### From Second Laptop
```
checkpoints/laptop2_*/          → Main laptop checkpoints/
outputs/experiments/laptop2_*/  → Main laptop outputs/experiments/
```

### From Google Colab (Drive)
```
checkpoints/colab_*/            → Main laptop checkpoints/
outputs/colab_*/                → Main laptop outputs/
```

### Compare Results
```bash
python scripts/predict.py --list-models
```

---

## 🔧 TROUBLESHOOTING

### Out of Memory
- Reduce batch_size (try 8 or even 4)
- Reduce num_images in config

### Slow Training
- Ensure `data/processed/images/` exists (preprocessed 128x128)
- Larger batch_size if RAM allows

### Missing Files Error
- Re-run verification script above
- Ensure all data folders copied correctly

### Colab Disconnects
- Results auto-save to Drive every 5 epochs
- Resume from latest checkpoint if disconnected
