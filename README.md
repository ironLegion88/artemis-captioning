# ArtEmis Image Caption Generation

This project implements image captioning for artwork using the ArtEmis dataset, generating emotional captions for art images. Two architectures are implemented:

1. **CNN + LSTM with Attention** (ResNet18 encoder + LSTM decoder with Bahdanau attention)
2. **Vision Transformer (ViT)** (Custom ViT encoder + Transformer decoder)

## Results Summary

| Model | Best BLEU-4 | Val Loss | Notes |
|-------|-------------|----------|-------|
| CNN+LSTM (30 epochs) | **0.0197** | 3.845 | Best overall |
| CNN+LSTM (15 epochs) | 0.0155 | 3.925 | Fast training |
| ViT (12 epochs) | 0.0131 | 4.328 | With regularization |

**Best Model**: `checkpoints/colab_cnn_high_lr/best_model.pth`

## Quick Start - Generate Captions

```bash
# Activate environment
.venv\Scripts\activate

# Generate captions for an image
python scripts/predict.py --model checkpoints/primary_cnn_high_lr/best_model.pth \
                          --image path/to/your_artwork.jpg \
                          --emotion awe \
                          --top-k 3

# Available emotions: amusement, awe, contentment, excitement, anger, disgust, fear, sadness, nostalgia, "something else"

# List all available trained models
python scripts/predict.py --list-models
```

## Project Structure

```
artemis-captioning/
├── data/
│   ├── raw/                  # Raw datasets (ArtEmis CSV, WikiArt images)
│   ├── processed/            # Preprocessed data (splits, vocabulary)
│   │   └── images/           # Resized images 128x128)
│   └── embeddings/           # TF-IDF embeddings
├── models/                   # Model architectures
│   ├── cnn_lstm.py          # CNN+LSTM with attention
│   └── vision_transformer.py # ViT implementation
├── utils/                    # Utility functions
├── notebooks/                # Colab training notebooks
├── scripts/                  # Training and evaluation scripts
├── checkpoints/              # Trained model checkpoints
├── outputs/                  # Training logs and results
└── docs/                     # Documentation
    └── RESULTS_SUMMARY.md    # Detailed results analysis
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (Linux/Mac)
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

```bash
# Download and preprocess images
python scripts/prepare_dataset.py
python scripts/preprocess_images.py --num-images 15000
```

## Training

### Local Training (3000 images)

```bash
# Train CNN+LSTM (high learning rate - recommended)
python scripts/train_variations.py --config 2

# Train ViT (deep regularized)
python scripts/train_variations.py --config 7
```

### Colab Training (full dataset)

Use the notebooks in `notebooks/`:
- `Colab_Train_CNN_GloVe.ipynb` - Best CNN+LSTM configuration
- `Colab_Train_ViT.ipynb` - ViT training

## Inference

```bash
# Generate top-3 captions with beam search
python scripts/predict.py --model checkpoints/primary_cnn_high_lr/best_model.pth \
                          --image data/processed/images/Impressionism/claude-monet_water-lilies-1917-4.jpg \
                          --emotion awe \
                          --top-k 3 \
                          --beam-size 5
```

## Dataset Preprocessing

1. **Image Preprocessing:**
   - Resize images to 224×224
   - Normalize using ImageNet statistics
   - Handle corrupt/missing images

2. **Text Preprocessing:**
   - Lowercase and clean captions
   - Tokenize using NLTK
   - Build vocabulary (10,000 most frequent words)
   - Special tokens: `<PAD>`, `<START>`, `<END>`, `<UNK>`
   - Max caption length: 30 tokens

3. **Dataset Splits:**
   - Training: 80%
   - Validation: 10%
   - Test: 10%

## Model Architectures

### CNN + LSTM with Attention
- **Encoder:** ResNet18 (pretrained, fine-tuned)
- **Decoder:** 2-layer LSTM with Bahdanau attention
- **Feature Dim:** 256, Hidden Dim: 512, Attention Dim: 256
- **Embeddings:** TF-IDF based

### Vision Transformer
- **Encoder:** Custom ViT (16×16 patches)
- **Decoder:** Transformer decoder with cross-attention
- **Layers:** 4-6 encoder, 4 decoder
- **Heads:** 8 attention heads

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- torchvision
- NLTK
- scikit-learn
- PIL/Pillow
- numpy, pandas
- See `requirements.txt` for complete list

## References

1. Achlioptas et al., "ArtEmis: Affective Language for Visual Art", CVPR 2021
2. Vinyals et al., "Show and Tell: A Neural Image Caption Generator", CVPR 2015
3. Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021

## Documentation

See `docs/RESULTS_SUMMARY.md` for detailed results analysis and training insights.
