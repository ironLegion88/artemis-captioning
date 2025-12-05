# ArtEmis Caption Generation - Results Summary

## Project Overview

This project implements an image captioning system for artwork using the ArtEmis dataset. The system generates emotional captions for art images by combining visual features with emotion tokens.

## Dataset

- **Source**: ArtEmis Dataset (WikiArt images with emotional annotations)
- **Training Images**: 15,000 preprocessed images (224×224)
- **Vocabulary Size**: 10,000 tokens
- **Emotions**: 9 categories (amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something else)
- **Art Styles**: 27 different styles (Impressionism, Cubism, Romanticism, etc.)

## Model Architectures

### 1. CNN+LSTM with Attention
- **Encoder**: ResNet18 (pretrained on ImageNet, fine-tuned)
- **Decoder**: LSTM with Bahdanau attention mechanism
- **Feature Dimension**: 256
- **Hidden Dimension**: 512
- **Attention Dimension**: 256
- **Embeddings**: TF-IDF based (custom trained on ArtEmis captions)

### 2. Vision Transformer (ViT)
- **Encoder**: Custom ViT with patch embedding (16×16 patches)
- **Decoder**: Transformer decoder with cross-attention
- **Encoder Layers**: 4-6
- **Decoder Layers**: 4
- **Embedding Dimension**: 256-384
- **Attention Heads**: 8

## Training Configurations

### Experiments Summary

| Experiment | Model | Epochs | Batch Size | LR | Val Loss | BLEU-4 |
|------------|-------|--------|------------|-----|----------|--------|
| colab_cnn_high_lr | CNN+LSTM | 30 | 32 | 3e-4 | 3.845 | **0.0197** |
| primary_cnn_large_batch | CNN+LSTM | 15 | 32 | 3e-4 | 3.925 | 0.0155 |
| primary_cnn_high_lr | CNN+LSTM | 15 | 24 | 3e-4 | 3.946 | 0.0154 |
| primary_vit_deep_regularized | ViT | 12 | 16 | 1e-4 | 4.328 | 0.0131 |
| local_3k_vit | ViT | 30 | 16 | 3e-4 | 4.355 | 0.0127 |
| local_3k | CNN+LSTM | 30 | 16 | 1e-4 | 4.042 | 0.0125 |
| primary_vit_regularized | ViT | 10 | 16 | 1e-4 | 4.348 | 0.0108 |
| primary_vit_higher_lr | ViT | 10 | 16 | 1e-4 | 4.413 | 0.0081 |
| colab_vit_compact | ViT | 30 | 64 | 1e-4 | 4.400 | 0.0030 |

## Key Findings

### 1. Model Architecture Comparison
- **CNN+LSTM outperforms ViT** on this dataset with the given training budget
- Best CNN+LSTM BLEU: 0.0197 vs Best ViT BLEU: 0.0131
- CNN+LSTM benefits from pretrained ResNet18 features

### 2. Learning Rate Impact
- **Higher learning rates (3e-4) work better** for CNN+LSTM
- ViT requires lower LR (1e-4) with more regularization

### 3. Training Dynamics
- **CNN+LSTM**: Stable training, consistent improvement
- **ViT**: Peaks early (epoch 6-10), then may degrade without early stopping
- More epochs benefit CNN+LSTM; ViT needs early stopping

### 4. Regularization
- Dropout of 0.25-0.3 works well for both architectures
- ViT benefits from additional regularization (label smoothing, weight decay)

## Best Model Details

**Model**: `colab_cnn_high_lr`
- Architecture: CNN+LSTM with Attention
- Training: 30 epochs on 15k images (Google Colab A100)
- Best Validation Loss: 3.845
- Best BLEU-4 Score: 0.0197

### Training Curves

```
Epoch  Train Loss  Val Loss   BLEU-4
─────  ──────────  ────────   ──────
1      5.616       4.941      0.0065
5      4.360       4.253      0.0069
10     4.042       4.041      0.0104
15     3.856       3.946      0.0105
20     3.735       3.883      0.0067
25     3.621       3.862      0.0089
28     3.557       3.847      0.0197 ← Best
30     3.526       3.858      0.0156
```

## Sample Predictions

### Example 1: Monet Water Lilies (Impressionism)
```
Image: claude-monet_water-lilies-1917-4.jpg
Emotion: awe

Top-3 Captions:
1. "[awe] it looks like a beautiful painting" (Score: -12.80)
2. "[awe] it looks like a lot of colors" (Score: -14.07)
3. "[awe] it looks like a lot of flowers and water" (Score: -17.64)
```

### Example 2: Picasso Bullfight (Cubism)
```
Image: pablo-picasso_a-bullfight-1934.jpg
Emotion: excitement

Top-3 Captions:
1. "[excitement] it looks like a chaotic scene" (Score: -12.36)
2. "[excitement] this painting looks like a battle" (Score: -14.11)
3. "[excitement] this painting looks like a scene of action" (Score: -16.40)
```

## Usage Instructions

### Generate Captions for New Images

```bash
# Basic usage
python scripts/predict.py --model checkpoints/primary_cnn_high_lr/best_model.pth \
                          --image path/to/image.jpg \
                          --emotion awe \
                          --top-k 3

# Available emotions
# amusement, awe, contentment, excitement, anger, disgust, fear, sadness, nostalgia, "something else"

# List available models
python scripts/predict.py --list-models
```

### Model Files

```
checkpoints/
├── colab_cnn_glove/        # Best overall model (30 epochs)
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── primary_cnn_high_lr/    # Best 15-epoch model
│   ├── best_model.pth
│   └── latest_checkpoint.pth
└── primary_vit_*/          # ViT experiments
```

## Limitations & Future Work

### Current Limitations
1. Uses pretrained ResNet18 encoder (transfer learning)
2. Limited vocabulary leads to <UNK> tokens for rare words
3. BLEU scores are relatively low (typical for art captioning)
4. Caption diversity is limited

### Future Improvements
1. Train CNN from scratch for stricter compliance
2. Use larger vocabulary with subword tokenization (BPE)
3. Implement contrastive learning for better emotion embeddings
4. Add attention visualization for interpretability
5. Increase training data size and epochs

## Reproducibility

### Environment
- Python 3.10+
- PyTorch 2.0+
- CUDA (for GPU training)

### Training
```bash
# Local training (3000 images)
python scripts/train_variations.py --config 2

# Colab training (full dataset)
# See notebooks/Colab_Train_CNN_GloVe.ipynb
```

### Data Preparation
```bash
python scripts/prepare_dataset.py
python scripts/preprocess_images.py
```

## Files Structure

```
artemis-captioning/
├── models/                 # Model architectures
│   ├── cnn_lstm.py        # CNN+LSTM with attention
│   └── vision_transformer.py  # ViT implementation
├── scripts/
│   ├── predict.py         # Inference script
│   ├── train_variations.py # Training configurations
│   └── analyze_dataset.py # EDA
├── checkpoints/           # Saved models
├── outputs/               # Training logs and results
├── data/
│   ├── processed/         # Preprocessed data
│   │   ├── images/        # Resized images (224×224)
│   │   └── vocabulary.json
│   └── embeddings/        # TF-IDF embeddings
└── notebooks/             # Jupyter notebooks for Colab
```

---

**Author**: ArtEmis Caption Generation Project  
**Date**: December 2025  
**Assignment**: Introduction to Machine Learning, Monsoon 2025
