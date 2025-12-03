# ArtEmis Image Caption Generation

This project implements image caption generation for artwork images using the ArtEmis dataset. Two architectures are developed and compared:
1. CNN + LSTM model (custom CNN encoder + LSTM decoder)
2. Vision-Language Transformer (ViT encoder + Transformer decoder)

## Project Structure

```
artemis-captioning/
├── data/
│   ├── raw/                  # Raw datasets (not in git)
│   ├── processed/            # Preprocessed data
│   └── embeddings/           # Pre-trained embeddings
├── models/                   # Model architectures
├── utils/                    # Utility functions
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Training and evaluation scripts
├── checkpoints/              # Model checkpoints (not in git)
├── outputs/                  # Generated outputs
└── requirements.txt          # Dependencies
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
uv venv artemis-env

# Activate environment (Windows)
artemis-env\Scripts\activate
```

### 2. Install Dependencies

```bash
# Using uv add (recommended)
uv add torch torchvision numpy pandas pillow matplotlib seaborn nltk scikit-learn gensim transformers jupyter tqdm tensorboard rouge-score

# Or using pip
pip install -r requirements.txt
```

### 3. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 4. Dataset Setup

- Place ArtEmis dataset in `data/raw/artemis/`
- Place WikiArt images in `data/raw/wikiart/`

## Execution Instructions

### Preprocessing

```bash
python scripts/preprocessing.py --artemis-path data/raw/artemis/artemis_dataset_release_v0.csv --wikiart-path data/raw/wikiart --output-path data/processed --num-samples 5000
```

### Training

```bash
# Test pipeline first
python scripts/test_pipeline.py --samples 100 --epochs 2

# Train CNN+LSTM
python scripts/train.py --model cnn_lstm --embedding word2vec --config configs/cnn_lstm_config.json

# Train Transformer
python scripts/train.py --model transformer --config configs/transformer_config.json
```

### Evaluation

```bash
python scripts/evaluation.py --model checkpoints/best_model.pt --test-data data/processed/test.json --output outputs/results.json
```

### Inference

```bash
python scripts/predict.py --image path/to/image.jpg --model checkpoints/best_model.pt --vocab data/processed/vocabulary.json --config configs/config.json --top-k 3
```

## Dataset Preprocessing Steps

1. **Image Preprocessing:**
   - Resize images to 128×128 or 224×224
   - Normalize pixel values using ImageNet statistics
   - Handle corrupt/missing images

2. **Text Preprocessing:**
   - Lowercase and remove punctuation
   - Tokenize captions
   - Build vocabulary (10,000 most frequent words)
   - Add special tokens: `<PAD>`, `<START>`, `<END>`, `<UNK>`
   - Pad/truncate sequences to uniform length (30 tokens)

3. **Dataset Splits:**
   - Training: 80% (4,000 images)
   - Validation: 10% (500 images)
   - Test: 10% (500 images)

## Model Architectures

### CNN + LSTM
- **Encoder:** Custom 4-block CNN (trained from scratch)
- **Decoder:** 2-layer LSTM with attention
- **Embeddings:** TF-IDF, Word2Vec, or GloVe

### Vision-Language Transformer
- **Encoder:** Vision Transformer (patch-based)
- **Decoder:** Transformer decoder with cross-attention
- **End-to-end training**

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- torchvision
- NLTK
- scikit-learn
- gensim
- See `requirements.txt` for complete list

## References

1. Achlioptas et al., "ArtEmis: Affective Language for Visual Art", CVPR 2021
2. Vinyals et al., "Show and Tell: A Neural Image Caption Generator", CVPR 2015
3. Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
