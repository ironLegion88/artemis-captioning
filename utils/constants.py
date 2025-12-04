"""
Configuration constants for the ArtEmis Caption Generation project
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Raw data paths
ARTEMIS_DIR = RAW_DATA_DIR / "artemis"
WIKIART_DIR = RAW_DATA_DIR / "wikiart"
ARTEMIS_CSV = ARTEMIS_DIR / "artemis_dataset_release_v0.csv"

# Processed data paths
PROCESSED_DIR = PROCESSED_DATA_DIR  # Alias for convenience
PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"
PROCESSED_CAPTIONS_DIR = PROCESSED_DATA_DIR / "captions"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"
VOCABULARY_PATH = PROCESSED_DATA_DIR / "vocabulary.json"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data preprocessing constants
IMAGE_SIZE = (128, 128)  # Can be changed to (224, 224)
MAX_CAPTION_LENGTH = 30
VOCAB_SIZE = 10000
MIN_WORD_FREQ = 2

# ImageNet normalization (standard for CNNs)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Special tokens
PAD_TOKEN = '<PAD>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'
UNK_TOKEN = '<UNK>'

SPECIAL_TOKENS = {
    PAD_TOKEN: 0,
    START_TOKEN: 1,
    END_TOKEN: 2,
    UNK_TOKEN: 3
}

# Dataset split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Dataset subset size (for memory constraints)
NUM_IMAGES_SUBSET = 15000  # Select 5,000 images from 81,444 total
MIN_CAPTIONS_PER_IMAGE = 3  # Ensure each image has at least 3 captions

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10  # Early stopping patience
MIN_DELTA = 0.001  # Minimum improvement for early stopping

# Model hyperparameters
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
IMAGE_FEATURE_DIM = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.5

# Transformer hyperparameters
PATCH_SIZE = 16
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NUM_ATTENTION_HEADS = 8
MLP_RATIO = 4

# Pre-trained embeddings
WORD2VEC_MODEL = 'word2vec-google-news-300'
GLOVE_MODEL = 'glove-wiki-gigaword-300'
FASTTEXT_MODEL = 'fasttext-wiki-news-subwords-300'

# Evaluation
BLEU_WEIGHTS = {
    'BLEU-1': (1.0, 0, 0, 0),
    'BLEU-2': (0.5, 0.5, 0, 0),
    'BLEU-3': (0.33, 0.33, 0.33, 0),
    'BLEU-4': (0.25, 0.25, 0.25, 0.25)
}

# Emotions in ArtEmis dataset
ARTEMIS_EMOTIONS = [
    'amusement', 'awe', 'contentment', 'excitement',
    'anger', 'disgust', 'fear', 'sadness', 'something else'
]

# Random seed for reproducibility
RANDOM_SEED = 42

# Device configuration
DEVICE = 'cpu'  # No GPU available (i5-1235U)
USE_AMP = True  # Mixed precision training (FP16) - critical for memory efficiency
NUM_WORKERS = 0  # DataLoader workers (0 for Windows compatibility)

# Logging
LOG_INTERVAL = 50  # Log every N batches
SAVE_INTERVAL = 5  # Save checkpoint every N epochs
