"""Quick training test - run 1 epoch with small batch to verify pipeline."""
import sys
import os
sys.path.insert(0, '.')

import torch
import numpy as np

from utils.constants import (
    PROCESSED_DIR, EMBEDDINGS_DIR, VOCAB_SIZE, EMBEDDING_DIM,
    HIDDEN_DIM, IMAGE_FEATURE_DIM, NUM_LSTM_LAYERS, DROPOUT, DEVICE,
    CHECKPOINTS_DIR, OUTPUTS_DIR, RANDOM_SEED
)
from utils.data_loader import create_data_loaders
from utils.text_preprocessing import TextPreprocessor
from models.cnn_lstm import create_model
from train import Trainer

print("=" * 70)
print("FULL TRAINING PIPELINE TEST (1 EPOCH)")
print("=" * 70)

# Set seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load text preprocessor
print("\n✓ Loading text preprocessor...")
text_preprocessor = TextPreprocessor()
vocab_path = PROCESSED_DIR / 'vocabulary.json'
text_preprocessor.load_vocabulary(vocab_path)
print(f"  - Vocabulary size: {text_preprocessor.vocab_size}")

# Create data loaders with small batch
print("\n✓ Creating data loaders...")
data_loaders = create_data_loaders(
    text_preprocessor=text_preprocessor,
    batch_size=8,  # Small batch for faster test
    num_workers=0,
    splits=['train', 'val']
)
train_loader = data_loaders['train']
val_loader = data_loaders['val']

# Limit loaders for quick test (first 50 batches only)
class LimitedLoader:
    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = max_batches
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
    
    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                break
            yield batch
    
    def __len__(self):
        return min(self.max_batches, len(self.loader))

train_loader_limited = LimitedLoader(train_loader, 50)
val_loader_limited = LimitedLoader(val_loader, 20)

print(f"  - Train batches (limited): {len(train_loader_limited)}")
print(f"  - Val batches (limited): {len(val_loader_limited)}")

# Load embeddings
print("\n✓ Loading TF-IDF embeddings...")
tfidf_path = EMBEDDINGS_DIR / 'tfidf_embeddings.npy'
tfidf_matrix = np.load(tfidf_path)
embedding_tensor = torch.FloatTensor(tfidf_matrix)
print(f"  - Embedding shape: {embedding_tensor.shape}")

# Create model (no pretrained for speed)
print("\n✓ Creating model...")
model = create_model(
    embedding_matrix=embedding_tensor,
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBEDDING_DIM,
    encoder_dim=IMAGE_FEATURE_DIM,
    decoder_dim=HIDDEN_DIM,
    num_layers=NUM_LSTM_LAYERS,
    dropout=DROPOUT,
    pretrained_cnn=False  # Faster for testing
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  - Total parameters: {total_params:,}")
print(f"  - Trainable parameters: {trainable_params:,}")

# Create trainer
print("\n✓ Creating trainer...")
test_checkpoint_dir = str(CHECKPOINTS_DIR / 'test')
test_output_dir = str(OUTPUTS_DIR / 'test')

trainer = Trainer(
    model=model,
    train_loader=train_loader_limited,
    val_loader=val_loader_limited,
    text_preprocessor=text_preprocessor,
    learning_rate=1e-4,
    device=DEVICE,
    checkpoint_dir=test_checkpoint_dir,
    output_dir=test_output_dir
)

# Run 1 epoch
print("\n" + "=" * 70)
print("RUNNING 1 EPOCH")
print("=" * 70)

history = trainer.train(num_epochs=1)

print("\n" + "=" * 70)
print("✅ FULL TRAINING PIPELINE TEST PASSED")
print("=" * 70)
print(f"\nResults:")
print(f"  - Final train loss: {history['train_loss'][-1]:.4f}")
print(f"  - Final val loss: {history['val_loss'][-1]:.4f}")
print(f"  - Final BLEU: {history['val_bleu'][-1]:.4f}")
