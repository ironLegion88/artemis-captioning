"""Quick test of training pipeline with minimal data."""
import sys
import os
sys.path.insert(0, '.')

import torch
import numpy as np

# Override constants for quick test
os.environ['QUICK_TEST'] = '1'

from utils.constants import (
    PROCESSED_DIR, EMBEDDINGS_DIR, VOCAB_SIZE, EMBEDDING_DIM,
    HIDDEN_DIM, IMAGE_FEATURE_DIM, NUM_LSTM_LAYERS, DROPOUT, DEVICE
)
from utils.data_loader import create_data_loaders
from utils.text_preprocessing import TextPreprocessor
from models.cnn_lstm import create_model
from train import Trainer

print("=" * 70)
print("QUICK TRAINING PIPELINE TEST")
print("=" * 70)

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Load text preprocessor first (needed for data loader)
print("\n✓ Loading text preprocessor...")
text_preprocessor = TextPreprocessor()
vocab_path = os.path.join(PROCESSED_DIR, 'vocabulary.json')
text_preprocessor.load_vocabulary(vocab_path)
print(f"  - Vocabulary size: {text_preprocessor.vocab_size}")

# Create small data loaders
print("\n✓ Creating data loaders (batch_size=4)...")
data_loaders = create_data_loaders(
    text_preprocessor=text_preprocessor,
    batch_size=4, 
    num_workers=0,
    splits=['train', 'val']
)
train_loader = data_loaders['train']
val_loader = data_loaders['val']
print(f"  - Train batches: {len(train_loader)}")
print(f"  - Val batches: {len(val_loader)}")

# Load embeddings
print("\n✓ Loading TF-IDF embeddings...")
tfidf_path = os.path.join(EMBEDDINGS_DIR, 'tfidf_embeddings.npy')
tfidf_matrix = np.load(tfidf_path)
embedding_tensor = torch.FloatTensor(tfidf_matrix)
print(f"  - Embedding shape: {embedding_tensor.shape}")

# Create model (no pretrained CNN for speed)
print("\n✓ Creating model (no pretrained CNN for speed)...")
model = create_model(
    embedding_matrix=embedding_tensor,
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBEDDING_DIM,
    encoder_dim=IMAGE_FEATURE_DIM,
    decoder_dim=HIDDEN_DIM,
    num_layers=NUM_LSTM_LAYERS,
    dropout=DROPOUT,
    pretrained_cnn=False
)

# Test single batch forward/backward
print("\n✓ Testing single batch training...")
batch = next(iter(train_loader))
images = batch['images'].to(DEVICE)
captions = batch['captions'].to(DEVICE)
lengths = batch['lengths'].to(DEVICE)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

# Forward
predictions, alphas, sorted_captions, decode_lengths, sort_ind = model(
    images, captions, lengths
)
print(f"  - Predictions shape: {predictions.shape}")

# Compute loss (simplified)
batch_size = predictions.size(0)
loss = 0
for i in range(batch_size):
    length = decode_lengths[i]
    pred = predictions[i, :length]
    target = sorted_captions[i, 1:length+1]
    loss += criterion(pred, target)
loss = loss / batch_size
print(f"  - Loss: {loss.item():.4f}")

# Backward
loss.backward()
optimizer.step()
print("  - Backward pass: SUCCESS")

# Test validation
print("\n✓ Testing validation...")
model.eval()
with torch.no_grad():
    val_batch = next(iter(val_loader))
    val_images = val_batch['images'].to(DEVICE)
    
    # Generate caption
    encoder_out = model.encoder(val_images[:1])
    gen_caption, gen_alphas = model.decoder.predict(encoder_out, max_length=20)
    
    # Decode
    gen_words = text_preprocessor.decode(gen_caption.cpu().numpy())
    print(f"  - Generated caption: '{gen_words}'")
    print(f"  - Attention shape: {gen_alphas.shape}")

print("\n" + "=" * 70)
print("✅ TRAINING PIPELINE TEST PASSED")
print("=" * 70)
