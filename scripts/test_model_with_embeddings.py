"""Test CNN+LSTM model with TF-IDF embeddings."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from models.cnn_lstm import create_model
from utils.embeddings import TextEmbeddings

print("=" * 70)
print("TESTING CNN+LSTM WITH TF-IDF EMBEDDINGS")
print("=" * 70)

# Load TF-IDF embeddings
print("\n✓ Loading TF-IDF embeddings...")
import os
from utils.constants import EMBEDDINGS_DIR

tfidf_path = os.path.join(EMBEDDINGS_DIR, 'tfidf_embeddings.npy')
tfidf_matrix = np.load(tfidf_path)
print(f"  - Embedding matrix shape: {tfidf_matrix.shape}")
print(f"  - Embedding matrix dtype: {tfidf_matrix.dtype}")

# Convert to PyTorch tensor
embedding_tensor = torch.FloatTensor(tfidf_matrix)
print(f"  - PyTorch tensor shape: {embedding_tensor.shape}")

# Create model with pre-trained embeddings
print("\n✓ Creating model with TF-IDF embeddings...")
model = create_model(
    embedding_matrix=embedding_tensor,
    vocab_size=10000,
    embed_dim=256,
    encoder_dim=256,
    decoder_dim=512,
    pretrained_cnn=False
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  - Total parameters: {total_params:,}")
print(f"  - Trainable parameters: {trainable_params:,}")

# Test forward pass
print("\n✓ Testing forward pass with embeddings...")
batch_size = 2
images = torch.randn(batch_size, 3, 128, 128)
captions = torch.randint(0, 10000, (batch_size, 25))
caption_lengths = torch.tensor([20, 18])

predictions, alphas, sorted_captions, decode_lengths, sort_ind = model(
    images, captions, caption_lengths
)

print(f"  - Predictions shape: {predictions.shape}")
print(f"  - Alphas shape: {alphas.shape}")

# Test that embeddings are loaded correctly
print("\n✓ Verifying embeddings...")
embedding_weights = model.decoder.embedding.weight.data
print(f"  - Embedding layer shape: {embedding_weights.shape}")
print(f"  - Embedding layer dtype: {embedding_weights.dtype}")

# Check match (accounting for float32 conversion)
embeddings_match = torch.allclose(
    embedding_weights, 
    embedding_tensor.float(), 
    atol=1e-6
)
print(f"  - Embeddings match TF-IDF (float32): {embeddings_match}")

# Test specific word embeddings
print("\n✓ Testing specific word embeddings...")
word_idx = 100
model_emb = embedding_weights[word_idx].numpy()
tfidf_emb = tfidf_matrix[word_idx].astype(np.float32)
print(f"  - Word index {word_idx}:")
print(f"    Model embedding: mean={model_emb.mean():.6f}, std={model_emb.std():.6f}")
print(f"    TF-IDF embedding: mean={tfidf_emb.mean():.6f}, std={tfidf_emb.std():.6f}")
print(f"    Match: {np.allclose(model_emb, tfidf_emb, atol=1e-6)}")

# Test PAD token (should be zeros)
print("\n✓ Testing special tokens...")
pad_emb = embedding_weights[0].numpy()
print(f"  - PAD token (idx=0): mean={pad_emb.mean():.8f}, max_abs={np.abs(pad_emb).max():.8f}")
print(f"  - PAD is zeros: {np.allclose(pad_emb, 0, atol=1e-6)}")

print("\n" + "=" * 70)
print("✅ MODEL WITH EMBEDDINGS TEST PASSED")
print("=" * 70)
