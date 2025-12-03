"""Test script for CNN+LSTM model."""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from models.cnn_lstm import create_model

print("=" * 70)
print("TESTING CNN+LSTM MODEL")
print("=" * 70)

# Create model
print("\n✓ Creating model...")
model = create_model(
    vocab_size=10000,
    embed_dim=256,
    encoder_dim=256,
    decoder_dim=512,
    pretrained_cnn=False  # Faster for testing
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  - Total parameters: {total_params:,}")
print(f"  - Trainable parameters: {trainable_params:,}")

# Test forward pass
print("\n✓ Testing forward pass...")
batch_size = 4
images = torch.randn(batch_size, 3, 128, 128)
captions = torch.randint(0, 10000, (batch_size, 30))
caption_lengths = torch.tensor([25, 20, 18, 15])

# Forward
predictions, alphas, sorted_captions, decode_lengths, sort_ind = model(
    images, captions, caption_lengths
)

print(f"  - Predictions shape: {predictions.shape}")
print(f"  - Alphas shape: {alphas.shape}")
print(f"  - Decode lengths: {decode_lengths}")

# Test generation
print("\n✓ Testing caption generation...")
single_image = torch.randn(1, 3, 128, 128)
caption, alphas_gen = model.generate_caption(single_image, max_length=20)

print(f"  - Generated caption length: {len(caption)}")
print(f"  - Generated tokens: {caption.tolist()}")

# Test encoder separately
print("\n✓ Testing encoder...")
encoder_out = model.encoder(images)
print(f"  - Encoder output shape: {encoder_out.shape}")

print("\n" + "=" * 70)
print("✅ MODEL TEST PASSED")
print("=" * 70)
