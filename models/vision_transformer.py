"""
Vision Transformer (ViT) + Transformer Decoder for Image Captioning

This module implements a Transformer-based architecture:
- Vision Transformer (ViT) encoder for image feature extraction
- Transformer decoder for caption generation with cross-attention

Architecture:
1. Image -> Patches -> Linear Projection -> Positional Encoding
2. Transformer Encoder layers
3. Transformer Decoder with cross-attention to image features
4. Linear projection to vocabulary

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    EMBEDDING_DIM,
    VOCAB_SIZE,
    MAX_CAPTION_LENGTH,
    DROPOUT,
    IMAGE_SIZE,
    PATCH_SIZE,
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    NUM_ATTENTION_HEADS,
    MLP_RATIO,
    DEVICE
)


class PatchEmbedding(nn.Module):
    """
    Convert image into patch embeddings.
    
    Splits image into non-overlapping patches and projects them.
    
    Args:
        img_size: Input image size (assumed square)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch projection using Conv2d
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (batch, channels, height, width)
        
        Returns:
            Patch embeddings (batch, num_patches, embed_dim)
        """
        # (batch, embed_dim, H/patch, W/patch)
        x = self.projection(x)
        
        # Flatten spatial dimensions
        # (batch, embed_dim, num_patches) -> (batch, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequences.
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_len: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder.
    
    Processes image patches through transformer layers.
    
    Args:
        img_size: Input image size
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (batch, channels, height, width)
        
        Returns:
            Encoded features (batch, num_patches + 1, embed_dim)
        """
        batch_size = x.size(0)
        
        # Get patch embeddings
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoder
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for caption generation.
    
    Generates captions using cross-attention to image features.
    
    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        num_layers: Number of decoder layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        max_length: Maximum sequence length
        dropout: Dropout probability
        embedding_matrix: Optional pre-trained embeddings
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        max_length: int = 30,
        dropout: float = 0.1,
        embedding_matrix: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.data[0] = 0  # PAD token
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=embed_dim,
            max_len=max_length,
            dropout=dropout
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target tokens (batch, seq_len)
            memory: Encoder output (batch, src_len, embed_dim)
            tgt_mask: Causal mask for target
            tgt_key_padding_mask: Padding mask for target
        
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        # Embed tokens
        tgt_embed = self.embedding(tgt)  # (batch, seq_len, embed_dim)
        tgt_embed = self.pos_encoding(tgt_embed)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(
                tgt.size(1), tgt.device
            )
        
        # Transformer decoder
        output = self.transformer(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        output = self.norm(output)
        output = self.dropout(output)
        
        # Project to vocabulary
        logits = self.fc_out(output)
        
        return logits
    
    def generate(
        self,
        memory: torch.Tensor,
        max_length: int = 30,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate caption autoregressively.
        
        Args:
            memory: Encoder output (1, src_len, embed_dim)
            max_length: Maximum caption length
            temperature: Sampling temperature
        
        Returns:
            Tuple of (generated_tokens, attention_weights)
        """
        device = memory.device
        batch_size = memory.size(0)
        
        # Start with SOS token
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Get logits
            logits = self.forward(generated, memory)
            
            # Get next token prediction
            next_token_logits = logits[:, -1, :] / temperature
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token generated
            if next_token.item() == 2:  # EOS token
                break
        
        return generated.squeeze(0), None  # Return tokens, no attention weights for now


class VisionTransformerCaptioning(nn.Module):
    """
    Complete Vision Transformer captioning model.
    
    Combines ViT encoder with Transformer decoder.
    
    Args:
        img_size: Input image size
        patch_size: Patch size for ViT
        embed_dim: Embedding dimension
        encoder_layers: Number of encoder layers
        decoder_layers: Number of decoder layers
        num_heads: Number of attention heads
        vocab_size: Vocabulary size
        max_length: Maximum caption length
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout probability
        embedding_matrix: Optional pre-trained embeddings
    """
    
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 256,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        num_heads: int = 8,
        vocab_size: int = 10000,
        max_length: int = 30,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        embedding_matrix: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Vision Transformer encoder
        self.encoder = VisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            max_length=max_length,
            dropout=dropout,
            embedding_matrix=embedding_matrix
        )
    
    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        caption_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            images: Input images (batch, 3, H, W)
            captions: Target captions (batch, seq_len)
            caption_padding_mask: Padding mask for captions
        
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        # Encode images
        memory = self.encoder(images)
        
        # Create padding mask from captions (PAD = 0)
        # Must be boolean for PyTorch transformer
        if caption_padding_mask is None:
            caption_padding_mask = (captions == 0).bool()
        else:
            caption_padding_mask = caption_padding_mask.bool()
        
        # Decode
        logits = self.decoder(
            captions,
            memory,
            tgt_key_padding_mask=caption_padding_mask
        )
        
        return logits
    
    def generate_caption(
        self,
        image: torch.Tensor,
        max_length: int = 30,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate caption for a single image.
        
        Args:
            image: Input image (1, 3, H, W)
            max_length: Maximum caption length
            temperature: Sampling temperature
        
        Returns:
            Tuple of (generated_tokens, attention_weights)
        """
        # Encode image
        memory = self.encoder(image)
        
        # Generate caption
        tokens, attn = self.decoder.generate(
            memory,
            max_length=max_length,
            temperature=temperature
        )
        
        return tokens, attn


def create_vit_model(
    embedding_matrix: Optional[torch.Tensor] = None,
    img_size: int = 128,
    patch_size: int = 16,
    embed_dim: int = 256,
    encoder_layers: int = 4,  # Reduced for CPU
    decoder_layers: int = 4,  # Reduced for CPU
    num_heads: int = 8,
    vocab_size: int = 10000,
    max_length: int = 30,
    dropout: float = 0.1
) -> VisionTransformerCaptioning:
    """
    Factory function to create Vision Transformer model.
    
    Uses reduced layers for CPU training efficiency.
    
    Args:
        embedding_matrix: Optional pre-trained embeddings
        img_size: Input image size
        patch_size: Patch size
        embed_dim: Embedding dimension
        encoder_layers: Number of encoder layers
        decoder_layers: Number of decoder layers
        num_heads: Number of attention heads
        vocab_size: Vocabulary size
        max_length: Maximum caption length
        dropout: Dropout probability
    
    Returns:
        VisionTransformerCaptioning model
    """
    model = VisionTransformerCaptioning(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        max_length=max_length,
        mlp_ratio=4,
        dropout=dropout,
        embedding_matrix=embedding_matrix
    )
    
    return model


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING VISION TRANSFORMER MODEL")
    print("=" * 70)
    
    # Create model
    print("\n✓ Creating model...")
    model = create_vit_model(
        vocab_size=10000,
        embed_dim=256,
        encoder_layers=2,  # Small for testing
        decoder_layers=2,
        num_heads=8
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n✓ Testing forward pass...")
    batch_size = 2
    images = torch.randn(batch_size, 3, 128, 128)
    captions = torch.randint(1, 10000, (batch_size, 25))
    captions[:, 0] = 1  # SOS token
    
    logits = model(images, captions)
    print(f"  - Input images: {images.shape}")
    print(f"  - Input captions: {captions.shape}")
    print(f"  - Output logits: {logits.shape}")
    
    # Test generation
    print("\n✓ Testing caption generation...")
    single_image = torch.randn(1, 3, 128, 128)
    generated, _ = model.generate_caption(single_image, max_length=20)
    print(f"  - Generated tokens: {generated.shape}")
    print(f"  - Token IDs: {generated.tolist()}")
    
    # Test encoder separately
    print("\n✓ Testing encoder...")
    encoder_out = model.encoder(images)
    print(f"  - Encoder output: {encoder_out.shape}")
    print(f"  - Num patches + cls: {encoder_out.size(1)}")
    
    print("\n" + "=" * 70)
    print("✅ VISION TRANSFORMER TEST PASSED")
    print("=" * 70)
