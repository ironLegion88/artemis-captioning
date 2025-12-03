"""
CNN+LSTM Model Architecture for ArtEmis Caption Generation

This module implements the CNN+LSTM encoder-decoder architecture:
- Encoder: CNN (ResNet or custom) to extract image features
- Decoder: LSTM with attention mechanism to generate captions

Architecture Overview:
1. CNN Encoder: Extracts visual features from images (128x128 -> feature vector)
2. Attention Mechanism: Focuses on relevant image regions during generation
3. LSTM Decoder: Generates captions word-by-word with attention

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torchvision.models as models

from utils.constants import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    IMAGE_FEATURE_DIM,
    NUM_LSTM_LAYERS,
    DROPOUT,
    VOCAB_SIZE,
    MAX_CAPTION_LENGTH,
    DEVICE
)


class CNNEncoder(nn.Module):
    """
    CNN Encoder for extracting image features.
    
    Uses pre-trained ResNet18 (modified for 128x128 input) or custom CNN.
    Extracts spatial feature maps for attention mechanism.
    
    Args:
        encoded_image_size: Size of encoded feature maps (default: 8)
        feature_dim: Dimension of feature vectors (default: 256)
        pretrained: Whether to use pretrained weights (ImageNet)
    """
    
    def __init__(
        self,
        encoded_image_size: int = 8,
        feature_dim: int = IMAGE_FEATURE_DIM,
        pretrained: bool = True
    ):
        super(CNNEncoder, self).__init__()
        
        self.encoded_image_size = encoded_image_size
        self.feature_dim = feature_dim
        
        # Use ResNet18 as base (lightweight for CPU)
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove final fully connected layer and avgpool
        # We want spatial features (N, C, H, W) not (N, C)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # ResNet18 outputs 512 channels
        # Add adaptive pooling to get desired spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        # Project to desired feature dimension
        self.projection = nn.Sequential(
            nn.Conv2d(512, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT)
        )
        
        # Fine-tune only last layers (save computation)
        self._freeze_base_layers()
    
    def _freeze_base_layers(self):
        """Freeze early layers of ResNet (feature extraction is good enough)."""
        # Freeze first 6 layers (conv1, bn1, relu, maxpool, layer1, layer2)
        for i, child in enumerate(self.resnet.children()):
            if i < 6:
                for param in child.parameters():
                    param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            images: Input images (batch_size, 3, 128, 128)
        
        Returns:
            Encoded features (batch_size, feature_dim, enc_size, enc_size)
        """
        # Extract features
        features = self.resnet(images)  # (batch, 512, H, W)
        
        # Adaptive pooling to fixed size
        features = self.adaptive_pool(features)  # (batch, 512, enc_size, enc_size)
        
        # Project to target dimension
        features = self.projection(features)  # (batch, feature_dim, enc_size, enc_size)
        
        return features


class Attention(nn.Module):
    """
    Attention mechanism for focusing on relevant image regions.
    
    Implements Bahdanau (additive) attention:
    - Query: Hidden state from LSTM decoder
    - Keys/Values: Encoded image features from CNN
    
    Args:
        encoder_dim: Dimension of encoder features
        decoder_dim: Dimension of decoder hidden state
        attention_dim: Dimension of attention layer
    """
    
    def __init__(
        self,
        encoder_dim: int = IMAGE_FEATURE_DIM,
        decoder_dim: int = HIDDEN_DIM,
        attention_dim: int = 512
    ):
        super(Attention, self).__init__()
        
        # Linear layers for attention computation
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention mechanism.
        
        Args:
            encoder_out: Encoded image features (batch, num_pixels, encoder_dim)
            decoder_hidden: Decoder hidden state (batch, decoder_dim)
        
        Returns:
            Tuple of:
            - attention_weighted_encoding: Context vector (batch, encoder_dim)
            - alpha: Attention weights (batch, num_pixels)
        """
        # Compute attention scores
        att1 = self.encoder_att(encoder_out)  # (batch, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch, attention_dim)
        
        # Add decoder attention to each pixel
        att2 = att2.unsqueeze(1)  # (batch, 1, attention_dim)
        
        # Combined attention score
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch, num_pixels)
        
        # Softmax to get attention weights
        alpha = self.softmax(att)  # (batch, num_pixels)
        
        # Apply attention weights to encoder output
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        # (batch, encoder_dim)
        
        return attention_weighted_encoding, alpha


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder with attention for caption generation.
    
    Generates captions word-by-word using:
    - Word embeddings (from pre-trained or learned)
    - Attention over image features
    - LSTM to maintain generation state
    
    Args:
        attention_dim: Dimension of attention layer
        embed_dim: Dimension of word embeddings
        decoder_dim: Dimension of LSTM hidden state
        vocab_size: Size of vocabulary
        encoder_dim: Dimension of encoder features
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        embedding_matrix: Optional pre-trained embeddings
    """
    
    def __init__(
        self,
        attention_dim: int = 512,
        embed_dim: int = EMBEDDING_DIM,
        decoder_dim: int = HIDDEN_DIM,
        vocab_size: int = VOCAB_SIZE,
        encoder_dim: int = IMAGE_FEATURE_DIM,
        num_layers: int = NUM_LSTM_LAYERS,
        dropout: float = DROPOUT,
        embedding_matrix: Optional[torch.Tensor] = None
    ):
        super(LSTMDecoder, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Attention mechanism
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout_emb = nn.Dropout(dropout)
        
        # LSTM decoder
        # Input: [embedding + attention context]
        self.lstm = nn.LSTM(
            embed_dim + encoder_dim,
            decoder_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Linear layers
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # Gate for attention
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        self.dropout_fc = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        # Load pre-trained embeddings AFTER weight initialization
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            # Keep PAD token as zeros
            self.embedding.weight.data[0] = 0
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state and cell state from encoder output.
        
        Args:
            encoder_out: Encoded image features (batch, num_pixels, encoder_dim)
        
        Returns:
            Tuple of (h0, c0) for LSTM
        """
        # Take mean of encoder output across pixels
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch, encoder_dim)
        
        # Initialize hidden and cell states
        h = self.init_h(mean_encoder_out)  # (batch, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch, decoder_dim)
        
        # Expand for num_layers
        h = h.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c = c.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        
        return h, c
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder (training mode).
        
        Args:
            encoder_out: Encoded image features (batch, feature_dim, enc_size, enc_size)
            captions: Target captions (batch, max_length)
            caption_lengths: Actual caption lengths (batch,)
        
        Returns:
            Tuple of:
            - predictions: Word predictions (batch, max_length-1, vocab_size)
            - alphas: Attention weights (batch, max_length-1, num_pixels)
            - sorted_captions: Sorted captions by length
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(1)
        
        # Flatten image features
        encoder_out = encoder_out.view(batch_size, encoder_dim, -1)  # (batch, encoder_dim, num_pixels)
        encoder_out = encoder_out.permute(0, 2, 1)  # (batch, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Sort by decreasing caption length (for pack_padded_sequence)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        captions = captions[sort_ind]
        
        # Embeddings
        embeddings = self.embedding(captions)  # (batch, max_length, embed_dim)
        embeddings = self.dropout_emb(embeddings)
        
        # Initialize LSTM hidden state
        h, c = self.init_hidden_state(encoder_out)
        
        # We won't decode at the <EOS> position, since we've finished generating
        decode_lengths = (caption_lengths - 1).tolist()
        max_decode_length = max(decode_lengths)
        
        # Create tensors to hold predictions and alphas
        predictions = torch.zeros(batch_size, max_decode_length, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_length, num_pixels).to(encoder_out.device)
        
        # At each time step, decode by attention-weighting encoder output
        for t in range(max_decode_length):
            # Get batch size at this timestep (decreases as sequences end)
            batch_size_t = sum([l > t for l in decode_lengths])
            
            # Attention
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t],
                h[-1, :batch_size_t]  # Use last layer hidden state
            )
            
            # Gate for attention
            gate = torch.sigmoid(self.f_beta(h[-1, :batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # LSTM input: [embedding, attention context]
            lstm_input = torch.cat([
                embeddings[:batch_size_t, t, :],
                attention_weighted_encoding
            ], dim=1).unsqueeze(1)  # (batch_size_t, 1, embed_dim + encoder_dim)
            
            # LSTM forward
            lstm_out, (h, c) = self.lstm(
                lstm_input,
                (h[:, :batch_size_t], c[:, :batch_size_t])
            )
            
            # Predict next word
            preds = self.fc(self.dropout_fc(lstm_out.squeeze(1)))  # (batch_size_t, vocab_size)
            
            # Store predictions and attention weights
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        
        return predictions, alphas, captions, decode_lengths, sort_ind
    
    def predict(
        self,
        encoder_out: torch.Tensor,
        max_length: int = MAX_CAPTION_LENGTH,
        beam_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate caption for a single image (inference mode).
        
        Args:
            encoder_out: Encoded image features (1, feature_dim, enc_size, enc_size)
            max_length: Maximum caption length
            beam_size: Beam search width (1 = greedy)
        
        Returns:
            Tuple of:
            - caption: Generated caption tokens (max_length,)
            - alphas: Attention weights (max_length, num_pixels)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(1)
        
        # Flatten image features
        encoder_out = encoder_out.view(batch_size, encoder_dim, -1)
        encoder_out = encoder_out.permute(0, 2, 1)
        num_pixels = encoder_out.size(1)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)
        
        # Start with <SOS> token
        input_word = torch.tensor([1]).to(encoder_out.device)  # SOS token ID
        
        captions = []
        alphas_list = []
        
        for t in range(max_length):
            # Embed current word
            embeddings = self.embedding(input_word).unsqueeze(1)  # (1, 1, embed_dim)
            
            # Attention
            attention_weighted_encoding, alpha = self.attention(
                encoder_out,
                h[-1]  # Last layer hidden state
            )
            
            # Gate
            gate = torch.sigmoid(self.f_beta(h[-1]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # LSTM input
            lstm_input = torch.cat([
                embeddings.squeeze(1),
                attention_weighted_encoding
            ], dim=1).unsqueeze(1)
            
            # LSTM forward
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Predict next word
            preds = self.fc(lstm_out.squeeze(1))  # (1, vocab_size)
            
            # Get most likely word
            _, input_word = preds.max(dim=1)
            
            # Store
            captions.append(input_word.item())
            alphas_list.append(alpha.squeeze(0))
            
            # Stop if <EOS> token generated
            if input_word.item() == 2:  # EOS token ID
                break
        
        captions = torch.tensor(captions)
        alphas = torch.stack(alphas_list) if alphas_list else torch.zeros(0, num_pixels)
        
        return captions, alphas


class ImageCaptioningModel(nn.Module):
    """
    Complete CNN+LSTM Image Captioning Model.
    
    Combines encoder and decoder into a single model.
    
    Args:
        encoder: CNN encoder
        decoder: LSTM decoder with attention
    """
    
    def __init__(
        self,
        encoder: CNNEncoder,
        decoder: LSTMDecoder
    ):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete model.
        
        Args:
            images: Input images (batch, 3, 128, 128)
            captions: Target captions (batch, max_length)
            caption_lengths: Caption lengths (batch,)
        
        Returns:
            Tuple from decoder forward pass
        """
        # Encode images
        encoder_out = self.encoder(images)
        
        # Decode captions
        predictions, alphas, sorted_captions, decode_lengths, sort_ind = self.decoder(
            encoder_out,
            captions,
            caption_lengths
        )
        
        return predictions, alphas, sorted_captions, decode_lengths, sort_ind
    
    def generate_caption(
        self,
        image: torch.Tensor,
        max_length: int = MAX_CAPTION_LENGTH
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate caption for a single image.
        
        Args:
            image: Input image (1, 3, 128, 128)
            max_length: Maximum caption length
        
        Returns:
            Tuple of (caption, alphas)
        """
        # Encode image
        encoder_out = self.encoder(image)
        
        # Generate caption
        caption, alphas = self.decoder.predict(encoder_out, max_length)
        
        return caption, alphas


def create_model(
    embedding_matrix: Optional[torch.Tensor] = None,
    vocab_size: int = VOCAB_SIZE,
    embed_dim: int = EMBEDDING_DIM,
    encoder_dim: int = IMAGE_FEATURE_DIM,
    decoder_dim: int = HIDDEN_DIM,
    attention_dim: int = 512,
    num_layers: int = NUM_LSTM_LAYERS,
    dropout: float = DROPOUT,
    pretrained_cnn: bool = True
) -> ImageCaptioningModel:
    """
    Factory function to create the complete model.
    
    Args:
        embedding_matrix: Optional pre-trained word embeddings
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        encoder_dim: CNN encoder output dimension
        decoder_dim: LSTM hidden state dimension
        attention_dim: Attention mechanism dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        pretrained_cnn: Whether to use pretrained CNN weights
    
    Returns:
        Complete ImageCaptioningModel
    """
    # Create encoder
    encoder = CNNEncoder(
        encoded_image_size=8,
        feature_dim=encoder_dim,
        pretrained=pretrained_cnn
    )
    
    # Create decoder
    decoder = LSTMDecoder(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=encoder_dim,
        num_layers=num_layers,
        dropout=dropout,
        embedding_matrix=embedding_matrix
    )
    
    # Create complete model
    model = ImageCaptioningModel(encoder, decoder)
    
    return model


if __name__ == "__main__":
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
    
    # Test generation
    print("\n✓ Testing caption generation...")
    single_image = torch.randn(1, 3, 128, 128)
    caption, alphas = model.generate_caption(single_image, max_length=20)
    
    print(f"  - Generated caption length: {len(caption)}")
    print(f"  - Generated tokens: {caption.tolist()}")
    
    print("\n" + "=" * 70)
    print("✅ MODEL TEST PASSED")
    print("=" * 70)
