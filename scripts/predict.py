"""
Inference and Prediction Script for Image Captioning

This module provides utilities for:
- Loading trained models
- Generating captions for single images
- Batch inference
- Beam search decoding
- Visualization of attention weights
- Visualization of generated captions

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import os
import sys
import json
import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    CHECKPOINTS_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, OUTPUTS_DIR,
    VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, IMAGE_FEATURE_DIM,
    NUM_LSTM_LAYERS, DROPOUT, MAX_CAPTION_LENGTH, DEVICE, IMAGE_SIZE
)
from utils.image_preprocessing import ImagePreprocessor
from utils.text_preprocessing import TextPreprocessor
from models.cnn_lstm import create_model as create_cnn_lstm
from models.vision_transformer import create_vit_model


def visualize_caption(
    image_path: str,
    captions: List[Tuple[str, float]],
    emotion: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = True
) -> Optional[str]:
    """
    Create a visualization of an image with its generated captions.
    
    Args:
        image_path: Path to the image file
        captions: List of (caption, score) tuples from beam search
        emotion: Optional emotion label
        output_path: Optional path to save the visualization
        show: Whether to display the plot
        
    Returns:
        Path to saved visualization if output_path is provided
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Create grid: image on left, captions on right
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
    
    # Image subplot
    ax_img = fig.add_subplot(gs[0])
    try:
        img = Image.open(image_path)
        ax_img.imshow(img)
    except Exception as e:
        ax_img.text(0.5, 0.5, f"Could not load image\n{e}", 
                   ha='center', va='center', fontsize=10)
    ax_img.axis('off')
    
    # Get image name for title
    img_name = Path(image_path).stem
    ax_img.set_title(f"Image: {img_name}", fontsize=11, fontweight='bold', pad=10)
    
    # Caption subplot
    ax_cap = fig.add_subplot(gs[1])
    ax_cap.axis('off')
    
    # Build caption text (using ASCII-safe characters)
    lines = []
    if emotion:
        lines.append(f"Emotion: {emotion}\n")
    
    lines.append("Generated Captions:\n")
    lines.append("-" * 35 + "\n")
    
    for i, (caption, score) in enumerate(captions, 1):
        # Wrap long captions
        wrapped = textwrap.fill(caption, width=40)
        lines.append(f"{i}. {wrapped}")
        lines.append(f"   Score: {score:.4f}\n")
    
    caption_text = "\n".join(lines)
    
    ax_cap.text(0.05, 0.95, caption_text, 
               transform=ax_cap.transAxes,
               fontsize=10, fontfamily='monospace',
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"   Visualization saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path if output_path else None


class CaptionGenerator:
    """
    Caption generator for inference.
    
    Supports both CNN+LSTM and Vision Transformer models.
    """
    
    def __init__(
        self,
        model_type: str = 'cnn_lstm',
        checkpoint_path: Optional[str] = None,
        device: str = DEVICE,
        model_config: Optional[Dict] = None
    ):
        """
        Initialize generator.
        
        Args:
            model_type: 'cnn_lstm' or 'vit'
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            model_config: Optional dict with model architecture parameters
        """
        self.model_type = model_type
        self.device = device
        self.model_config = model_config or {}
        
        # Try to load config from outputs folder if checkpoint path is provided
        if checkpoint_path and not model_config:
            self.model_config = self._load_config_from_outputs(checkpoint_path)
        
        # Load text preprocessor
        self.text_preprocessor = TextPreprocessor()
        vocab_path = PROCESSED_DIR / 'vocabulary.json'
        if vocab_path.exists():
            self.text_preprocessor.load_vocabulary(vocab_path)
        else:
            raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")
        
        # Load image preprocessor
        self.image_preprocessor = ImagePreprocessor(
            image_size=IMAGE_SIZE,
            normalize=True,
            augment=False
        )
        
        # Load embeddings
        tfidf_path = EMBEDDINGS_DIR / 'tfidf_embeddings.npy'
        if tfidf_path.exists():
            tfidf_matrix = np.load(tfidf_path)
            self.embedding_matrix = torch.FloatTensor(tfidf_matrix)
        else:
            self.embedding_matrix = None
        
        # Create model
        self.model = self._create_model()
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        
        self.model.to(device)
        self.model.eval()
    
    def _load_config_from_outputs(self, checkpoint_path: str) -> Dict:
        """Try to load config from outputs folder based on checkpoint name."""
        # Extract experiment name from checkpoint path
        # e.g., checkpoints/primary_cnn_high_lr/best_model.pth -> primary_cnn_high_lr
        checkpoint_path = Path(checkpoint_path)
        exp_name = checkpoint_path.parent.name
        
        # Try to find config in outputs folder
        outputs_dir = Path(__file__).parent.parent / 'outputs' / exp_name
        config_path = outputs_dir / 'config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"  - Loaded config from: {config_path}")
                return config
        
        return {}
    
    def _create_model(self) -> torch.nn.Module:
        """Create model based on type."""
        if self.model_type == 'cnn_lstm':
            # Get architecture params from config or use defaults
            embed_dim = self.model_config.get('embed_dim', EMBEDDING_DIM)
            hidden_dim = self.model_config.get('hidden_dim', HIDDEN_DIM)
            attention_dim = self.model_config.get('attention_dim', 512)
            dropout = self.model_config.get('dropout', DROPOUT)
            
            return create_cnn_lstm(
                embedding_matrix=self.embedding_matrix,
                vocab_size=VOCAB_SIZE,
                embed_dim=embed_dim,
                encoder_dim=IMAGE_FEATURE_DIM,
                decoder_dim=hidden_dim,
                attention_dim=attention_dim,
                num_layers=NUM_LSTM_LAYERS,
                dropout=dropout,
                pretrained_cnn=True
            )
        elif self.model_type == 'vit':
            # Get ViT-specific params from config
            embed_dim = self.model_config.get('embed_dim', EMBEDDING_DIM)
            encoder_layers = self.model_config.get('encoder_layers', 4)
            decoder_layers = self.model_config.get('decoder_layers', 4)
            num_heads = self.model_config.get('num_heads', 8)
            dropout = self.model_config.get('dropout', DROPOUT)
            
            return create_vit_model(
                embedding_matrix=self.embedding_matrix,
                vocab_size=VOCAB_SIZE,
                embed_dim=embed_dim,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"✓ Loaded checkpoint from: {checkpoint_path}")
    
    def preprocess_image(
        self,
        image_path: Union[str, Path]
    ) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image tensor (1, 3, H, W)
        """
        image = Image.open(image_path).convert('RGB')
        tensor = self.image_preprocessor.val_transform(image)
        return tensor.unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def generate_caption(
        self,
        image_path: Union[str, Path],
        max_length: int = MAX_CAPTION_LENGTH,
        temperature: float = 1.0
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Generate caption for a single image.
        
        Args:
            image_path: Path to image file
            max_length: Maximum caption length
            temperature: Sampling temperature
        
        Returns:
            Tuple of (caption_string, attention_weights)
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Generate caption
        if self.model_type == 'cnn_lstm':
            tokens, alphas = self.model.generate_caption(image, max_length)
            if alphas is not None:
                alphas = alphas.cpu().numpy()
        else:  # vit
            tokens, alphas = self.model.generate_caption(
                image, max_length, temperature
            )
            if alphas is not None:
                alphas = alphas.cpu().numpy()
        
        # Decode tokens to words
        caption = self.text_preprocessor.decode(
            tokens.cpu().numpy().tolist(),
            skip_special_tokens=True
        )
        
        return caption, alphas
    
    @torch.no_grad()
    def generate_caption_beam_search(
        self,
        image_path: Union[str, Path],
        beam_size: int = 5,
        max_length: int = MAX_CAPTION_LENGTH
    ) -> List[Tuple[str, float]]:
        """
        Generate captions using beam search.
        
        Args:
            image_path: Path to image file
            beam_size: Number of beams
            max_length: Maximum caption length
        
        Returns:
            List of (caption, score) tuples sorted by score
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        if self.model_type == 'cnn_lstm':
            return self._beam_search_lstm(image, beam_size, max_length)
        else:
            return self._beam_search_vit(image, beam_size, max_length)
    
    def _beam_search_lstm(
        self,
        image: torch.Tensor,
        beam_size: int,
        max_length: int
    ) -> List[Tuple[str, float]]:
        """Beam search for CNN+LSTM model."""
        # Encode image
        encoder_out = self.model.encoder(image)
        encoder_dim = encoder_out.size(1)
        
        # Flatten for decoder
        encoder_out = encoder_out.view(1, encoder_dim, -1).permute(0, 2, 1)
        
        # Initialize LSTM state
        h, c = self.model.decoder.init_hidden_state(encoder_out)
        
        # Start with SOS token
        k = beam_size
        vocab_size = self.model.decoder.vocab_size
        
        # Expand encoder output for beam
        encoder_out = encoder_out.expand(k, -1, -1)
        
        # Tensor to store top k sequences
        seqs = torch.ones(k, 1, dtype=torch.long, device=self.device)
        top_k_scores = torch.zeros(k, 1, device=self.device)
        
        # Lists to store completed sequences
        complete_seqs = []
        complete_seqs_scores = []
        
        # Expand h and c for beam
        h = h.expand(-1, k, -1).contiguous()
        c = c.expand(-1, k, -1).contiguous()
        
        for step in range(max_length):
            # Get embeddings
            embeddings = self.model.decoder.embedding(seqs[:, -1])
            
            # Attention
            awe, alpha = self.model.decoder.attention(encoder_out, h[-1])
            gate = torch.sigmoid(self.model.decoder.f_beta(h[-1]))
            awe = gate * awe
            
            # LSTM input
            lstm_input = torch.cat([embeddings, awe], dim=1).unsqueeze(1)
            
            # LSTM forward
            _, (h, c) = self.model.decoder.lstm(lstm_input, (h, c))
            
            # Predict
            scores = self.model.decoder.fc(h[-1])
            scores = F.log_softmax(scores, dim=1)
            
            # Add to previous scores
            scores = top_k_scores.expand_as(scores) + scores
            
            # For first step, all beams have same score
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)
            
            # Convert to beam indices and word indices
            prev_word_inds = top_k_words // vocab_size
            next_word_inds = top_k_words % vocab_size
            
            # Add to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            
            # Check for complete sequences
            incomplete_inds = []
            for idx, next_word in enumerate(next_word_inds):
                if next_word == 2:  # EOS token
                    complete_seqs.append(seqs[idx].tolist())
                    complete_seqs_scores.append(top_k_scores[idx].item())
                else:
                    incomplete_inds.append(idx)
            
            if len(incomplete_inds) == 0:
                break
            
            # Continue with incomplete sequences
            incomplete_inds = torch.tensor(incomplete_inds, device=self.device)
            seqs = seqs[incomplete_inds]
            h = h[:, prev_word_inds[incomplete_inds]]
            c = c[:, prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k = len(incomplete_inds)
            
            if k == 0:
                break
        
        # Add remaining incomplete sequences
        if len(complete_seqs) == 0:
            complete_seqs = seqs.tolist()
            complete_seqs_scores = top_k_scores.squeeze(1).tolist()
        
        # Sort by score and decode
        results = []
        for seq, score in sorted(zip(complete_seqs, complete_seqs_scores),
                                 key=lambda x: -x[1]):
            caption = self.text_preprocessor.decode(seq, skip_special_tokens=True)
            results.append((caption, score))
        
        return results[:beam_size]
    
    def _beam_search_vit(
        self,
        image: torch.Tensor,
        beam_size: int,
        max_length: int
    ) -> List[Tuple[str, float]]:
        """Beam search for Vision Transformer model."""
        # Encode image
        memory = self.model.encoder(image)
        
        vocab_size = self.model.decoder.vocab_size
        k = beam_size
        
        # Expand memory for beam
        memory = memory.expand(k, -1, -1)
        
        # Initialize with SOS token
        seqs = torch.ones(k, 1, dtype=torch.long, device=self.device)
        top_k_scores = torch.zeros(k, device=self.device)
        
        complete_seqs = []
        complete_seqs_scores = []
        
        for step in range(max_length):
            # Get logits
            logits = self.model.decoder(seqs, memory)
            scores = F.log_softmax(logits[:, -1, :], dim=1)
            
            # Add to previous scores
            scores = top_k_scores.unsqueeze(1) + scores
            
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)
            
            prev_word_inds = top_k_words // vocab_size
            next_word_inds = top_k_words % vocab_size
            
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            
            # Check for complete sequences
            incomplete_inds = []
            for idx, next_word in enumerate(next_word_inds):
                if next_word == 2:  # EOS
                    complete_seqs.append(seqs[idx].tolist())
                    complete_seqs_scores.append(top_k_scores[idx].item())
                else:
                    incomplete_inds.append(idx)
            
            if len(incomplete_inds) == 0:
                break
            
            incomplete_inds = torch.tensor(incomplete_inds, device=self.device)
            seqs = seqs[incomplete_inds]
            memory = memory[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds]
            k = len(incomplete_inds)
        
        if len(complete_seqs) == 0:
            complete_seqs = seqs.tolist()
            complete_seqs_scores = top_k_scores.tolist()
        
        results = []
        for seq, score in sorted(zip(complete_seqs, complete_seqs_scores),
                                 key=lambda x: -x[1]):
            caption = self.text_preprocessor.decode(seq, skip_special_tokens=True)
            results.append((caption, score))
        
        return results[:beam_size]
    
    @torch.no_grad()
    def generate_batch(
        self,
        image_paths: List[Union[str, Path]],
        max_length: int = MAX_CAPTION_LENGTH
    ) -> List[str]:
        """
        Generate captions for multiple images.
        
        Args:
            image_paths: List of image paths
            max_length: Maximum caption length
        
        Returns:
            List of caption strings
        """
        captions = []
        for path in image_paths:
            caption, _ = self.generate_caption(path, max_length)
            captions.append(caption)
        return captions


def list_available_models():
    """List all available model checkpoints with their training info."""
    print("\n" + "=" * 70)
    print("AVAILABLE MODEL CHECKPOINTS")
    print("=" * 70)
    
    if not CHECKPOINTS_DIR.exists():
        print("\nNo checkpoints directory found!")
        return
    
    for model_dir in sorted(CHECKPOINTS_DIR.iterdir()):
        if model_dir.is_dir():
            checkpoints = list(model_dir.glob("*.pth"))
            if checkpoints:
                print(f"\n📁 {model_dir.name}/")
                
                # Check for training history/results
                results_paths = [
                    Path(f"outputs/{model_dir.name}/training_results.json"),
                    Path(f"outputs/{model_dir.name}/results.json"),
                    Path(f"outputs/{model_dir.name}/training_history.json"),
                ]
                
                for results_path in results_paths:
                    if results_path.exists():
                        try:
                            with open(results_path, encoding='utf-8') as f:
                                results = json.load(f)
                            if 'best_bleu' in results:
                                print(f"   BLEU: {results['best_bleu']:.4f}")
                            if 'best_val_loss' in results:
                                print(f"   Val Loss: {results['best_val_loss']:.4f}")
                            if 'total_epochs' in results:
                                print(f"   Epochs: {results['total_epochs']}")
                            if 'model_type' in results:
                                print(f"   Model: {results['model_type']}")
                            break
                        except Exception as e:
                            pass
                
                for cp in sorted(checkpoints):
                    size_mb = cp.stat().st_size / (1024 * 1024)
                    marker = "⭐" if "best" in cp.name.lower() else "  "
                    print(f"   {marker} {cp.name} ({size_mb:.1f} MB)")


def run_demo(generator):
    """Run demo with sample images from the dataset."""
    import random
    
    print("\n" + "=" * 70)
    print("DEMO: CAPTION GENERATION")
    print("=" * 70)
    
    # Find sample images
    from utils.constants import WIKIART_DIR
    processed_images = PROCESSED_DIR / 'images'
    
    sample_images = []
    
    # Try processed images first (smaller, faster to load)
    if processed_images.exists():
        for style_dir in processed_images.iterdir():
            if style_dir.is_dir():
                images = list(style_dir.glob("*.jpg"))[:3]
                sample_images.extend(images)
                if len(sample_images) >= 10:
                    break
    
    # Fallback to raw images
    if not sample_images and WIKIART_DIR.exists():
        for style_dir in WIKIART_DIR.iterdir():
            if style_dir.is_dir():
                images = list(style_dir.glob("*.jpg"))[:3]
                sample_images.extend(images)
                if len(sample_images) >= 10:
                    break
    
    if not sample_images:
        print("\n❌ No sample images found in data/processed/images or data/raw/wikiart")
        return
    
    # Select random samples
    samples = random.sample(sample_images, min(5, len(sample_images)))
    
    print(f"\n✓ Generating captions for {len(samples)} random images...\n")
    
    for i, img_path in enumerate(samples, 1):
        style = img_path.parent.name.replace('_', ' ')
        print(f"\n{'─' * 60}")
        print(f"🎨 Image {i}: {img_path.name}")
        print(f"   Style: {style}")
        
        try:
            # Greedy caption
            caption, alphas = generator.generate_caption(img_path)
            print(f"\n   📝 Generated Caption:")
            print(f"   \"{caption}\"")
            
            # Beam search alternatives
            try:
                beam_results = generator.generate_caption_beam_search(img_path, beam_size=3)
                if len(beam_results) > 1:
                    print(f"\n   🔍 Alternative captions (beam search):")
                    for j, (cap, score) in enumerate(beam_results[:3], 1):
                        print(f"      {j}. \"{cap}\" (score: {score:.2f})")
            except Exception:
                pass  # Skip beam search if it fails
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n{'─' * 60}")
    print("✅ Demo complete!")


def interactive_mode(generator):
    """Run interactive caption generation session."""
    print("\n" + "=" * 70)
    print("INTERACTIVE CAPTION GENERATION")
    print("=" * 70)
    print("\nCommands:")
    print("  - Enter an image path to generate caption")
    print("  - Type 'demo' to run demo with sample images")
    print("  - Type 'beam' to toggle beam search (currently OFF)")
    print("  - Type 'quit' or 'exit' to exit")
    print("=" * 70)
    
    use_beam_search = False
    
    while True:
        try:
            user_input = input("\n🖼️  Image path: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'demo':
                run_demo(generator)
                continue
            
            if user_input.lower() == 'beam':
                use_beam_search = not use_beam_search
                status = "ON" if use_beam_search else "OFF"
                print(f"   Beam search: {status}")
                continue
            
            if not os.path.exists(user_input):
                print(f"   ❌ File not found: {user_input}")
                print("   💡 Tip: Use full path like 'data/processed/images/Baroque/image.jpg'")
                continue
            
            print("\n   ⏳ Generating caption...")
            
            if use_beam_search:
                beam_results = generator.generate_caption_beam_search(user_input, beam_size=5)
                print(f"\n   📝 Beam search results:")
                for i, (cap, score) in enumerate(beam_results, 1):
                    marker = "→" if i == 1 else " "
                    print(f"   {marker} {i}. \"{cap}\" (score: {score:.2f})")
            else:
                caption, alphas = generator.generate_caption(user_input)
                print(f"\n   📝 Generated Caption:")
                print(f"   \"{caption}\"")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")


def find_checkpoint(model_type: str = None, experiment_name: str = None) -> Optional[Path]:
    """Find a suitable checkpoint file."""
    # Priority order for checkpoints
    search_paths = []
    
    if experiment_name:
        search_paths.append(CHECKPOINTS_DIR / experiment_name / 'best_model.pth')
        search_paths.append(CHECKPOINTS_DIR / experiment_name / 'latest_checkpoint.pth')
    
    # Default search order
    search_paths.extend([
        CHECKPOINTS_DIR / 'local_3k' / 'best_model.pth',
        CHECKPOINTS_DIR / 'local_3k_vit' / 'best_model.pth',
        CHECKPOINTS_DIR / 'test_300' / 'best_model.pth',
        CHECKPOINTS_DIR / 'best_model.pth',
    ])
    
    for path in search_paths:
        if path.exists():
            return path
    
    # Search for any checkpoint
    for cp_dir in CHECKPOINTS_DIR.iterdir():
        if cp_dir.is_dir():
            best = cp_dir / 'best_model.pth'
            if best.exists():
                return best
    
    return None


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Image Caption Generation - Inference and Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict.py --list-models
  python scripts/predict.py --demo
  python scripts/predict.py --interactive
  python scripts/predict.py --image path/to/image.jpg
  python scripts/predict.py --image path/to/image.jpg --emotion awe
  python scripts/predict.py --image path/to/image.jpg --top-k 3
  python scripts/predict.py --model checkpoints/local_3k/best_model.pth --image image.jpg
  python scripts/predict.py --model-type vit --experiment local_3k_vit --demo
        """
    )
    
    # Available emotions from ArtEmis dataset
    EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement', 
                'anger', 'disgust', 'fear', 'sadness', 'nostalgia', 'something else']
    
    parser.add_argument("--list-models", action="store_true",
                       help="List all available model checkpoints")
    parser.add_argument("--model", type=str,
                       help="Path to specific model checkpoint")
    parser.add_argument("--model-type", type=str, choices=['cnn_lstm', 'vit'],
                       default='cnn_lstm', help="Model architecture type")
    parser.add_argument("--experiment", type=str,
                       help="Experiment name to load checkpoint from")
    parser.add_argument("--image", type=str,
                       help="Path to single image (.jpg) for caption generation")
    parser.add_argument("--emotion", type=str, choices=EMOTIONS,
                       help="Emotion to condition the caption (e.g., awe, fear, joy)")
    parser.add_argument("--top-k", type=int, default=3,
                       help="Number of top captions to generate (default: 3)")
    parser.add_argument("--beam-size", type=int, default=5,
                       help="Beam search size (default: 5)")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization of image with captions")
    parser.add_argument("--save-viz", type=str,
                       help="Path to save caption visualization image")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive mode")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo with sample images")
    
    args = parser.parse_args()
    
    # List models command
    if args.list_models:
        list_available_models()
        return
    
    # Find checkpoint
    if args.model:
        checkpoint_path = Path(args.model)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {args.model}")
            return
    else:
        checkpoint_path = find_checkpoint(args.model_type, args.experiment)
        if checkpoint_path:
            print(f"✓ Found checkpoint: {checkpoint_path}")
        else:
            print("❌ No checkpoint found. Use --model to specify path or --list-models to see available.")
            print("   Running with untrained model for testing purposes...")
    
    # Determine model type from checkpoint path or argument
    model_type = args.model_type
    if checkpoint_path and 'vit' in str(checkpoint_path).lower():
        model_type = 'vit'
    
    print(f"\n{'=' * 70}")
    print("IMAGE CAPTIONING - MODEL TESTING")
    print(f"{'=' * 70}")
    print(f"Model Type: {model_type.upper()}")
    print(f"Checkpoint: {checkpoint_path if checkpoint_path else 'None (untrained)'}")
    print(f"Device: {DEVICE}")
    print(f"{'=' * 70}")
    
    # Initialize generator
    print("\n⏳ Loading model...")
    try:
        generator = CaptionGenerator(
            model_type=model_type,
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            device=DEVICE
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run appropriate mode
    if args.demo:
        run_demo(generator)
    elif args.image:
        print(f"\n🖼️  Image: {args.image}")
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        
        # Show emotion if specified
        if args.emotion:
            print(f"🎭 Emotion: {args.emotion}")
        
        print(f"\n{'=' * 50}")
        print(f"TOP-{args.top_k} GENERATED CAPTIONS")
        print(f"{'=' * 50}")
        
        # Generate top-k captions using beam search
        try:
            beam_results = generator.generate_caption_beam_search(
                args.image, beam_size=max(args.beam_size, args.top_k)
            )
            
            # Prepare captions with emotion prefix
            display_results = []
            for cap, score in beam_results[:args.top_k]:
                if args.emotion:
                    display_cap = f"[{args.emotion}] {cap}"
                else:
                    display_cap = cap
                display_results.append((display_cap, score))
            
            # Display top-k results in terminal
            for i, (cap, score) in enumerate(display_results, 1):
                print(f"\n  {i}. \"{cap}\"")
                print(f"     Score: {score:.4f}")
            
            # Visualize if requested
            if args.visualize or args.save_viz:
                output_path = args.save_viz
                if not output_path and args.visualize:
                    # Auto-generate output path
                    output_dir = OUTPUTS_DIR / 'visualizations'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    img_name = Path(args.image).stem
                    output_path = str(output_dir / f'{img_name}_captions.png')
                
                visualize_caption(
                    image_path=args.image,
                    captions=display_results,
                    emotion=args.emotion,
                    output_path=output_path,
                    show=args.visualize
                )
            
        except Exception as e:
            print(f"❌ Beam search error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to greedy decoding
            print("\n  Falling back to greedy decoding...")
            caption, alphas = generator.generate_caption(args.image)
            if args.emotion:
                caption = f"[{args.emotion}] {caption}"
            print(f"\n  1. \"{caption}\"")
        
        print(f"\n{'=' * 50}")
        
    elif args.interactive:
        interactive_mode(generator)
    else:
        # Default to demo if no specific action
        print("\n💡 No action specified. Running demo...")
        print("   Use --help to see all options")
        run_demo(generator)


if __name__ == "__main__":
    main()
