"""
Visualization Utilities for Image Captioning

This module provides visualization tools for:
- Attention heatmaps on images
- Training curves (loss, BLEU)
- Caption comparison displays
- Model architecture diagrams

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_bleu', etc.
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss plot
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # BLEU plot
    ax2 = axes[1]
    if 'val_bleu' in history:
        ax2.plot(epochs, history['val_bleu'], 'g-', label='Val BLEU-4')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('BLEU Score')
        ax2.set_title('Validation BLEU Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax3 = axes[2]
    if 'learning_rate' in history:
        ax3.plot(epochs, history['learning_rate'], 'm-', label='Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training history plot to: {save_path}")
    
    return fig


def plot_attention_heatmap(
    image: Union[str, Path, np.ndarray, Image.Image],
    attention_weights: np.ndarray,
    words: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Plot attention heatmaps over image for each word.
    
    Args:
        image: Image path, array, or PIL Image
        attention_weights: Attention weights (seq_len, num_patches)
        words: List of words in caption
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Load image if needed
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    img_array = np.array(image)
    
    # Calculate grid size from attention weights
    num_patches = attention_weights.shape[1]
    grid_size = int(np.sqrt(num_patches))
    
    # Number of words to show (max 10)
    num_words = min(len(words), 10)
    
    # Create figure
    ncols = min(5, num_words)
    nrows = (num_words + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_words):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        
        # Get attention for this word
        attn = attention_weights[idx]
        
        # Reshape to grid
        attn = attn.reshape(grid_size, grid_size)
        
        # Resize attention to image size
        attn_resized = np.array(Image.fromarray(attn).resize(
            (img_array.shape[1], img_array.shape[0]),
            resample=Image.BILINEAR
        ))
        
        # Normalize
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        # Display image with attention overlay
        ax.imshow(img_array)
        ax.imshow(attn_resized, alpha=0.6, cmap='jet')
        ax.set_title(words[idx], fontsize=12)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_words, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')
    
    plt.suptitle('Attention Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved attention visualization to: {save_path}")
    
    return fig


def plot_caption_comparison(
    image: Union[str, Path, np.ndarray, Image.Image],
    ground_truth: List[str],
    generated: str,
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot image with ground truth and generated captions.
    
    Args:
        image: Image path, array, or PIL Image
        ground_truth: List of ground truth captions
        generated: Generated caption
        metrics: Optional dictionary of evaluation metrics
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Load image
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    img_array = np.array(image)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                    gridspec_kw={'width_ratios': [1, 1.5]})
    
    # Display image
    ax1.imshow(img_array)
    ax1.axis('off')
    ax1.set_title('Image', fontsize=12, fontweight='bold')
    
    # Display captions
    ax2.axis('off')
    
    text_y = 0.95
    ax2.text(0, text_y, 'Generated Caption:', fontsize=11, fontweight='bold',
             transform=ax2.transAxes, verticalalignment='top')
    text_y -= 0.08
    ax2.text(0.02, text_y, f'"{generated}"', fontsize=10, color='blue',
             transform=ax2.transAxes, verticalalignment='top',
             wrap=True)
    
    text_y -= 0.15
    ax2.text(0, text_y, 'Ground Truth Captions:', fontsize=11, fontweight='bold',
             transform=ax2.transAxes, verticalalignment='top')
    
    for i, gt in enumerate(ground_truth[:3]):  # Show max 3
        text_y -= 0.08
        ax2.text(0.02, text_y, f'{i+1}. "{gt}"', fontsize=10, color='green',
                 transform=ax2.transAxes, verticalalignment='top',
                 wrap=True)
    
    # Display metrics if provided
    if metrics:
        text_y -= 0.15
        ax2.text(0, text_y, 'Evaluation Metrics:', fontsize=11, fontweight='bold',
                 transform=ax2.transAxes, verticalalignment='top')
        
        for metric, value in metrics.items():
            text_y -= 0.06
            color = 'darkgreen' if value > 0.3 else 'orange' if value > 0.1 else 'red'
            ax2.text(0.02, text_y, f'{metric}: {value:.4f}', fontsize=10,
                     color=color, transform=ax2.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved caption comparison to: {save_path}")
    
    return fig


def plot_sample_predictions(
    images: List[Union[str, Path]],
    predictions: List[str],
    ground_truths: List[List[str]],
    save_path: Optional[str] = None,
    ncols: int = 3,
    figsize_per_image: Tuple[int, int] = (4, 5)
) -> plt.Figure:
    """
    Plot grid of images with predictions and ground truths.
    
    Args:
        images: List of image paths
        predictions: List of predicted captions
        ground_truths: List of ground truth caption lists
        save_path: Optional path to save figure
        ncols: Number of columns
        figsize_per_image: Size per image in figure
    
    Returns:
        Matplotlib figure
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    
    figsize = (figsize_per_image[0] * ncols, figsize_per_image[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        
        # Load and display image
        img = Image.open(images[idx]).convert('RGB')
        ax.imshow(np.array(img))
        ax.axis('off')
        
        # Add captions as title
        pred = predictions[idx][:50] + '...' if len(predictions[idx]) > 50 else predictions[idx]
        gt = ground_truths[idx][0][:40] + '...' if len(ground_truths[idx][0]) > 40 else ground_truths[idx][0]
        
        ax.set_title(f'Pred: {pred}\nGT: {gt}', fontsize=8, wrap=True)
    
    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved sample predictions to: {save_path}")
    
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Compare metrics across different models/experiments.
    
    Args:
        metrics_dict: Dictionary mapping model names to metric dictionaries
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        values = [metrics_dict[model].get(m, 0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=colors[i])
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved metrics comparison to: {save_path}")
    
    return fig


def create_attention_gif(
    image: Union[str, Path, Image.Image],
    attention_weights: np.ndarray,
    words: List[str],
    save_path: str,
    duration: int = 500
):
    """
    Create animated GIF showing attention over words.
    
    Args:
        image: Image path or PIL Image
        attention_weights: Attention weights (seq_len, num_patches)
        words: List of words
        save_path: Path to save GIF
        duration: Duration per frame in milliseconds
    """
    from PIL import Image as PILImage
    
    # Load image
    if isinstance(image, (str, Path)):
        image = PILImage.open(image).convert('RGB')
    
    img_array = np.array(image)
    img_size = img_array.shape[:2]
    
    # Calculate grid size
    num_patches = attention_weights.shape[1]
    grid_size = int(np.sqrt(num_patches))
    
    frames = []
    
    for idx, word in enumerate(words):
        if idx >= attention_weights.shape[0]:
            break
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Get attention
        attn = attention_weights[idx].reshape(grid_size, grid_size)
        attn_resized = np.array(PILImage.fromarray(attn).resize(
            (img_size[1], img_size[0]),
            resample=PILImage.BILINEAR
        ))
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        # Display
        ax.imshow(img_array)
        ax.imshow(attn_resized, alpha=0.6, cmap='jet')
        ax.set_title(f'Word: "{word}"', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Convert to PIL Image
        fig.canvas.draw()
        frame = PILImage.frombytes('RGB', fig.canvas.get_width_height(),
                                    fig.canvas.tostring_rgb())
        frames.append(frame)
        plt.close(fig)
    
    if frames:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"✓ Saved attention GIF to: {save_path}")


def test_visualizations():
    """Test visualization functions with dummy data."""
    print("=" * 70)
    print("TESTING VISUALIZATION UTILITIES")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path('outputs/test_viz')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test training history plot
    print("\n✓ Testing training history plot...")
    history = {
        'train_loss': [5.0, 4.0, 3.2, 2.8, 2.5, 2.3],
        'val_loss': [5.5, 4.5, 3.8, 3.2, 3.0, 2.9],
        'val_bleu': [0.01, 0.05, 0.10, 0.15, 0.18, 0.20],
        'learning_rate': [1e-4, 1e-4, 5e-5, 5e-5, 2.5e-5, 2.5e-5]
    }
    fig = plot_training_history(history, save_path=str(output_dir / 'training_history.png'))
    plt.close(fig)
    print("  - Created training_history.png")
    
    # Test metrics comparison
    print("\n✓ Testing metrics comparison plot...")
    metrics = {
        'CNN+LSTM': {'BLEU-1': 0.65, 'BLEU-4': 0.25, 'METEOR': 0.22, 'CIDEr': 0.45},
        'ViT': {'BLEU-1': 0.68, 'BLEU-4': 0.28, 'METEOR': 0.24, 'CIDEr': 0.50}
    }
    fig = plot_metrics_comparison(metrics, save_path=str(output_dir / 'metrics_comparison.png'))
    plt.close(fig)
    print("  - Created metrics_comparison.png")
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION TEST PASSED")
    print("=" * 70)
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    test_visualizations()
