"""
Batch Evaluation Script for All Trained Models

This script:
1. Loads all trained model checkpoints
2. Generates captions on the test set
3. Computes comprehensive metrics (BLEU-1,2,3,4, ROUGE-L, METEOR, CIDEr)
4. Creates visualizations of generated captions
5. Saves consolidated results

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    CHECKPOINTS_DIR, PROCESSED_DIR, OUTPUTS_DIR, DEVICE
)
from utils.evaluation import CaptionEvaluator, compute_bleu_score, compute_rouge_l, compute_meteor, compute_cider_score
from scripts.predict import CaptionGenerator


def get_all_checkpoints() -> Dict[str, Dict]:
    """
    Find all available model checkpoints.
    
    Returns:
        Dictionary mapping experiment name to checkpoint info
    """
    checkpoints = {}
    
    if not CHECKPOINTS_DIR.exists():
        return checkpoints
    
    for exp_dir in CHECKPOINTS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        
        best_model = exp_dir / 'best_model.pth'
        if best_model.exists():
            # Determine model type from name
            model_type = 'vit' if 'vit' in exp_dir.name.lower() else 'cnn_lstm'
            
            # Try to load config
            config_path = OUTPUTS_DIR / exp_dir.name / 'config.json'
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            
            # Try to load training history for BLEU score
            history_path = OUTPUTS_DIR / exp_dir.name / 'training_history.json'
            best_bleu = None
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
                    if 'val_bleu' in history and history['val_bleu']:
                        best_bleu = max(history['val_bleu'])
            
            checkpoints[exp_dir.name] = {
                'path': str(best_model),
                'model_type': model_type,
                'config': config,
                'best_bleu': best_bleu,
                'size_mb': best_model.stat().st_size / (1024 * 1024)
            }
    
    return checkpoints


def load_test_data(num_samples: Optional[int] = None) -> Tuple[List[Dict], Dict]:
    """
    Load test set data.
    
    Args:
        num_samples: Optional limit on number of samples
        
    Returns:
        List of test samples and vocabulary
    """
    # Load vocabulary
    vocab_path = PROCESSED_DIR / 'vocabulary.json'
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    # Try different possible locations for test split
    possible_paths = [
        PROCESSED_DIR / 'splits' / 'test.json',
        PROCESSED_DIR / 'splits_nfd' / 'test.json',
        PROCESSED_DIR / 'splits_colab' / 'test.json',
        PROCESSED_DIR / 'test_split.json',
        PROCESSED_DIR / 'test_split_nfd.json',
    ]
    
    test_data = None
    for test_path in possible_paths:
        if test_path.exists():
            with open(test_path, encoding='utf-8') as f:
                raw_data = json.load(f)
            print(f"   Loaded test data from: {test_path}")
            
            # Handle different formats
            if isinstance(raw_data, list):
                test_data = raw_data
            elif isinstance(raw_data, dict) and 'paintings' in raw_data:
                # Format: {'paintings': [...], 'split': 'test', ...}
                paintings = raw_data['paintings']
                test_data = []
                for p in paintings:
                    # Expand each painting with its captions
                    for i, caption in enumerate(p.get('captions', [])):
                        test_data.append({
                            'painting': p['painting'],
                            'art_style': p.get('style', 'unknown'),
                            'utterance': caption,
                            'utterance_list': p.get('captions', [caption]),
                            'emotion': p.get('emotions', ['unknown'])[i] if i < len(p.get('emotions', [])) else 'unknown',
                            'image_path': str(PROCESSED_DIR / 'images' / p.get('style', 'unknown').replace('_', ' ') / f"{p['painting']}.jpg")
                        })
            else:
                test_data = list(raw_data.values()) if isinstance(raw_data, dict) else []
            break
    
    if test_data is None or len(test_data) == 0:
        raise FileNotFoundError(f"No test split found or empty. Tried: {possible_paths}")
    
    if num_samples:
        test_data = test_data[:num_samples]
    
    print(f"   Total test samples: {len(test_data)}")
    return test_data, vocab


def evaluate_model(
    generator: CaptionGenerator,
    test_data: List[Dict],
    vocab: Dict,
    num_samples: Optional[int] = None
) -> Dict:
    """
    Evaluate a model on test data.
    
    Args:
        generator: Caption generator
        test_data: List of test samples
        vocab: Vocabulary dictionary
        num_samples: Optional limit
        
    Returns:
        Dictionary of metrics and predictions
    """
    evaluator = CaptionEvaluator(tokenize=True)
    
    all_references = []
    all_hypotheses = []
    predictions = []
    
    samples = test_data[:num_samples] if num_samples else test_data
    
    for sample in tqdm(samples, desc="Evaluating"):
        image_path = sample.get('image_path', '')
        
        # Find image
        if not os.path.exists(image_path):
            # Try relative path
            full_path = PROCESSED_DIR / 'images' / sample.get('art_style', '') / sample.get('painting', '')
            if not full_path.exists():
                continue
            image_path = str(full_path)
        
        # Get reference captions
        refs = sample.get('utterance_list', [sample.get('utterance', '')])
        if isinstance(refs, str):
            refs = [refs]
        refs = [r for r in refs if r.strip()]
        
        if not refs:
            continue
        
        # Generate caption
        try:
            caption, _ = generator.generate_caption(image_path)
        except Exception as e:
            print(f"  Warning: Failed to generate caption for {image_path}: {e}")
            continue
        
        all_references.append(refs)
        all_hypotheses.append(caption)
        
        predictions.append({
            'image_path': image_path,
            'references': refs,
            'hypothesis': caption,
            'emotion': sample.get('emotion', 'unknown'),
            'art_style': sample.get('art_style', 'unknown')
        })
    
    if not all_hypotheses:
        return {'error': 'No valid predictions generated'}
    
    # Compute corpus-level metrics
    corpus_metrics = evaluator.evaluate_corpus(all_references, all_hypotheses)
    
    # Compute per-sample metrics
    sample_metrics = []
    for refs, hyp in zip(all_references, all_hypotheses):
        m = evaluator.evaluate_single(refs, hyp)
        sample_metrics.append(m)
    
    return {
        'corpus_metrics': corpus_metrics,
        'sample_metrics': sample_metrics,
        'predictions': predictions,
        'num_samples': len(predictions)
    }


def visualize_captions(
    predictions: List[Dict],
    output_path: Path,
    num_samples: int = 12,
    cols: int = 3
) -> str:
    """
    Create visualization grid of images with generated captions.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save visualization
        num_samples: Number of samples to show
        cols: Number of columns in grid
        
    Returns:
        Path to saved visualization
    """
    import textwrap
    
    samples = predictions[:num_samples]
    n = len(samples)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, sample in enumerate(samples):
        row, col = idx // cols, idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        
        # Load and display image
        try:
            img = Image.open(sample['image_path'])
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
        
        ax.axis('off')
        
        # Prepare caption text
        hyp = sample['hypothesis']
        ref = sample['references'][0] if sample['references'] else ""
        emotion = sample.get('emotion', '')
        
        # Wrap long captions
        hyp_wrapped = "\n".join(textwrap.wrap(f"Generated: {hyp}", 40))
        ref_wrapped = "\n".join(textwrap.wrap(f"Reference: {ref[:100]}...", 40)) if len(ref) > 100 else f"Reference: {ref}"
        
        title = f"[{emotion}]\n{hyp_wrapped}"
        ax.set_title(title, fontsize=9, wrap=True)
    
    # Hide empty subplots
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def visualize_single_prediction(
    image_path: str,
    captions: List[Tuple[str, float]],
    emotion: Optional[str] = None,
    reference: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = True
) -> Optional[str]:
    """
    Visualize a single image with its generated captions.
    
    Args:
        image_path: Path to the image
        captions: List of (caption, score) tuples
        emotion: Optional emotion label
        reference: Optional reference caption
        output_path: Optional path to save figure
        show: Whether to display the figure
        
    Returns:
        Path to saved figure if output_path provided
    """
    import textwrap
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Load and display image
    try:
        img = Image.open(image_path)
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not load image:\n{e}", ha='center', va='center')
    
    ax.axis('off')
    
    # Build caption text
    caption_lines = []
    if emotion:
        caption_lines.append(f"🎭 Emotion: {emotion}")
    caption_lines.append("")
    caption_lines.append("📝 Generated Captions:")
    
    for i, (cap, score) in enumerate(captions, 1):
        wrapped = textwrap.fill(f"{i}. {cap}", 60)
        caption_lines.append(f"{wrapped}")
        caption_lines.append(f"   (Score: {score:.4f})")
    
    if reference:
        caption_lines.append("")
        caption_lines.append("📖 Reference:")
        ref_wrapped = textwrap.fill(reference, 60)
        caption_lines.append(ref_wrapped)
    
    # Add text below image
    caption_text = "\n".join(caption_lines)
    
    # Get image name for title
    img_name = Path(image_path).stem
    ax.set_title(f"Image: {img_name}", fontsize=12, fontweight='bold')
    
    # Add caption as figure text
    fig.text(0.5, 0.02, caption_text, ha='center', va='bottom', fontsize=10,
             fontfamily='monospace', wrap=True,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path


def create_metrics_comparison_chart(
    all_results: Dict[str, Dict],
    output_path: Path
) -> str:
    """
    Create bar chart comparing metrics across models.
    
    Args:
        all_results: Dictionary of results per model
        output_path: Path to save chart
        
    Returns:
        Path to saved chart
    """
    models = list(all_results.keys())
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L', 'METEOR', 'CIDEr']
    
    # Filter to models with valid results
    valid_models = [m for m in models if 'corpus_metrics' in all_results[m]]
    
    if not valid_models:
        print("No valid results to plot")
        return ""
    
    x = np.arange(len(metrics))
    width = 0.8 / len(valid_models)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, model in enumerate(valid_models):
        values = [all_results[model]['corpus_metrics'].get(m, 0) for m in metrics]
        offset = (i - len(valid_models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Caption Generation Metrics Comparison Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, max(0.1, ax.get_ylim()[1] * 1.2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def run_batch_evaluation(
    models: Optional[List[str]] = None,
    num_samples: int = 100,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run batch evaluation on all models.
    
    Args:
        models: Optional list of model names to evaluate
        num_samples: Number of test samples to use
        output_dir: Output directory for results
        
    Returns:
        Dictionary with all results
    """
    if output_dir is None:
        output_dir = OUTPUTS_DIR / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all checkpoints
    all_checkpoints = get_all_checkpoints()
    
    if models:
        checkpoints = {k: v for k, v in all_checkpoints.items() if k in models}
    else:
        checkpoints = all_checkpoints
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return {}
    
    print(f"\n{'=' * 70}")
    print("BATCH MODEL EVALUATION")
    print(f"{'=' * 70}")
    print(f"Models to evaluate: {len(checkpoints)}")
    print(f"Test samples: {num_samples}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 70}\n")
    
    # Load test data
    print("📂 Loading test data...")
    test_data, vocab = load_test_data(num_samples)
    print(f"   Loaded {len(test_data)} test samples")
    
    # Evaluate each model
    all_results = {}
    
    for model_name, info in checkpoints.items():
        print(f"\n{'─' * 50}")
        print(f"🔄 Evaluating: {model_name}")
        print(f"   Type: {info['model_type']}")
        print(f"   Checkpoint: {info['path']}")
        
        try:
            # Load model
            generator = CaptionGenerator(
                model_type=info['model_type'],
                checkpoint_path=info['path'],
                device=DEVICE
            )
            
            # Evaluate
            results = evaluate_model(generator, test_data, vocab, num_samples)
            all_results[model_name] = results
            
            if 'corpus_metrics' in results:
                print(f"\n   📊 Results:")
                for metric, value in results['corpus_metrics'].items():
                    print(f"      {metric}: {value:.4f}")
                
                # Create caption visualization for this model
                if results['predictions']:
                    viz_path = output_dir / f'{model_name}_captions.png'
                    visualize_captions(results['predictions'], viz_path, num_samples=9)
                    print(f"   📸 Saved visualization: {viz_path}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # Save consolidated results
    print(f"\n{'=' * 70}")
    print("💾 Saving results...")
    
    # JSON results (without predictions for size)
    json_results = {}
    for model_name, results in all_results.items():
        if 'corpus_metrics' in results:
            json_results[model_name] = {
                'corpus_metrics': results['corpus_metrics'],
                'num_samples': results['num_samples']
            }
        else:
            json_results[model_name] = results
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"   📄 Results JSON: {results_path}")
    
    # Metrics comparison chart
    chart_path = output_dir / 'metrics_comparison.png'
    create_metrics_comparison_chart(all_results, chart_path)
    print(f"   📊 Comparison chart: {chart_path}")
    
    # Summary table
    print(f"\n{'=' * 70}")
    print("📋 EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<30} {'BLEU-4':>10} {'ROUGE-L':>10} {'METEOR':>10} {'CIDEr':>10}")
    print("─" * 70)
    
    for model_name, results in sorted(all_results.items(), 
                                       key=lambda x: x[1].get('corpus_metrics', {}).get('BLEU-4', 0),
                                       reverse=True):
        if 'corpus_metrics' in results:
            m = results['corpus_metrics']
            print(f"{model_name:<30} {m.get('BLEU-4', 0):>10.4f} {m.get('ROUGE-L', 0):>10.4f} "
                  f"{m.get('METEOR', 0):>10.4f} {m.get('CIDEr', 0):>10.4f}")
        else:
            print(f"{model_name:<30} {'ERROR':>10}")
    
    print(f"{'=' * 70}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation of Caption Models")
    parser.add_argument("--models", nargs="+", help="Specific models to evaluate")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of test samples (default: 100)")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    if args.list_models:
        checkpoints = get_all_checkpoints()
        print("\n📦 Available Models:")
        print("─" * 50)
        for name, info in sorted(checkpoints.items()):
            bleu = f"BLEU: {info['best_bleu']:.4f}" if info['best_bleu'] else "BLEU: N/A"
            print(f"  {name:<30} ({info['model_type']}) - {bleu}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    run_batch_evaluation(args.models, args.num_samples, output_dir)


if __name__ == "__main__":
    main()
