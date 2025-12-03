"""
Experiment Runner for Model Comparison

This module provides utilities for:
- Running training experiments with different configurations
- Comparing CNN+LSTM vs Vision Transformer models
- Logging and tracking experiment results

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import Trainer
from utils.data_loader import create_data_loaders
from utils.text_preprocessing import TextPreprocessor
from models.cnn_lstm import create_model as create_cnn_lstm_model
from models.vision_transformer import create_vit_model
from utils.visualization import plot_training_history, plot_metrics_comparison
from utils.constants import (
    PROCESSED_DATA_DIR,
    CHECKPOINTS_DIR,
    VOCAB_SIZE,
    VOCABULARY_PATH
)


class LimitedLoader:
    """Wrapper to limit number of batches in a data loader."""
    
    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = max_batches
        self.batch_size = loader.batch_size
        self.dataset = loader.dataset
    
    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                break
            yield batch
    
    def __len__(self):
        return min(len(self.loader), self.max_batches)


class ExperimentRunner:
    """
    Run and compare different model experiments.
    """
    
    def __init__(
        self,
        output_dir: str = 'outputs/experiments',
        device: str = 'cpu'
    ):
        """
        Initialize the experiment runner.
        
        Args:
            output_dir: Directory to save experiment results
            device: Device to use (cpu/cuda)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.results = {}
        
        # Experiment timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize text preprocessor
        print("\n✓ Loading vocabulary...")
        self.text_preprocessor = TextPreprocessor()
        self.text_preprocessor.load_vocabulary(VOCABULARY_PATH)
        print(f"  - Vocabulary size: {self.text_preprocessor.vocab_size}")
    
    def run_experiment(
        self,
        name: str,
        model_type: str,
        config: Dict[str, Any],
        max_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a single experiment.
        
        Args:
            name: Experiment name
            model_type: 'cnn_lstm' or 'vit'
            config: Training configuration
            max_batches: Limit batches per epoch (for testing)
        
        Returns:
            Dictionary with experiment results
        """
        print("=" * 70)
        print(f"EXPERIMENT: {name}")
        print(f"Model: {model_type.upper()}")
        print("=" * 70)
        
        experiment_dir = self.output_dir / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        print("\n✓ Creating model...")
        if model_type == 'cnn_lstm':
            model = create_cnn_lstm_model(vocab_size=VOCAB_SIZE)
        elif model_type == 'vit':
            model = create_vit_model(vocab_size=VOCAB_SIZE)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Create data loaders
        print("\n✓ Creating data loaders...")
        data_loaders = create_data_loaders(
            text_preprocessor=self.text_preprocessor,
            batch_size=config['batch_size'],
            num_workers=0,
            splits=['train', 'val']
        )
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        # Apply batch limit if specified
        if max_batches is not None:
            print(f"  - Limiting to {max_batches} batches per epoch")
            train_loader = LimitedLoader(train_loader, max_batches)
            val_loader = LimitedLoader(val_loader, max_batches)
        
        # Create trainer
        checkpoint_dir = experiment_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        output_dir = experiment_dir / 'outputs'
        output_dir.mkdir(exist_ok=True)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            text_preprocessor=self.text_preprocessor,
            learning_rate=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
            device=self.device,
            checkpoint_dir=str(checkpoint_dir),
            output_dir=str(output_dir)
        )
        
        # Train
        print("\n✓ Starting training...")
        start_time = datetime.now()
        
        try:
            history = trainer.train(num_epochs=config.get('epochs', 5))
            training_success = True
        except Exception as e:
            print(f"\n⚠ Training failed: {e}")
            import traceback
            traceback.print_exc()
            history = {}
            training_success = False
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Compile results
        results = {
            'name': name,
            'model_type': model_type,
            'config': config,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'training_time_seconds': training_time,
            'training_success': training_success,
            'history': history,
            'final_metrics': {
                'train_loss': history.get('train_loss', [0])[-1] if history else 0,
                'val_loss': history.get('val_loss', [0])[-1] if history else 0,
                'val_bleu': history.get('val_bleu', [0])[-1] if history else 0,
            }
        }
        
        # Save results
        results_path = experiment_dir / 'results.json'
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        
        # Save training history plot
        if history:
            plot_path = experiment_dir / 'training_history.png'
            try:
                plot_training_history(history, save_path=str(plot_path))
            except Exception as e:
                print(f"  - Could not save plot: {e}")
        
        self.results[name] = results
        
        print(f"\n✓ Experiment complete: {name}")
        print(f"  - Training time: {training_time:.1f}s")
        print(f"  - Final val loss: {results['final_metrics']['val_loss']:.4f}")
        print(f"  - Final val BLEU: {results['final_metrics']['val_bleu']:.4f}")
        print(f"  - Results saved to: {experiment_dir}")
        
        return results
    
    def _convert_for_json(self, obj):
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def run_comparison(
        self,
        epochs: int = 5,
        batch_size: int = 16,
        max_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run CNN+LSTM vs ViT comparison experiment.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            max_batches: Limit batches per epoch (for quick testing)
        
        Returns:
            Comparison results
        """
        print("\n" + "=" * 70)
        print("MODEL COMPARISON EXPERIMENT")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        if max_batches:
            print(f"  - Max batches: {max_batches}")
        
        config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
        }
        
        # Run CNN+LSTM experiment
        cnn_results = self.run_experiment(
            name=f'cnn_lstm_{self.timestamp}',
            model_type='cnn_lstm',
            config=config,
            max_batches=max_batches
        )
        
        # Run ViT experiment
        vit_results = self.run_experiment(
            name=f'vit_{self.timestamp}',
            model_type='vit',
            config=config,
            max_batches=max_batches
        )
        
        # Create comparison
        comparison = {
            'timestamp': self.timestamp,
            'config': config,
            'experiments': {
                'cnn_lstm': self._convert_for_json(cnn_results),
                'vit': self._convert_for_json(vit_results)
            },
            'comparison': {
                'params': {
                    'CNN+LSTM': cnn_results['total_params'],
                    'ViT': vit_results['total_params']
                },
                'final_val_loss': {
                    'CNN+LSTM': cnn_results['final_metrics']['val_loss'],
                    'ViT': vit_results['final_metrics']['val_loss']
                },
                'final_val_bleu': {
                    'CNN+LSTM': cnn_results['final_metrics']['val_bleu'],
                    'ViT': vit_results['final_metrics']['val_bleu']
                },
                'training_time': {
                    'CNN+LSTM': cnn_results['training_time_seconds'],
                    'ViT': vit_results['training_time_seconds']
                }
            }
        }
        
        # Save comparison
        comparison_path = self.output_dir / f'comparison_{self.timestamp}.json'
        with open(comparison_path, 'w') as f:
            json.dump(self._convert_for_json(comparison), f, indent=2)
        
        # Create comparison plot
        if cnn_results['training_success'] and vit_results['training_success']:
            metrics = {
                'CNN+LSTM': cnn_results['final_metrics'],
                'ViT': vit_results['final_metrics']
            }
            plot_path = self.output_dir / f'comparison_{self.timestamp}.png'
            try:
                plot_metrics_comparison(metrics, save_path=str(plot_path))
            except Exception as e:
                print(f"  - Could not save comparison plot: {e}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print("\nModel Parameters:")
        print(f"  - CNN+LSTM: {cnn_results['total_params']:,}")
        print(f"  - ViT:      {vit_results['total_params']:,}")
        
        print("\nFinal Validation Loss:")
        print(f"  - CNN+LSTM: {cnn_results['final_metrics']['val_loss']:.4f}")
        print(f"  - ViT:      {vit_results['final_metrics']['val_loss']:.4f}")
        
        print("\nFinal Validation BLEU:")
        print(f"  - CNN+LSTM: {cnn_results['final_metrics']['val_bleu']:.4f}")
        print(f"  - ViT:      {vit_results['final_metrics']['val_bleu']:.4f}")
        
        print("\nTraining Time:")
        print(f"  - CNN+LSTM: {cnn_results['training_time_seconds']:.1f}s")
        print(f"  - ViT:      {vit_results['training_time_seconds']:.1f}s")
        
        print(f"\nComparison saved to: {comparison_path}")
        
        return comparison


def quick_test():
    """Run a quick test experiment with minimal data."""
    print("=" * 70)
    print("QUICK TEST EXPERIMENT")
    print("=" * 70)
    print("Running minimal test with 10 batches, 2 epochs...")
    
    runner = ExperimentRunner(output_dir='outputs/experiments/test')
    
    results = runner.run_comparison(
        epochs=2,
        batch_size=8,
        max_batches=10
    )
    
    print("\n✅ Quick test completed!")
    return results


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description='Run image captioning experiments')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'cnn', 'vit'],
                       help='Experiment mode')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Limit batches per epoch (for testing)')
    parser.add_argument('--output-dir', type=str, default='outputs/experiments',
                       help='Output directory')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
    }
    
    if args.mode == 'quick':
        quick_test()
    
    elif args.mode == 'full':
        runner.run_comparison(
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_batches=args.max_batches
        )
    
    elif args.mode == 'cnn':
        runner.run_experiment(
            name=f'cnn_lstm_{runner.timestamp}',
            model_type='cnn_lstm',
            config=config,
            max_batches=args.max_batches
        )
    
    elif args.mode == 'vit':
        runner.run_experiment(
            name=f'vit_{runner.timestamp}',
            model_type='vit',
            config=config,
            max_batches=args.max_batches
        )


if __name__ == "__main__":
    main()
