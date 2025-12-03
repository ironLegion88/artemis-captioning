"""
ArtEmis Image Captioning Project

A deep learning project for generating emotional captions for artworks
using CNN+LSTM and Vision Transformer models.

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import argparse
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent


def train(args):
    """Run training pipeline."""
    from train import main as train_main
    train_main()


def predict(args):
    """Run inference on images."""
    from scripts.predict import main as predict_main
    sys.argv = ['predict.py']
    if args.image:
        sys.argv.extend(['--image', args.image])
    if args.model:
        sys.argv.extend(['--model', args.model])
    if args.model_type:
        sys.argv.extend(['--model-type', args.model_type])
    predict_main()


def evaluate(args):
    """Run evaluation on test set."""
    from utils.evaluation import test_evaluator
    test_evaluator()


def experiment(args):
    """Run experiments."""
    from scripts.run_experiments import main as exp_main
    sys.argv = ['run_experiments.py', '--mode', args.mode]
    if args.epochs:
        sys.argv.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        sys.argv.extend(['--batch-size', str(args.batch_size)])
    if args.max_batches:
        sys.argv.extend(['--max-batches', str(args.max_batches)])
    exp_main()


def analyze(args):
    """Run dataset analysis."""
    from scripts.analyze_dataset import main as analyze_main
    analyze_main()


def preprocess(args):
    """Run data preprocessing."""
    from scripts.create_splits import main as splits_main
    splits_main()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ArtEmis Image Captioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Train CNN+LSTM model
  python main.py predict --image img.jpg  # Generate caption for image
  python main.py experiment --mode quick  # Run quick experiment
  python main.py evaluate                 # Evaluate on test set
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the captioning model')
    train_parser.set_defaults(func=train)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate captions for images')
    predict_parser.add_argument('--image', type=str, help='Path to image file')
    predict_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    predict_parser.add_argument('--model-type', type=str, choices=['cnn_lstm', 'vit'],
                                default='cnn_lstm', help='Model type')
    predict_parser.set_defaults(func=predict)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on test set')
    eval_parser.set_defaults(func=evaluate)
    
    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run training experiments')
    exp_parser.add_argument('--mode', type=str, default='quick',
                           choices=['quick', 'full', 'cnn', 'vit'],
                           help='Experiment mode')
    exp_parser.add_argument('--epochs', type=int, help='Number of epochs')
    exp_parser.add_argument('--batch-size', type=int, help='Batch size')
    exp_parser.add_argument('--max-batches', type=int, help='Limit batches (for testing)')
    exp_parser.set_defaults(func=experiment)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset')
    analyze_parser.set_defaults(func=analyze)
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.set_defaults(func=preprocess)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\n" + "=" * 60)
        print("ARTEMIS IMAGE CAPTIONING")
        print("=" * 60)
        print("\nProject components:")
        print("  - CNN+LSTM model (ResNet18 encoder + LSTM decoder)")
        print("  - Vision Transformer model (ViT encoder + Transformer decoder)")
        print("  - BLEU, METEOR, ROUGE-L, CIDEr evaluation metrics")
        print("  - Training pipeline with early stopping")
        print("  - Inference with beam search decoding")
        print("\nUse 'python main.py <command> --help' for more info.")
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
