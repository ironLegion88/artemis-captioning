"""
Second Laptop Training Script
==============================
Training configurations for Intel Ultra 5 125H laptop.
Trains 3 models with ~5000 images for ~30 epochs each.

Usage:
    python scripts/train_second_laptop.py --config 1  # CNN+LSTM Standard
    python scripts/train_second_laptop.py --config 2  # ViT Compact
    python scripts/train_second_laptop.py --config 3  # CNN+LSTM High LR
    python scripts/train_second_laptop.py --all       # Train all sequentially
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
import torch

from train import Trainer
from models.cnn_lstm import create_model as create_cnn_lstm
from models.vision_transformer import create_vit_model
from utils.data_loader import create_data_loaders
from utils.text_preprocessing import TextPreprocessor


# Three configurations optimized for CPU training on ~5000 images
LAPTOP2_CONFIGS = {
    1: {
        "name": "laptop2_cnn_standard",
        "model_type": "cnn_lstm",
        "description": "CNN+LSTM Standard - Baseline for comparison",
        "batch_size": 16,
        "num_images": 5000,
        "epochs": 30,
        "learning_rate": 1e-4,
        "embed_dim": 256,
        "hidden_dim": 512,
    },
    2: {
        "name": "laptop2_vit_compact",
        "model_type": "vit",
        "description": "ViT Compact - Fewer layers, faster training",
        "batch_size": 16,
        "num_images": 5000,
        "epochs": 30,
        "learning_rate": 1e-4,
        "embed_dim": 256,
        "num_layers": 4,
    },
    3: {
        "name": "laptop2_cnn_high_lr",
        "model_type": "cnn_lstm",
        "description": "CNN+LSTM High LR - Faster convergence experiment",
        "batch_size": 24,
        "num_images": 5000,
        "epochs": 25,
        "learning_rate": 5e-4,
        "embed_dim": 256,
        "hidden_dim": 512,
    },
}


class LimitedLoader:
    """Wrapper to limit batches per epoch."""
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


def train_config(config_num):
    """Train a single configuration."""
    if config_num not in LAPTOP2_CONFIGS:
        print(f"Error: Config {config_num} not found. Available: {list(LAPTOP2_CONFIGS.keys())}")
        return None
    
    config = LAPTOP2_CONFIGS[config_num]
    config_name = config['name']
    start_time = datetime.now()
    
    print("=" * 70)
    print(f"SECOND LAPTOP TRAINING: {config_name}")
    print(f"Description: {config['description']}")
    print("=" * 70)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create directories
    checkpoint_dir = f"checkpoints/{config_name}"
    output_dir = f"outputs/{config_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(f"{output_dir}/config.json", "w", encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    # Load vocabulary using TextPreprocessor
    text_proc = TextPreprocessor()
    text_proc.load_vocabulary('data/processed/vocabulary.json')
    print(f"Vocabulary size: {text_proc.vocab_size}")
    
    # Create data loaders
    loaders = create_data_loaders(
        text_preprocessor=text_proc,
        batch_size=config['batch_size'],
        num_workers=0,
        splits=['train', 'val']
    )
    
    # Limit batches to target number of images
    max_train_batches = config['num_images'] // config['batch_size']
    max_val_batches = max(20, max_train_batches // 5)
    
    train_loader = LimitedLoader(loaders['train'], max_train_batches)
    val_loader = LimitedLoader(loaders['val'], max_val_batches)
    
    print(f"Train batches: {len(train_loader)} (~{len(train_loader) * config['batch_size']} images)")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    if config['model_type'] == 'cnn_lstm':
        model = create_cnn_lstm(
            vocab_size=text_proc.vocab_size,
            embed_dim=config['embed_dim']
        )
    else:
        model = create_vit_model(
            vocab_size=text_proc.vocab_size,
            embed_dim=config['embed_dim'],
            encoder_layers=config.get('num_layers', 4),
            decoder_layers=config.get('num_layers', 4)
        )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model_type']}")
    print(f"Parameters: {total_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        text_preprocessor=text_proc,
        learning_rate=config['learning_rate'],
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir
    )
    
    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    history = trainer.train(num_epochs=config['epochs'])
    
    # Save results
    end_time = datetime.now()
    duration = end_time - start_time
    
    results = {
        "config_name": config_name,
        "description": config['description'],
        "model_type": config['model_type'],
        "num_images": config['num_images'],
        "epochs": config['epochs'],
        "parameters": total_params,
        "final_train_loss": history['train_loss'][-1],
        "final_val_loss": history['val_loss'][-1],
        "best_val_loss": min(history['val_loss']),
        "best_bleu": max(history['val_bleu']),
        "duration_minutes": duration.total_seconds() / 60,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }
    
    with open(f"{output_dir}/training_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE: {config_name}")
    print("=" * 70)
    print(f"Duration: {duration}")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"Best BLEU: {results['best_bleu']:.4f}")
    print(f"Results saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Second Laptop Training Script")
    parser.add_argument("--config", type=int, choices=[1, 2, 3],
                       help="Configuration number (1, 2, or 3)")
    parser.add_argument("--all", action="store_true",
                       help="Train all configurations sequentially")
    
    args = parser.parse_args()
    
    if args.all:
        print("=" * 70)
        print("TRAINING ALL CONFIGURATIONS SEQUENTIALLY")
        print("=" * 70)
        
        all_results = {}
        for config_num in [1, 2, 3]:
            try:
                result = train_config(config_num)
                all_results[config_num] = result
            except Exception as e:
                print(f"\n❌ Error training config {config_num}: {e}")
                import traceback
                traceback.print_exc()
                all_results[config_num] = {"error": str(e)}
            
            # Clear memory between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Print summary
        print("\n" + "=" * 70)
        print("ALL TRAINING COMPLETE - SUMMARY")
        print("=" * 70)
        
        for config_num, result in all_results.items():
            config = LAPTOP2_CONFIGS[config_num]
            if "error" in result:
                print(f"\n{config['name']}: FAILED - {result['error']}")
            else:
                print(f"\n{config['name']}: {config['description']}")
                print(f"  Duration: {result['duration_minutes']:.1f} minutes")
                print(f"  Best Val Loss: {result['best_val_loss']:.4f}")
                print(f"  Best BLEU: {result['best_bleu']:.4f}")
    
    elif args.config:
        train_config(args.config)
    
    else:
        parser.print_help()
        print("\nAvailable configurations:")
        for num, cfg in LAPTOP2_CONFIGS.items():
            print(f"  {num}: {cfg['name']} - {cfg['description']}")


if __name__ == "__main__":
    main()
