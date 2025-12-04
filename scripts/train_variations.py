"""
Primary Laptop Training Script - Variations
=============================================
Training configurations for primary laptop (i5-1235U).
Trains models with ~3000 images with various hyperparameter experiments.

Usage:
    python scripts/train_variations.py --config 1  # CNN+LSTM Word2Vec
    python scripts/train_variations.py --config 2  # CNN+LSTM Low Dropout
    python scripts/train_variations.py --config 3  # ViT Small Patches
    python scripts/train_variations.py --config 4  # Quick CNN Test (15 epochs)
    python scripts/train_variations.py --config 5  # Quick ViT Test (15 epochs)
    python scripts/train_variations.py --list      # List all configs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
import time
import numpy as np
import torch

from utils.text_preprocessing import TextPreprocessor
from utils.data_loader import create_data_loaders
from train import Trainer


VARIATION_CONFIGS = {
    1: {
        "name": "primary_cnn_word2vec",
        "model_type": "cnn_lstm",
        "description": "CNN+LSTM with Word2Vec Embeddings",
        "batch_size": 16,
        "num_images": 3000,
        "epochs": 25,
        "learning_rate": 5e-5,
        "embed_dim": 300,
        "hidden_dim": 512,
        "attention_dim": 256,
        "dropout": 0.3,
        "encoder_lr_factor": 0.1,
        "embedding_type": "word2vec",
    },
    2: {
        "name": "primary_cnn_low_dropout",
        "model_type": "cnn_lstm",
        "description": "CNN+LSTM Low Dropout - Less regularization",
        "batch_size": 16,
        "num_images": 3000,
        "epochs": 25,
        "learning_rate": 1e-4,
        "embed_dim": 256,
        "hidden_dim": 512,
        "attention_dim": 256,
        "dropout": 0.2,
        "encoder_lr_factor": 0.1,
    },
    3: {
        "name": "primary_vit_small_patch",
        "model_type": "vit",
        "description": "ViT with Smaller Patches (8x8) - More tokens",
        "batch_size": 8,
        "num_images": 3000,
        "epochs": 20,
        "learning_rate": 5e-5,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 4,
        "ff_dim": 512,
        "dropout": 0.1,
        "patch_size": 8,
    },
    4: {
        "name": "quick_cnn_test",
        "model_type": "cnn_lstm",
        "description": "Quick CNN Test - 15 epochs for fast iteration",
        "batch_size": 16,
        "num_images": 3000,
        "epochs": 15,
        "learning_rate": 2e-4,
        "embed_dim": 256,
        "hidden_dim": 512,
        "attention_dim": 256,
        "dropout": 0.3,
        "encoder_lr_factor": 0.1,
    },
    5: {
        "name": "quick_vit_test",
        "model_type": "vit",
        "description": "Quick ViT Test - 15 epochs for fast iteration",
        "batch_size": 16,
        "num_images": 3000,
        "epochs": 15,
        "learning_rate": 2e-4,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 4,
        "ff_dim": 512,
        "dropout": 0.1,
    },
}


class LimitedLoader:
    """Wrapper to limit batches per epoch."""
    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = max_batches
    
    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                break
            yield batch
    
    def __len__(self):
        return min(len(self.loader), self.max_batches)


def load_embeddings(embedding_type):
    """Load pretrained embeddings."""
    path = f"data/embeddings/{embedding_type}_embeddings.npy"
    if os.path.exists(path):
        embeddings = np.load(path)
        print(f"  ✓ Loaded {embedding_type} embeddings: {embeddings.shape}")
        return torch.tensor(embeddings, dtype=torch.float32)
    else:
        print(f"  ⚠ {embedding_type} embeddings not found, using random init")
        return None


def train_config(config_num):
    """Train a single configuration."""
    if config_num not in VARIATION_CONFIGS:
        print(f"Error: Config {config_num} not found. Available: {list(VARIATION_CONFIGS.keys())}")
        return None
    
    config = VARIATION_CONFIGS[config_num]
    config_name = config['name']
    
    print("=" * 70)
    print(f"PRIMARY LAPTOP TRAINING: {config_name}")
    print(f"Description: {config['description']}")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    checkpoint_dir = f"checkpoints/{config_name}"
    output_dir = f"outputs/{config_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load vocabulary using TextPreprocessor
    vocab_path = "data/processed/vocabulary.json"
    text_proc = TextPreprocessor()
    text_proc.load_vocabulary(vocab_path)
    vocab_size = text_proc.vocab_size
    print(f"\n✓ Vocabulary loaded from: {vocab_path}")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Max caption length: {text_proc.max_length}")
    
    # Create data loaders using correct API
    print("\n" + "=" * 70)
    print("CREATING DATA LOADERS")
    print("=" * 70)
    
    loaders = create_data_loaders(
        text_preprocessor=text_proc,
        batch_size=config['batch_size'],
        num_workers=0,
        splits=['train', 'val']
    )
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    # Limit batches for faster training
    max_train_batches = config['num_images'] // config['batch_size']
    max_val_batches = max(20, max_train_batches // 5)
    
    train_loader = LimitedLoader(train_loader, max_train_batches)
    val_loader = LimitedLoader(val_loader, max_val_batches)
    
    print(f"Train batches: {len(train_loader)} (~{len(train_loader) * config['batch_size']} images)")
    print(f"Val batches: {len(val_loader)} (~{len(val_loader) * config['batch_size']} images)")
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    if config['model_type'] == 'cnn_lstm':
        from models.cnn_lstm import ImageCaptioningModel
        model = ImageCaptioningModel(
            embed_dim=config['embed_dim'],
            attention_dim=config['attention_dim'],
            decoder_dim=config['hidden_dim'],
            vocab_size=vocab_size,
            encoder_dim=2048,
            dropout=config['dropout'],
            pretrained_encoder=True
        )
        
        # Load pretrained embeddings if specified
        embedding_type = config.get('embedding_type')
        if embedding_type:
            embeddings = load_embeddings(embedding_type)
            if embeddings is not None:
                model.decoder.embedding.weight.data.copy_(embeddings)
    else:
        from models.vision_transformer import VisionTransformerCaptioning
        patch_size = config.get('patch_size', 16)
        model = VisionTransformerCaptioning(
            vocab_size=vocab_size,
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_layers'],
            num_decoder_layers=config['num_layers'],
            ff_dim=config['ff_dim'],
            max_seq_len=30,
            dropout=config['dropout'],
            img_size=128,
            patch_size=patch_size
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model_type'].upper()}")
    print(f"Device: {device}")
    print(f"Parameters: {total_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        text_preprocessor=text_proc,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir
    )
    
    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    start_time = time.time()
    history = trainer.train(
        epochs=config['epochs'],
        learning_rate=config['learning_rate']
    )
    duration = time.time() - start_time
    
    # Save results
    results = {
        "config_name": config_name,
        "description": config['description'],
        "model_type": config['model_type'],
        "num_images": config['num_images'],
        "epochs": config['epochs'],
        "parameters": total_params,
        "duration_minutes": duration / 60,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE: {config_name}")
    print("=" * 70)
    print(f"Duration: {duration/60:.1f} minutes ({duration/3600:.2f} hours)")
    print(f"Results saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Primary Laptop Training - Variations")
    parser.add_argument("--config", type=int, help="Config number (1-5)")
    parser.add_argument("--list", action="store_true", help="List available configs")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Configurations for Primary Laptop:")
        print("=" * 70)
        
        print("\n--- Full Training Runs ---")
        for num in [1, 2, 3]:
            cfg = VARIATION_CONFIGS[num]
            est_time = cfg['epochs'] * 5  # ~5 min per epoch
            print(f"\n  --config {num}: {cfg['name']}")
            print(f"      {cfg['description']}")
            print(f"      Model: {cfg['model_type']}, Epochs: {cfg['epochs']}, Est: ~{est_time} min")
        
        print("\n--- Quick Test Runs ---")
        for num in [4, 5]:
            cfg = VARIATION_CONFIGS[num]
            est_time = cfg['epochs'] * 5
            print(f"\n  --config {num}: {cfg['name']}")
            print(f"      {cfg['description']}")
            print(f"      Model: {cfg['model_type']}, Epochs: {cfg['epochs']}, Est: ~{est_time} min")
        
        print("\nUsage:")
        print("  python scripts/train_variations.py --config 4  # Quick test first")
        print("  python scripts/train_variations.py --config 1  # Full Word2Vec run")
        return
    
    if args.config:
        train_config(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
