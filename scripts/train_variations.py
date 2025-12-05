"""
Primary Laptop Training Script - Variations
=============================================
Training configurations for primary laptop (i5-1235U).
Trains models with ~3000 images with various hyperparameter experiments.

Usage:
    python scripts/train_variations.py --config 1  # CNN+LSTM Word2Vec
    python scripts/train_variations.py --config 2  # CNN+LSTM High LR
    python scripts/train_variations.py --config 3  # CNN+LSTM Very High LR
    python scripts/train_variations.py --config 4  # ViT Higher LR
    python scripts/train_variations.py --config 5  # CNN Large Batch
    python scripts/train_variations.py --config 6  # ViT Regularized (NEW)
    python scripts/train_variations.py --config 7  # ViT Deep Regularized (NEW)
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


# OPTIMIZED based on training analysis (laptop2_cnn_high_lr achieved BLEU 0.0152 with 5e-4 LR)
VARIATION_CONFIGS = {
    1: {
        "name": "primary_cnn_word2vec",
        "model_type": "cnn_lstm",
        "description": "CNN+LSTM with Word2Vec Embeddings (COMPLETED)",
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
        "name": "primary_cnn_high_lr",
        "model_type": "cnn_lstm",
        "description": "CNN+LSTM High LR - Best config from analysis (0.0152 BLEU)",
        "batch_size": 24,
        "num_images": 3000,
        "epochs": 15,  # BLEU peaks around epoch 10
        "learning_rate": 3e-4,  # Higher LR works best
        "embed_dim": 256,
        "hidden_dim": 512,
        "attention_dim": 256,
        "dropout": 0.3,
        "encoder_lr_factor": 0.1,
    },
    3: {
        "name": "primary_cnn_very_high_lr",
        "model_type": "cnn_lstm",
        "description": "CNN+LSTM Very High LR - Testing 5e-4 like laptop2",
        "batch_size": 24,
        "num_images": 3000,
        "epochs": 15,
        "learning_rate": 5e-4,  # Matched laptop2_cnn_high_lr exactly
        "embed_dim": 256,
        "hidden_dim": 512,
        "attention_dim": 256,
        "dropout": 0.3,
        "encoder_lr_factor": 0.1,
    },
    4: {
        "name": "primary_vit_higher_lr",
        "model_type": "vit",
        "description": "ViT with Higher LR - Testing 2e-4",
        "batch_size": 16,
        "num_images": 3000,
        "epochs": 15,  # ViT BLEU peaks early (epoch 6)
        "learning_rate": 2e-4,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 4,
        "ff_dim": 512,
        "dropout": 0.1,
    },
    5: {
        "name": "primary_cnn_large_batch",
        "model_type": "cnn_lstm",
        "description": "CNN+LSTM Larger Batch - Testing batch_size 32",
        "batch_size": 32,
        "num_images": 3000,
        "epochs": 15,
        "learning_rate": 3e-4,
        "embed_dim": 256,
        "hidden_dim": 512,
        "attention_dim": 256,
        "dropout": 0.3,
        "encoder_lr_factor": 0.1,
    },
    # NEW ViT CONFIGS based on analysis:
    # - ViT BLEU peaks early (epoch 6) then degrades → overfitting
    # - Need more regularization & shorter training
    6: {
        "name": "primary_vit_regularized",
        "model_type": "vit",
        "description": "ViT Regularized - High dropout, fewer epochs, early stopping",
        "batch_size": 16,
        "num_images": 3000,
        "epochs": 10,  # Stop before overfitting (peak was epoch 6)
        "learning_rate": 1e-4,  # Original LR worked best for BLEU
        "embed_dim": 256,
        "num_heads": 8,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "mlp_ratio": 4,
        "dropout": 0.25,  # Increased from 0.1 to combat overfitting
        "max_length": 30,
    },
    7: {
        "name": "primary_vit_deep_regularized",
        "model_type": "vit",
        "description": "ViT Deep Regularized - 6 layers, 384 dim, high dropout",
        "batch_size": 12,  # Smaller batch for larger model
        "num_images": 3000,
        "epochs": 12,  # Short training to avoid overfitting
        "learning_rate": 1e-4,  # Conservative LR for deep model
        "embed_dim": 384,  # Larger embedding dimension
        "num_heads": 8,
        "encoder_layers": 6,  # Deeper than previous (was 4)
        "decoder_layers": 6,
        "mlp_ratio": 4,
        "dropout": 0.2,  # High dropout for regularization
        "max_length": 30,
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
        from models.cnn_lstm import create_model
        
        # Load pretrained embeddings if specified
        embedding_type = config.get('embedding_type')
        embeddings = None
        if embedding_type:
            embeddings = load_embeddings(embedding_type)
        
        model = create_model(
            embedding_matrix=embeddings,
            vocab_size=vocab_size,
            embed_dim=config['embed_dim'],
            decoder_dim=config['hidden_dim'],
            attention_dim=config['attention_dim'],
            dropout=config['dropout'],
            pretrained_cnn=True
        )
    else:
        from models.vision_transformer import VisionTransformerCaptioning
        patch_size = config.get('patch_size', 16)
        # Support both old (num_layers) and new (encoder_layers/decoder_layers) configs
        encoder_layers = config.get('encoder_layers', config.get('num_layers', 4))
        decoder_layers = config.get('decoder_layers', config.get('num_layers', 4))
        mlp_ratio = config.get('mlp_ratio', 4)
        max_length = config.get('max_length', 30)
        model = VisionTransformerCaptioning(
            vocab_size=vocab_size,
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            mlp_ratio=mlp_ratio,
            max_length=max_length,
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
    
    # Create trainer (learning_rate goes in __init__)
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
    
    start_time = time.time()
    history = trainer.train(num_epochs=config['epochs'])
    duration = time.time() - start_time
    
    # Save results
    results = {
        "config_name": config_name,
        "description": config['description'],
        "model_type": config['model_type'],
        "num_images": config['num_images'],
        "epochs": config['epochs'],
        "parameters": total_params,
        "final_train_loss": history['train_loss'][-1] if history['train_loss'] else None,
        "final_val_loss": history['val_loss'][-1] if history['val_loss'] else None,
        "best_val_loss": min(history['val_loss']) if history['val_loss'] else None,
        "best_bleu": max(history['val_bleu']) if history['val_bleu'] else None,
        "duration_minutes": duration / 60,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE: {config_name}")
    print("=" * 70)
    print(f"Duration: {duration/60:.1f} minutes ({duration/3600:.2f} hours)")
    if history['val_loss']:
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"Best val loss: {min(history['val_loss']):.4f}")
        print(f"Best BLEU: {max(history['val_bleu']):.4f}")
    print(f"Results saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Primary Laptop Training - Variations")
    parser.add_argument("--config", type=int, help="Config number (1-7)")
    parser.add_argument("--list", action="store_true", help="List available configs")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Configurations for Primary Laptop:")
        print("=" * 70)
        
        print("\n--- CNN+LSTM Configs ---")
        for num in [1, 2, 3, 5]:
            cfg = VARIATION_CONFIGS[num]
            est_time = cfg['epochs'] * 5  # ~5 min per epoch
            print(f"\n  --config {num}: {cfg['name']}")
            print(f"      {cfg['description']}")
            print(f"      Model: {cfg['model_type']}, Epochs: {cfg['epochs']}, Est: ~{est_time} min")
        
        print("\n--- ViT Configs (NEW - optimized based on analysis) ---")
        for num in [4, 6, 7]:
            cfg = VARIATION_CONFIGS[num]
            est_time = cfg['epochs'] * 4  # ViT is faster
            print(f"\n  --config {num}: {cfg['name']}")
            print(f"      {cfg['description']}")
            print(f"      Model: {cfg['model_type']}, Epochs: {cfg['epochs']}, Est: ~{est_time} min")
        
        print("\nUsage:")
        print("  python scripts/train_variations.py --config 6  # ViT Regularized (recommended)")
        print("  python scripts/train_variations.py --config 7  # ViT Deep Regularized")
        return
    
    if args.config:
        train_config(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
