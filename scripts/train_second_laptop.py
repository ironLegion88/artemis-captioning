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

from utils.data_loader import create_dataloaders
from utils.evaluation import BLEUScore
from train import Trainer


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
        "attention_dim": 256,
        "dropout": 0.3,
        "encoder_lr_factor": 0.1,
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
        "num_heads": 8,
        "num_layers": 4,  # Fewer layers than standard
        "ff_dim": 512,
        "dropout": 0.1,
    },
    3: {
        "name": "laptop2_cnn_high_lr",
        "model_type": "cnn_lstm",
        "description": "CNN+LSTM High LR - Faster convergence experiment",
        "batch_size": 24,
        "num_images": 5000,
        "epochs": 25,  # Fewer epochs since higher LR
        "learning_rate": 5e-4,
        "embed_dim": 256,
        "hidden_dim": 512,
        "attention_dim": 256,
        "dropout": 0.4,  # Higher dropout with higher LR
        "encoder_lr_factor": 0.05,
    },
}


def create_cnn_lstm_model(vocab_size, config):
    """Create CNN+LSTM model."""
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
    return model


def create_vit_model(vocab_size, config):
    """Create Vision Transformer model."""
    from models.vision_transformer import VisionTransformerCaptioning
    
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
        patch_size=16
    )
    return model


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


def train_config(config_num):
    """Train a single configuration."""
    if config_num not in LAPTOP2_CONFIGS:
        print(f"Error: Config {config_num} not found. Available: {list(LAPTOP2_CONFIGS.keys())}")
        return None
    
    config = LAPTOP2_CONFIGS[config_num]
    config_name = config['name']
    
    print("=" * 70)
    print(f"SECOND LAPTOP TRAINING: {config_name}")
    print(f"Description: {config['description']}")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    checkpoint_dir = f"checkpoints/{config_name}"
    output_dir = f"outputs/{config_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(f"{output_dir}/config.json", "w", encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    # Load vocabulary
    vocab_file = "data/processed/vocabulary.json"
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    vocab_size = vocab_data['vocab_size']
    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
    
    print(f"\nVocabulary size: {vocab_size}")
    
    # Create data loaders
    print("\n" + "=" * 70)
    print("CREATING DATA LOADERS")
    print("=" * 70)
    
    train_loader, val_loader, _ = create_dataloaders(
        images_dir="data/processed/images",
        captions_dir="data/processed/captions",
        splits_dir="data/processed/splits",
        vocab_file=vocab_file,
        batch_size=config['batch_size'],
        num_workers=0
    )
    
    # Limit batches
    max_train_batches = config['num_images'] // config['batch_size']
    max_val_batches = max(20, max_train_batches // 5)
    
    train_loader = LimitedLoader(train_loader, max_train_batches)
    val_loader = LimitedLoader(val_loader, max_val_batches)
    
    print(f"Train batches: {len(train_loader)} (~{len(train_loader) * config['batch_size']} images)")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    if config['model_type'] == 'cnn_lstm':
        model = create_cnn_lstm_model(vocab_size, config)
    else:
        model = create_vit_model(vocab_size, config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model_type']}")
    print(f"Device: {device}")
    print(f"Parameters: {total_params:,}")
    
    # Create optimizer
    if config['model_type'] == 'cnn_lstm':
        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters()) + list(model.attention.parameters())
        optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': config['learning_rate'] * config['encoder_lr_factor']},
            {'params': decoder_params, 'lr': config['learning_rate']}
        ], weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-5
        )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Create trainer
    bleu_scorer = BLEUScore(idx_to_word, word_to_idx)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=bleu_scorer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        grad_clip=5.0
    )
    
    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    import time
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
        "final_train_loss": history[-1]['train_loss'],
        "final_val_loss": history[-1]['val_loss'],
        "best_val_loss": min(h['val_loss'] for h in history),
        "best_bleu": max(h.get('bleu', 0) for h in history),
        "duration_minutes": duration / 60,
        "start_time": datetime.now().isoformat(),
    }
    
    with open(f"{output_dir}/results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    with open(f"{output_dir}/training_history.json", "w", encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE: {config_name}")
    print("=" * 70)
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"Best BLEU: {results['best_bleu']:.4f}")
    print(f"Results saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Second Laptop Training Script")
    parser.add_argument("--config", type=int, help="Config number (1, 2, or 3)")
    parser.add_argument("--all", action="store_true", help="Train all configs sequentially")
    parser.add_argument("--list", action="store_true", help="List available configs")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Configurations for Second Laptop:")
        print("=" * 70)
        for num, cfg in LAPTOP2_CONFIGS.items():
            print(f"\n  --config {num}: {cfg['name']}")
            print(f"      {cfg['description']}")
            print(f"      Model: {cfg['model_type']}, Images: {cfg['num_images']}, Epochs: {cfg['epochs']}")
        print("\nUsage:")
        print("  python scripts/train_second_laptop.py --config 1")
        print("  python scripts/train_second_laptop.py --all")
        return
    
    if args.all:
        print("\n" + "=" * 70)
        print("TRAINING ALL CONFIGURATIONS")
        print("=" * 70)
        
        all_results = {}
        for config_num in LAPTOP2_CONFIGS.keys():
            try:
                results = train_config(config_num)
                all_results[LAPTOP2_CONFIGS[config_num]['name']] = results
            except Exception as e:
                print(f"\n❌ Error training config {config_num}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save combined results
        with open("outputs/laptop2_all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "=" * 70)
        print("ALL TRAINING COMPLETE!")
        print("Results saved to: outputs/laptop2_all_results.json")
        print("=" * 70)
    
    elif args.config:
        train_config(args.config)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
