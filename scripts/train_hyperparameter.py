"""
Hyperparameter Training Script
==============================
Configurable training script for parallel experimentation on multiple machines.

Usage:
    python scripts/train_hyperparameter.py --config 1  # Laptop 1 configuration
    python scripts/train_hyperparameter.py --config 2  # Laptop 2 configuration
    python scripts/train_hyperparameter.py --config custom --lr 0.0005 --batch_size 8 --model cnn_lstm
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


# Predefined hyperparameter configurations for parallel experiments
CONFIGS = {
    # Configuration 1: Lower learning rate, smaller batch (more stable)
    1: {
        "model_type": "cnn_lstm",
        "learning_rate": 5e-5,
        "batch_size": 8,
        "embed_dim": 256,
        "hidden_dim": 512,
        "attention_dim": 256,
        "encoder_lr_factor": 0.1,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "dropout": 0.3,
        "description": "CNN+LSTM with low LR, small batch (stable training)"
    },
    
    # Configuration 2: Higher learning rate, larger batch (faster convergence)  
    2: {
        "model_type": "cnn_lstm",
        "learning_rate": 5e-4,
        "batch_size": 32,
        "embed_dim": 256,
        "hidden_dim": 512,
        "attention_dim": 256,
        "encoder_lr_factor": 0.1,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "dropout": 0.5,
        "description": "CNN+LSTM with high LR, large batch (fast training)"
    },
    
    # Configuration 3: ViT with standard hyperparameters
    3: {
        "model_type": "vit",
        "learning_rate": 1e-4,
        "batch_size": 16,
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 4,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "dropout": 0.1,
        "description": "Vision Transformer with standard hyperparameters"
    },
    
    # Configuration 4: ViT with higher capacity
    4: {
        "model_type": "vit",
        "learning_rate": 5e-5,
        "batch_size": 8,
        "embed_dim": 512,
        "hidden_dim": 1024,
        "num_heads": 8,
        "num_layers": 6,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "dropout": 0.2,
        "description": "Vision Transformer with larger capacity"
    },
    
    # Configuration 5: CNN+LSTM with larger capacity
    5: {
        "model_type": "cnn_lstm",
        "learning_rate": 1e-4,
        "batch_size": 16,
        "embed_dim": 512,
        "hidden_dim": 1024,
        "attention_dim": 512,
        "encoder_lr_factor": 0.05,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "dropout": 0.4,
        "description": "CNN+LSTM with larger capacity"
    },
    
    # Configuration 6: CNN+LSTM with GloVe embeddings
    6: {
        "model_type": "cnn_lstm",
        "learning_rate": 1e-4,
        "batch_size": 16,
        "embed_dim": 300,  # GloVe dimension
        "hidden_dim": 512,
        "attention_dim": 256,
        "encoder_lr_factor": 0.1,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "dropout": 0.3,
        "embedding_type": "glove",
        "description": "CNN+LSTM with pretrained GloVe embeddings"
    },
}


def create_cnn_lstm_model(vocab_size, config):
    """Create CNN+LSTM model with specified hyperparameters."""
    from models.cnn_lstm import ImageCaptioningModel
    
    model = ImageCaptioningModel(
        embed_dim=config.get("embed_dim", 256),
        attention_dim=config.get("attention_dim", 256),
        decoder_dim=config.get("hidden_dim", 512),
        vocab_size=vocab_size,
        encoder_dim=2048,
        dropout=config.get("dropout", 0.3),
        pretrained_encoder=True
    )
    return model


def create_vit_model(vocab_size, config):
    """Create Vision Transformer model with specified hyperparameters."""
    from models.vision_transformer import VisionTransformerCaptioning
    
    model = VisionTransformerCaptioning(
        vocab_size=vocab_size,
        embed_dim=config.get("embed_dim", 256),
        num_heads=config.get("num_heads", 8),
        num_encoder_layers=config.get("num_layers", 4),
        num_decoder_layers=config.get("num_layers", 4),
        ff_dim=config.get("hidden_dim", 512) * 2,
        max_seq_len=30,
        dropout=config.get("dropout", 0.1),
        img_size=224,
        patch_size=16
    )
    return model


def load_embeddings(embedding_type, vocab_file):
    """Load pretrained embeddings if specified."""
    if not embedding_type:
        return None
    
    embedding_path = f"data/embeddings/{embedding_type}_embeddings.npy"
    if not os.path.exists(embedding_path):
        print(f"Warning: Embedding file {embedding_path} not found. Using random initialization.")
        return None
    
    import numpy as np
    embeddings = np.load(embedding_path)
    print(f"Loaded {embedding_type} embeddings: {embeddings.shape}")
    return torch.tensor(embeddings, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Training Script")
    parser.add_argument("--config", type=str, required=True, 
                        help="Configuration number (1-6) or 'custom'")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--num_images", type=int, default=3000, 
                        help="Approximate number of images to train on")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for the experiment output folder")
    
    # Custom hyperparameters (only used when --config custom)
    parser.add_argument("--model", type=str, choices=["cnn_lstm", "vit"], 
                        default="cnn_lstm")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--embedding_type", type=str, default=None,
                        choices=["glove", "word2vec", "tfidf"],
                        help="Pretrained embedding type")
    
    args = parser.parse_args()
    
    # Get configuration
    if args.config.lower() == "custom":
        config = {
            "model_type": args.model,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "attention_dim": args.embed_dim,
            "encoder_lr_factor": 0.1,
            "weight_decay": 1e-5,
            "grad_clip": 5.0,
            "embedding_type": args.embedding_type,
            "description": "Custom configuration"
        }
    else:
        config_num = int(args.config)
        if config_num not in CONFIGS:
            print(f"Error: Config {config_num} not found. Available: {list(CONFIGS.keys())}")
            sys.exit(1)
        config = CONFIGS[config_num]
    
    # Set experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = f"config_{args.config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=" * 70)
    print(f"HYPERPARAMETER TRAINING: {experiment_name}")
    print("=" * 70)
    print(f"Configuration: {config['description']}")
    print(f"Model: {config['model_type']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Embed Dim: {config['embed_dim']}")
    print(f"Hidden Dim: {config['hidden_dim']}")
    print(f"Dropout: {config['dropout']}")
    print(f"Epochs: {args.epochs}")
    print(f"Target Images: ~{args.num_images}")
    print("=" * 70)
    
    # Create output directories
    output_dir = f"outputs/experiments/{experiment_name}"
    checkpoint_dir = f"checkpoints/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration
    config_save = {**config, "epochs": args.epochs, "num_images": args.num_images}
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_save, f, indent=2)
    
    # Load vocabulary
    vocab_file = "data/processed/vocabulary.json"
    with open(vocab_file, 'r') as f:
        vocab_data = json.load(f)
    vocab_size = vocab_data['vocab_size']
    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
    
    print(f"\nVocabulary size: {vocab_size}")
    
    # Calculate batches for target image count
    batch_size = config["batch_size"]
    target_batches = args.num_images // batch_size
    
    # Create data loaders
    print("\n" + "=" * 70)
    print("CREATING DATA LOADERS")
    print("=" * 70)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        images_dir="data/processed/images",
        captions_dir="data/processed/captions",
        splits_dir="data/processed/splits",
        vocab_file=vocab_file,
        batch_size=batch_size,
        num_workers=0
    )
    
    print(f"Using ~{target_batches * batch_size} images per epoch ({target_batches} batches)")
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    if config["model_type"] == "cnn_lstm":
        model = create_cnn_lstm_model(vocab_size, config)
    else:
        model = create_vit_model(vocab_size, config)
    
    # Load pretrained embeddings if specified
    embedding_type = config.get("embedding_type")
    if embedding_type:
        embeddings = load_embeddings(embedding_type, vocab_file)
        if embeddings is not None:
            if config["model_type"] == "cnn_lstm":
                model.decoder.embedding.weight.data.copy_(embeddings)
            else:
                model.decoder.embedding.weight.data.copy_(embeddings)
            print(f"Loaded {embedding_type} embeddings into model")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model_type']}")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    if config["model_type"] == "cnn_lstm":
        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters()) + list(model.attention.parameters())
        
        optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': config['learning_rate'] * config['encoder_lr_factor']},
            {'params': decoder_params, 'lr': config['learning_rate']}
        ], weight_decay=config.get('weight_decay', 1e-5))
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Create evaluator
    bleu_scorer = BLEUScore(idx_to_word, word_to_idx)
    
    # Limit batches per epoch
    class LimitedLoader:
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
    
    limited_train_loader = LimitedLoader(train_loader, target_batches)
    limited_val_loader = LimitedLoader(val_loader, max(10, target_batches // 4))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=limited_train_loader,
        val_loader=limited_val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=bleu_scorer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        grad_clip=config.get('grad_clip', 5.0)
    )
    
    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    history = trainer.train(num_epochs=args.epochs)
    
    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Print summary
    if history:
        best_bleu = max(h.get('bleu', 0) for h in history)
        best_val_loss = min(h['val_loss'] for h in history)
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        print(f"Best BLEU score: {best_bleu:.4f}")


if __name__ == "__main__":
    main()
