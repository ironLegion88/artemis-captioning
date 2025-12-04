#!/usr/bin/env python
"""
Local training script for ~3000 images, 30 epochs.
Runs CNN+LSTM model on CPU.
Estimated time: ~3.5 hours on i5-1235U
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import Trainer
from models.cnn_lstm import create_model
from utils.data_loader import create_data_loaders
from utils.text_preprocessing import TextPreprocessor


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


def main():
    start_time = datetime.now()
    
    print('=' * 70)
    print('LOCAL TRAINING: CNN+LSTM on ~3000 images, 30 epochs')
    print('=' * 70)
    print(f'Started at: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    # Load vocabulary
    text_proc = TextPreprocessor()
    text_proc.load_vocabulary('data/processed/vocabulary.json')
    print(f'Vocabulary size: {text_proc.vocab_size}')
    
    # Create data loaders
    loaders = create_data_loaders(
        text_preprocessor=text_proc,
        batch_size=16,
        num_workers=0,
        splits=['train', 'val']
    )
    
    # Limit to ~3000 images (187 batches × 16 = 2992 images)
    max_batches_train = 187
    max_batches_val = 50  # Smaller for faster validation
    train_loader = LimitedLoader(loaders['train'], max_batches_train)
    val_loader = LimitedLoader(loaders['val'], max_batches_val)
    
    print(f'Train batches: {len(train_loader)} (~{len(train_loader)*16} images)')
    print(f'Val batches: {len(val_loader)} (~{len(val_loader)*16} images)')
    
    # Create model
    model = create_model(vocab_size=text_proc.vocab_size, embed_dim=256)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model: CNN+LSTM')
    print(f'Parameters: {total_params:,}')
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        text_preprocessor=text_proc,
        checkpoint_dir='checkpoints/local_3k',
        output_dir='outputs/local_3k'
    )
    
    # Train
    print()
    print('Starting training...')
    print('=' * 70)
    history = trainer.train(num_epochs=30)
    
    # Save history
    end_time = datetime.now()
    duration = end_time - start_time
    
    results = {
        'model': 'cnn_lstm',
        'images': '~3000',
        'epochs': 30,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'best_val_loss': min(history['val_loss']),
        'final_bleu': history['val_bleu'][-1],
        'best_bleu': max(history['val_bleu']),
        'duration_minutes': duration.total_seconds() / 60,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat()
    }
    
    os.makedirs('outputs/local_3k', exist_ok=True)
    with open('outputs/local_3k/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print('=' * 70)
    print('TRAINING COMPLETE')
    print('=' * 70)
    print(f"Duration: {duration}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    print(f"Best BLEU: {max(history['val_bleu']):.4f}")
    print(f"Model saved to: checkpoints/local_3k/")
    print(f"Results saved to: outputs/local_3k/training_results.json")
    
    return history


if __name__ == '__main__':
    main()
