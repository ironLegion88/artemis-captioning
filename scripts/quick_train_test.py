#!/usr/bin/env python
"""
Quick training test on ~300 images for 10 epochs.
Should complete in ~5 minutes on i5-1235U.
"""

import sys
import os
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
    print('=' * 60)
    print('TEST TRAINING: ~300 images, 10 epochs')
    print('=' * 60)
    
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
    
    # Limit to ~300 images (19 batches × 16 = 304 images)
    max_batches = 19
    train_loader = LimitedLoader(loaders['train'], max_batches)
    val_loader = LimitedLoader(loaders['val'], max_batches)
    print(f'Train batches: {len(train_loader)} (~{len(train_loader)*16} images)')
    print(f'Val batches: {len(val_loader)}')
    
    # Create model
    model = create_model(vocab_size=text_proc.vocab_size, embed_dim=256)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        text_preprocessor=text_proc,
        checkpoint_dir='checkpoints/test_300'
    )
    
    # Train for 10 epochs
    print()
    history = trainer.train(num_epochs=10)
    
    print()
    print('=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final BLEU: {history['val_bleu'][-1]:.4f}")
    print(f"Best BLEU: {max(history['val_bleu']):.4f}")
    
    # Show loss reduction
    loss_reduction = history['train_loss'][0] - history['train_loss'][-1]
    print(f"Train loss reduction: {loss_reduction:.4f}")
    
    return history


if __name__ == '__main__':
    main()
