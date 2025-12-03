"""
Training Pipeline for ArtEmis Caption Generation

This module implements the complete training pipeline including:
- Training loop with teacher forcing
- Validation with BLEU score computation
- Learning rate scheduling
- Early stopping
- Checkpointing

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, MIN_DELTA, DROPOUT, DEVICE, USE_AMP,
    CHECKPOINTS_DIR, OUTPUTS_DIR, LOG_INTERVAL, SAVE_INTERVAL,
    VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, IMAGE_FEATURE_DIM,
    NUM_LSTM_LAYERS, EMBEDDINGS_DIR, MAX_CAPTION_LENGTH,
    RANDOM_SEED, PROCESSED_DIR
)
from utils.data_loader import create_data_loaders
from utils.text_preprocessing import TextPreprocessor
from models.cnn_lstm import create_model

import numpy as np


# Set up logging
def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like BLEU
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            epoch: Current epoch number
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:  # mode == 'max'
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """
    Trainer class for CNN+LSTM image captioning model.
    
    Handles training, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        text_preprocessor: TextPreprocessor,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        device: str = DEVICE,
        checkpoint_dir: str = str(CHECKPOINTS_DIR),
        output_dir: str = str(OUTPUTS_DIR),
        use_amp: bool = USE_AMP
    ):
        """
        Initialize trainer.
        
        Args:
            model: The captioning model
            train_loader: Training data loader
            val_loader: Validation data loader
            text_preprocessor: Text preprocessor for decoding
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
            output_dir: Directory for logs and outputs
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.text_preprocessor = text_preprocessor
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.use_amp = use_amp and device == 'cuda'  # AMP only for GPU
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logging(output_dir)
        
        # Loss function (ignore padding index)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_bleu': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_bleu = 0.0
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch['images'].to(self.device)
            captions = batch['captions'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    predictions, alphas, sorted_captions, decode_lengths, sort_ind = self.model(
                        images, captions, lengths
                    )
                    
                    # Compute loss
                    loss = self._compute_loss(predictions, sorted_captions, decode_lengths)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                predictions, alphas, sorted_captions, decode_lengths, sort_ind = self.model(
                    images, captions, lengths
                )
                
                # Compute loss
                loss = self._compute_loss(predictions, sorted_captions, decode_lengths)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                # Optimizer step
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = total_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed
                eta = (num_batches - batch_idx - 1) / batches_per_sec
                
                self.logger.info(
                    f"Epoch {epoch+1} [{batch_idx+1}/{num_batches}] "
                    f"Loss: {avg_loss:.4f} | "
                    f"Speed: {batches_per_sec:.2f} batch/s | "
                    f"ETA: {eta:.0f}s"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        decode_lengths: List[int]
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Model predictions (batch, max_len-1, vocab_size)
            targets: Target captions (batch, max_len)
            decode_lengths: List of actual caption lengths
        
        Returns:
            Loss value
        """
        # Pack predictions and targets
        # Predictions are for positions 0 to T-1 (predicting words 1 to T)
        # Targets should be words 1 to T (not including <SOS>)
        
        batch_size = predictions.size(0)
        max_decode_len = predictions.size(1)
        
        # Create packed targets (excluding <SOS> token at position 0)
        packed_predictions = []
        packed_targets = []
        
        for i in range(batch_size):
            length = decode_lengths[i]
            packed_predictions.append(predictions[i, :length])
            packed_targets.append(targets[i, 1:length+1])  # Skip <SOS>, get next 'length' words
        
        packed_predictions = torch.cat(packed_predictions, dim=0)
        packed_targets = torch.cat(packed_targets, dim=0)
        
        loss = self.criterion(packed_predictions, packed_targets)
        
        return loss
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (validation loss, BLEU-4 score)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        # For BLEU calculation
        references = []
        hypotheses = []
        
        for batch in self.val_loader:
            images = batch['images'].to(self.device)
            captions = batch['captions'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            # Forward pass
            predictions, alphas, sorted_captions, decode_lengths, sort_ind = self.model(
                images, captions, lengths
            )
            
            # Compute loss
            loss = self._compute_loss(predictions, sorted_captions, decode_lengths)
            total_loss += loss.item()
            
            # Generate captions for BLEU (only for first image in batch)
            # This is sampling-based, not from teacher forcing
            if len(references) < 500:  # Limit for speed
                # Get original order
                reverse_sort_ind = torch.argsort(sort_ind)
                
                for i in range(min(2, images.size(0))):  # Sample 2 per batch
                    orig_idx = reverse_sort_ind[i].item()
                    
                    # Reference caption (ground truth)
                    ref_caption = captions[orig_idx].cpu().numpy().tolist()
                    ref_words = self.text_preprocessor.decode(ref_caption, skip_special_tokens=True)
                    references.append([ref_words.split()])
                    
                    # Generate hypothesis
                    single_image = images[orig_idx:orig_idx+1]
                    encoder_out = self.model.encoder(single_image)
                    gen_caption, _ = self.model.decoder.predict(encoder_out, max_length=MAX_CAPTION_LENGTH)
                    hyp_words = self.text_preprocessor.decode(gen_caption.cpu().numpy().tolist(), skip_special_tokens=True)
                    hypotheses.append(hyp_words.split())
        
        avg_loss = total_loss / num_batches
        
        # Calculate BLEU score
        bleu_score = self._calculate_bleu(references, hypotheses)
        
        return avg_loss, bleu_score
    
    def _calculate_bleu(
        self,
        references: List[List[List[str]]],
        hypotheses: List[List[str]]
    ) -> float:
        """
        Calculate corpus BLEU-4 score.
        
        Args:
            references: List of reference captions (each is list of lists of words)
            hypotheses: List of hypothesis captions (each is list of words)
        
        Returns:
            BLEU-4 score
        """
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            
            smoothie = SmoothingFunction().method4
            bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
            return bleu
        except ImportError:
            self.logger.warning("NLTK not available, using simple BLEU approximation")
            return self._simple_bleu(references, hypotheses)
    
    def _simple_bleu(
        self,
        references: List[List[List[str]]],
        hypotheses: List[List[str]]
    ) -> float:
        """Simple BLEU-1 approximation without NLTK."""
        total_score = 0.0
        
        for refs, hyp in zip(references, hypotheses):
            ref = refs[0]  # Take first reference
            ref_set = set(ref)
            hyp_set = set(hyp)
            
            if len(hyp) > 0:
                precision = len(ref_set & hyp_set) / len(hyp)
                total_score += precision
        
        return total_score / len(hypotheses) if hypotheses else 0.0
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        bleu_score: float,
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            bleu_score: BLEU score
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'bleu_score': bleu_score,
            'history': self.history
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with BLEU: {bleu_score:.4f}")
        
        # Save epoch checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Epoch to resume from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        
        return checkpoint['epoch'] + 1
    
    def train(
        self,
        num_epochs: int = NUM_EPOCHS,
        resume_from: Optional[str] = None
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        
        Returns:
            Training history dictionary
        """
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from)
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {num_epochs}")
        self.logger.info(f"Batch size: {self.train_loader.batch_size}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        self.logger.info("=" * 60)
        
        training_start = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, bleu_score = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_bleu'].append(bleu_score)
            self.history['learning_rate'].append(current_lr)
            
            # Check if best model
            is_best = bleu_score > self.best_bleu
            if is_best:
                self.best_bleu = bleu_score
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, bleu_score, is_best)
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            self.logger.info("-" * 60)
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"BLEU: {bleu_score:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.0f}s"
            )
            self.logger.info("-" * 60)
            
            # Early stopping check
            if self.early_stopping(val_loss, epoch):
                self.logger.info(
                    f"Early stopping triggered! "
                    f"Best epoch: {self.early_stopping.best_epoch+1}"
                )
                break
        
        # Training complete
        total_time = time.time() - training_start
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best BLEU score: {self.best_bleu:.4f}")
        self.logger.info("=" * 60)
        
        # Save training history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def main():
    """Main training function."""
    print("=" * 70)
    print("ARTEMIS CAPTION GENERATION - TRAINING PIPELINE")
    print("=" * 70)
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load text preprocessor first (needed for data loader)
    print("\n✓ Loading text preprocessor...")
    text_preprocessor = TextPreprocessor()
    vocab_path = PROCESSED_DIR / 'vocabulary.json'
    text_preprocessor.load_vocabulary(vocab_path)
    print(f"  - Vocabulary size: {text_preprocessor.vocab_size}")
    
    # Create data loaders
    print("\n✓ Loading data...")
    data_loaders = create_data_loaders(
        text_preprocessor=text_preprocessor,
        batch_size=BATCH_SIZE,
        num_workers=0,
        splits=['train', 'val']
    )
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    print(f"  - Training samples: {len(train_loader.dataset)}")
    print(f"  - Validation samples: {len(val_loader.dataset)}")
    
    # Load embeddings
    print("\n✓ Loading TF-IDF embeddings...")
    tfidf_path = EMBEDDINGS_DIR / 'tfidf_embeddings.npy'
    if tfidf_path.exists():
        tfidf_matrix = np.load(tfidf_path)
        embedding_tensor = torch.FloatTensor(tfidf_matrix)
        print(f"  - Embedding shape: {embedding_tensor.shape}")
    else:
        embedding_tensor = None
        print("  - No pre-trained embeddings found, using random initialization")
    
    # Create model
    print("\n✓ Creating model...")
    model = create_model(
        embedding_matrix=embedding_tensor,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        encoder_dim=IMAGE_FEATURE_DIM,
        decoder_dim=HIDDEN_DIM,
        num_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
        pretrained_cnn=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("\n✓ Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        text_preprocessor=text_preprocessor,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE
    )
    
    # Check for existing checkpoint
    resume_path = CHECKPOINTS_DIR / 'latest_checkpoint.pth'
    if resume_path.exists():
        print(f"\n✓ Found existing checkpoint, resuming training...")
        resume_path = str(resume_path)
    else:
        resume_path = None
    
    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    history = trainer.train(
        num_epochs=NUM_EPOCHS,
        resume_from=resume_path
    )
    
    print("\n✓ Training complete!")
    print(f"  - Best BLEU score: {trainer.best_bleu:.4f}")
    print(f"  - Best model saved to: {CHECKPOINTS_DIR}/best_model.pth")


if __name__ == "__main__":
    main()
