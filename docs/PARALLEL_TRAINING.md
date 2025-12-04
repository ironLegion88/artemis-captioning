# Parallel Training Setup Guide

This guide helps you set up training on a second machine for parallel hyperparameter experimentation.

## Quick Setup on Second Laptop

### 1. Clone the Repository
```bash
git clone https://github.com/ironLegion88/artemis-captioning.git
cd artemis-captioning
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Copy Data Files
You'll need to copy these from your main laptop (they're not in git):
- `data/processed/` - All processed images and captions
- `data/embeddings/` - Pretrained embeddings (optional)

**Option A: Use a USB drive or network share**
```bash
# Copy the entire data folder from your main laptop
```

**Option B: Re-run data processing** (slower but works independently)
```bash
# Download wikiart images and artemis captions, then run:
python scripts/analyze_dataset.py
python scripts/create_splits.py
```

## Available Training Configurations

The `train_hyperparameter.py` script has 6 predefined configurations:

| Config | Model | Learning Rate | Batch Size | Description |
|--------|-------|--------------|------------|-------------|
| 1 | CNN+LSTM | 5e-5 | 8 | Low LR, small batch (stable) |
| 2 | CNN+LSTM | 5e-4 | 32 | High LR, large batch (fast) |
| 3 | ViT | 1e-4 | 16 | Standard ViT configuration |
| 4 | ViT | 5e-5 | 8 | Larger capacity ViT |
| 5 | CNN+LSTM | 1e-4 | 16 | Larger capacity CNN+LSTM |
| 6 | CNN+LSTM | 1e-4 | 16 | CNN+LSTM with GloVe embeddings |

## Recommended Parallel Strategy

### Laptop 1 (Your Current Machine)
Run configurations 1, 3, 5:
```bash
# In terminal 1
python scripts/train_hyperparameter.py --config 1 --epochs 30

# In terminal 2 (after config 1 finishes, or if you have resources)
python scripts/train_hyperparameter.py --config 3 --epochs 30

# In terminal 3
python scripts/train_hyperparameter.py --config 5 --epochs 30
```

### Laptop 2 (Second Machine)
Run configurations 2, 4, 6:
```bash
# In terminal 1
python scripts/train_hyperparameter.py --config 2 --epochs 30

# In terminal 2
python scripts/train_hyperparameter.py --config 4 --epochs 30

# In terminal 3
python scripts/train_hyperparameter.py --config 6 --epochs 30
```

## Custom Configurations

You can also run with custom hyperparameters:

```bash
python scripts/train_hyperparameter.py --config custom \
    --model cnn_lstm \
    --lr 0.0003 \
    --batch_size 24 \
    --embed_dim 384 \
    --hidden_dim 768 \
    --dropout 0.35 \
    --epochs 30 \
    --experiment_name "custom_experiment_1"
```

### Custom Options
- `--model`: `cnn_lstm` or `vit`
- `--lr`: Learning rate (e.g., 1e-4, 5e-5, 3e-4)
- `--batch_size`: 8, 16, 24, 32 (adjust based on RAM)
- `--embed_dim`: 256, 384, 512
- `--hidden_dim`: 512, 768, 1024
- `--dropout`: 0.1 to 0.5
- `--embedding_type`: `glove`, `word2vec`, `tfidf`, or omit for random init
- `--num_images`: Number of images per epoch (default 3000)
- `--epochs`: Number of training epochs

## Training Time Estimates

On CPU (i5/i7 laptop):
- ~3-4 hours for 30 epochs with ~3000 images
- ViT is slightly faster than CNN+LSTM per batch

On GPU (if available):
- ~30-60 minutes for same configuration

## Monitoring and Results

Each experiment creates:
- `outputs/experiments/{experiment_name}/config.json` - Configuration used
- `outputs/experiments/{experiment_name}/training_history.json` - Training metrics
- `checkpoints/{experiment_name}/best_model.pth` - Best model checkpoint
- `checkpoints/{experiment_name}/latest_checkpoint.pth` - Latest checkpoint

## Combining Results

After training on both laptops, copy the results back:
1. Copy `outputs/experiments/` folders from second laptop
2. Copy `checkpoints/` folders from second laptop
3. Compare results to find best hyperparameters

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` to 8 or 4
- Reduce `--num_images` to 1000

### Slow Training
- Increase `--batch_size` if RAM allows
- Use GPU if available (CUDA will be auto-detected)

### Missing Files
- Ensure `data/processed/` is properly copied
- Check that `data/processed/vocabulary.json` exists
- Run `python scripts/analyze_dataset.py` if needed
