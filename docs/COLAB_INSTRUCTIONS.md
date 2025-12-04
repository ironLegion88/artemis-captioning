# Google Colab Training Instructions

## Files to Upload to Google Drive

Create a folder called `artemis-captioning` in your Google Drive and upload the following:

### Required Folders:
```
artemis-captioning/
├── data/
│   ├── processed/
│   │   ├── images/           # Pre-resized 128x128 images (~57 MB) - RECOMMENDED
│   │   │   ├── Impressionism/
│   │   │   ├── Realism/
│   │   │   └── ... (style folders)
│   │   ├── vocabulary.json
│   │   └── splits/
│   │       ├── train.json
│   │       ├── val.json
│   │       └── test.json
│   └── raw/
│       └── wikiart/          # Original images (optional if using preprocessed)
├── models/
│   ├── __init__.py
│   ├── cnn_lstm.py
│   └── vision_transformer.py
├── utils/
│   ├── __init__.py
│   ├── constants.py
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── image_preprocessing.py
│   ├── text_preprocessing.py
│   └── visualization.py
├── scripts/
│   └── preprocess_images.py  # For creating preprocessed images
└── train.py
```

### Preprocessing Images (Recommended - Do This First!)

Before uploading to Colab, run the preprocessing script locally:
```bash
python scripts/preprocess_images.py
```

This creates `data/processed/images/` with all 5,000 images pre-resized to 128x128.
- Takes ~30-60 seconds locally
- Saves ~57 MB (much smaller than raw images)
- Makes Colab training ~20-30% faster

### Upload Steps:

1. **Run preprocessing locally first** (optional but recommended):
   ```bash
   python scripts/preprocess_images.py
   ```

2. **Zip the data folder** (to speed up upload):
   ```
   # In your project directory, create a zip:
   # Right-click on 'data' folder → Send to → Compressed (zipped) folder
   # OR just zip data/processed/ if you don't need raw images
   ```

3. **Upload to Google Drive**:
   - Go to drive.google.com
   - Create folder: `artemis-captioning`
   - Upload: `data.zip`, `models/`, `utils/`, `scripts/`, `train.py`
   - Extract `data.zip` in Google Drive

4. **Open the Colab notebook**:
   - Upload `notebooks/Colab_Training.ipynb` to Colab
   - Or open directly from Drive

5. **Set runtime to GPU**:
   - Runtime → Change runtime type → T4 GPU

6. **Run all cells in order**

## Estimated Training Time on T4 GPU

| Model | ~15,000 images | 50 epochs | Time |
|-------|----------------|-----------|------|
| CNN+LSTM | ✓ | ✓ | ~2-3 hours |
| Vision Transformer | ✓ | ✓ | ~1.5-2 hours |
| **Total** | | | **~4-5 hours** |

## After Training

The trained models will be saved to your Google Drive:
- `artemis-captioning/checkpoints/colab_cnn_lstm/best_model.pth`
- `artemis-captioning/checkpoints/colab_vit/best_model.pth`

Download these and use with `scripts/predict.py` locally.

## Troubleshooting

### "Data folder not found"
- Check the path in cell 6 matches your Drive folder structure
- Make sure you extracted the zip file

### "Out of memory"
- Reduce `BATCH_SIZE` from 32 to 16
- Reduce `MAX_BATCHES` to use fewer images

### "Module not found"
- Make sure all files in `utils/` and `models/` are uploaded
- Check that `__init__.py` files exist
