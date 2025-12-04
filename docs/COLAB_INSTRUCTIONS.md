# Google Colab Training Instructions

## Files to Upload to Google Drive

Create a folder called `artemis-captioning` in your Google Drive and upload the following:

### Required Folders:
```
artemis-captioning/
├── data/
│   ├── processed/
│   │   ├── vocabulary.json
│   │   └── splits/
│   │       ├── train.json
│   │       ├── val.json
│   │       └── test.json
│   └── raw/
│       └── wikiart/          (all image folders)
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
└── train.py
```

### Upload Steps:

1. **Zip the data folder** (to speed up upload):
   ```
   # In your project directory, create a zip:
   # Right-click on 'data' folder → Send to → Compressed (zipped) folder
   ```

2. **Upload to Google Drive**:
   - Go to drive.google.com
   - Create folder: `artemis-captioning`
   - Upload: `data.zip`, `models/`, `utils/`, `train.py`
   - Extract `data.zip` in Google Drive

3. **Open the Colab notebook**:
   - Upload `notebooks/Colab_Training.ipynb` to Colab
   - Or open directly from Drive

4. **Set runtime to GPU**:
   - Runtime → Change runtime type → T4 GPU

5. **Run all cells in order**

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
