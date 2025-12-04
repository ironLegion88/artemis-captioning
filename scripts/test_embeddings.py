"""
Test Script for Text Embeddings

This script tests all embedding implementations:
1. TF-IDF (256D) - Always available
2. Word2Vec (300D) - Requires gensim
3. GloVe (300D) - Requires GloVe file download

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_preprocessing import TextPreprocessor
from utils.embeddings import TextEmbeddings, create_all_embeddings
from utils.constants import (
    PROCESSED_DIR,
    EMBEDDINGS_DIR,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    ARTEMIS_CSV
)


def test_tfidf_embeddings(text_preprocessor, captions):
    """Test TF-IDF embedding creation."""
    print("\n" + "=" * 70)
    print("TEST 1: TF-IDF EMBEDDINGS")
    print("=" * 70)
    
    embeddings = TextEmbeddings(text_preprocessor, embedding_dim=EMBEDDING_DIM)
    
    try:
        matrix = embeddings.create_tfidf_embeddings(captions)
        
        # Validate
        assert matrix.shape[0] == text_preprocessor.vocab_size, "Wrong vocab size"
        assert matrix.shape[1] == EMBEDDING_DIM, f"Wrong dim: {matrix.shape[1]}"
        assert np.allclose(matrix[0], 0), "PAD token should be zeros"
        assert not np.isnan(matrix).any(), "Contains NaN values"
        assert not np.isinf(matrix).any(), "Contains Inf values"
        
        # Test PyTorch layer creation
        embed_layer = embeddings.get_pytorch_embedding_layer(freeze=False)
        test_input = torch.tensor([[1, 2, 3, 100, 500]])  # Random indices
        output = embed_layer(test_input)
        assert output.shape == (1, 5, EMBEDDING_DIM), f"Wrong output shape: {output.shape}"
        
        print("\n✅ TF-IDF TEST PASSED")
        print(f"  - Shape: {matrix.shape}")
        print(f"  - Mean: {matrix.mean():.4f}")
        print(f"  - Std: {matrix.std():.4f}")
        print(f"  - PAD token norm: {np.linalg.norm(matrix[0]):.4f}")
        
        return True, matrix
        
    except Exception as e:
        print(f"\n❌ TF-IDF TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_word2vec_embeddings(text_preprocessor):
    """Test Word2Vec embedding creation."""
    print("\n" + "=" * 70)
    print("TEST 2: WORD2VEC EMBEDDINGS")
    print("=" * 70)
    
    try:
        import gensim.downloader as api
        print("✓ gensim is available")
    except ImportError:
        print("⚠ gensim not installed, skipping Word2Vec test")
        print("  Install with: pip install gensim")
        return None, None
    
    embeddings = TextEmbeddings(text_preprocessor, embedding_dim=300)
    
    try:
        print("\n✓ Loading Word2Vec model (this may take a few minutes)...")
        matrix = embeddings.create_word2vec_embeddings()
        
        # Validate
        assert matrix.shape[0] == text_preprocessor.vocab_size, "Wrong vocab size"
        assert matrix.shape[1] == 300, f"Wrong dim: {matrix.shape[1]}"
        assert np.allclose(matrix[0], 0), "PAD token should be zeros"
        
        # Count coverage
        non_random = 0
        for i in range(4, min(1000, matrix.shape[0])):
            # Check if embedding was mapped (not random)
            if np.abs(matrix[i]).max() > 0.5:  # Word2Vec vectors are larger
                non_random += 1
        
        coverage = non_random / min(996, matrix.shape[0] - 4) * 100
        
        # Test PyTorch layer
        embed_layer = embeddings.get_pytorch_embedding_layer(freeze=True)
        test_input = torch.tensor([[1, 2, 3, 100, 500]])
        output = embed_layer(test_input)
        assert output.shape == (1, 5, 300), f"Wrong output shape: {output.shape}"
        
        # Save embeddings
        save_path = EMBEDDINGS_DIR / "word2vec_embeddings.npy"
        embeddings.save_embeddings(save_path, embedding_type='word2vec')
        
        print("\n✅ WORD2VEC TEST PASSED")
        print(f"  - Shape: {matrix.shape}")
        print(f"  - Mean: {matrix.mean():.4f}")
        print(f"  - Std: {matrix.std():.4f}")
        print(f"  - Coverage (sample): ~{coverage:.1f}%")
        
        return True, matrix
        
    except Exception as e:
        print(f"\n❌ WORD2VEC TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_glove_embeddings(text_preprocessor):
    """Test GloVe embedding creation."""
    print("\n" + "=" * 70)
    print("TEST 3: GLOVE EMBEDDINGS")
    print("=" * 70)
    
    # Check if GloVe file exists
    glove_path = EMBEDDINGS_DIR / "glove.6B.300d.txt"
    
    if not glove_path.exists():
        print(f"⚠ GloVe file not found at: {glove_path}")
        print("\n  To use GloVe embeddings:")
        print("  1. Download from: https://nlp.stanford.edu/projects/glove/")
        print("  2. Extract glove.6B.zip")
        print(f"  3. Place glove.6B.300d.txt in: {EMBEDDINGS_DIR}")
        print("\n  Attempting to use gensim's GloVe instead...")
        
        # Try gensim's GloVe
        try:
            return test_glove_via_gensim(text_preprocessor)
        except Exception as e:
            print(f"  Could not load via gensim: {e}")
            return None, None
    
    embeddings = TextEmbeddings(text_preprocessor, embedding_dim=300)
    
    try:
        matrix = embeddings.create_glove_embeddings(glove_file=glove_path)
        
        # Validate
        assert matrix.shape[0] == text_preprocessor.vocab_size, "Wrong vocab size"
        assert matrix.shape[1] == 300, f"Wrong dim: {matrix.shape[1]}"
        assert np.allclose(matrix[0], 0), "PAD token should be zeros"
        
        # Save embeddings
        save_path = EMBEDDINGS_DIR / "glove_embeddings.npy"
        embeddings.save_embeddings(save_path, embedding_type='glove')
        
        print("\n✅ GLOVE TEST PASSED")
        print(f"  - Shape: {matrix.shape}")
        print(f"  - Mean: {matrix.mean():.4f}")
        print(f"  - Std: {matrix.std():.4f}")
        
        return True, matrix
        
    except Exception as e:
        print(f"\n❌ GLOVE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_glove_via_gensim(text_preprocessor):
    """Test GloVe embeddings using gensim downloader."""
    print("\n  Trying gensim's glove-wiki-gigaword-300...")
    
    try:
        import gensim.downloader as api
        
        # Load GloVe model via gensim
        print("  Loading GloVe model (this may take several minutes)...")
        glove_model = api.load('glove-wiki-gigaword-300')
        
        embeddings = TextEmbeddings(text_preprocessor, embedding_dim=300)
        
        # Initialize embedding matrix
        matrix = np.random.randn(text_preprocessor.vocab_size, 300) * 0.01
        
        # Map GloVe embeddings to vocabulary
        mapped_count = 0
        for word, idx in text_preprocessor.word2idx.items():
            if idx < 4:  # Skip special tokens
                continue
            
            try:
                matrix[idx] = glove_model[word]
                mapped_count += 1
            except KeyError:
                pass
        
        # Initialize special tokens
        matrix[text_preprocessor.PAD_IDX] = 0.0
        
        embeddings.embedding_matrix = matrix
        
        coverage = mapped_count / (text_preprocessor.vocab_size - 4) * 100
        
        # Save embeddings
        save_path = EMBEDDINGS_DIR / "glove_embeddings.npy"
        embeddings.save_embeddings(save_path, embedding_type='glove')
        
        print("\n✅ GLOVE (via gensim) TEST PASSED")
        print(f"  - Shape: {matrix.shape}")
        print(f"  - Coverage: {coverage:.1f}%")
        print(f"  - Mapped {mapped_count:,} words")
        
        return True, matrix
        
    except Exception as e:
        print(f"\n❌ GLOVE via gensim FAILED: {e}")
        return False, None


def compare_embeddings(results):
    """Compare embedding statistics."""
    print("\n" + "=" * 70)
    print("EMBEDDING COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Type':<15} {'Shape':<20} {'Mean':<10} {'Std':<10} {'Status'}")
    print("-" * 70)
    
    for emb_type, (success, matrix) in results.items():
        if matrix is not None:
            shape_str = f"{matrix.shape}"
            mean_str = f"{matrix.mean():.4f}"
            std_str = f"{matrix.std():.4f}"
            status = "✅ PASS" if success else "❌ FAIL"
        else:
            shape_str = "N/A"
            mean_str = "N/A"
            std_str = "N/A"
            status = "⚠ SKIP"
        
        print(f"{emb_type:<15} {shape_str:<20} {mean_str:<10} {std_str:<10} {status}")


def test_embedding_integration():
    """Test embedding integration with the CNN+LSTM model."""
    print("\n" + "=" * 70)
    print("TEST 4: MODEL INTEGRATION")
    print("=" * 70)
    
    try:
        from models.cnn_lstm import create_model
        
        # Load existing TF-IDF embeddings
        tfidf_path = EMBEDDINGS_DIR / "tfidf_embeddings.npy"
        if not tfidf_path.exists():
            print("⚠ TF-IDF embeddings not found, skipping integration test")
            return None
        
        tfidf_matrix = np.load(tfidf_path)
        embedding_tensor = torch.FloatTensor(tfidf_matrix)
        
        # Create model with embeddings
        model = create_model(
            vocab_size=tfidf_matrix.shape[0],
            embed_dim=tfidf_matrix.shape[1],
            embedding_matrix=embedding_tensor
        )
        
        # Test forward pass
        batch_size = 2
        images = torch.randn(batch_size, 3, 128, 128)
        captions = torch.randint(0, 100, (batch_size, 20))
        lengths = torch.tensor([18, 15])
        
        model.eval()
        with torch.no_grad():
            predictions, alphas, sorted_captions, decode_lengths, sort_ind = model(
                images, captions, lengths
            )
        
        print("\n✅ MODEL INTEGRATION TEST PASSED")
        print(f"  - Predictions shape: {predictions.shape}")
        print(f"  - Attention weights shape: {alphas.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ MODEL INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all embedding tests."""
    parser = argparse.ArgumentParser(description='Test text embeddings')
    parser.add_argument('--word2vec', action='store_true',
                       help='Test Word2Vec embeddings (downloads model)')
    parser.add_argument('--glove', action='store_true',
                       help='Test GloVe embeddings (requires download)')
    parser.add_argument('--all', action='store_true',
                       help='Test all embedding types')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TEXT EMBEDDING TESTS")
    print("=" * 70)
    
    # Load vocabulary
    vocab_path = PROCESSED_DIR / "vocabulary.json"
    if not vocab_path.exists():
        print(f"\n❌ Error: {vocab_path} not found")
        print("   Run data preprocessing first")
        return
    
    print("\n✓ Loading vocabulary...")
    text_preprocessor = TextPreprocessor()
    text_preprocessor.load_vocabulary(vocab_path)
    print(f"  - Vocabulary size: {text_preprocessor.vocab_size}")
    
    # Load captions for TF-IDF
    print("\n✓ Loading captions...")
    import pandas as pd
    
    selected_path = PROCESSED_DIR / "selected_images.json"
    with open(selected_path, 'r') as f:
        data = json.load(f)
    selected_paintings = {item['painting'] for item in data['selected_paintings']}
    
    df = pd.read_csv(ARTEMIS_CSV)
    captions = df[df['painting'].isin(selected_paintings)]['utterance'].tolist()
    print(f"  - Loaded {len(captions):,} captions")
    
    # Run tests
    results = {}
    
    # Always test TF-IDF
    results['tfidf'] = test_tfidf_embeddings(text_preprocessor, captions)
    
    # Test Word2Vec if requested
    if args.word2vec or args.all:
        results['word2vec'] = test_word2vec_embeddings(text_preprocessor)
    
    # Test GloVe if requested
    if args.glove or args.all:
        results['glove'] = test_glove_embeddings(text_preprocessor)
    
    # Compare results
    compare_embeddings(results)
    
    # Test model integration
    test_embedding_integration()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r[0] is True)
    skipped = sum(1 for r in results.values() if r[0] is None)
    failed = sum(1 for r in results.values() if r[0] is False)
    
    print(f"\n  ✅ Passed: {passed}")
    print(f"  ⚠ Skipped: {skipped}")
    print(f"  ❌ Failed: {failed}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")


if __name__ == "__main__":
    main()
