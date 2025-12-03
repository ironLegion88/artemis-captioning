"""
Text Embedding Module for ArtEmis Caption Generation

This module creates and manages text embeddings for caption generation.
Implements three embedding methods:
1. TF-IDF (256D) - Statistical embeddings
2. Word2Vec (300D) - Pre-trained word embeddings
3. GloVe (300D) - Pre-trained word embeddings

Each method creates embedding matrices that map vocabulary indices to dense vectors.

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from utils.constants import (
    PROCESSED_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_DIM,
    VOCAB_SIZE
)
from utils.text_preprocessing import TextPreprocessor


class TextEmbeddings:
    """
    Manages text embeddings for the vocabulary.
    
    Supports three embedding types:
    1. TF-IDF: Statistical embeddings based on term frequency-inverse document frequency
    2. Word2Vec: Pre-trained embeddings from gensim (if available)
    3. GloVe: Pre-trained embeddings from torchtext (if available)
    
    Attributes:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of embedding vectors
        text_preprocessor (TextPreprocessor): Text preprocessor with vocabulary
        embedding_matrix (np.ndarray): Embedding matrix (vocab_size, embedding_dim)
    """
    
    def __init__(
        self,
        text_preprocessor: TextPreprocessor,
        embedding_dim: int = EMBEDDING_DIM
    ):
        """
        Initialize TextEmbeddings.
        
        Args:
            text_preprocessor: TextPreprocessor with built vocabulary
            embedding_dim: Dimension of embedding vectors
        """
        self.text_preprocessor = text_preprocessor
        self.vocab_size = len(text_preprocessor.word2idx)
        self.embedding_dim = embedding_dim
        self.embedding_matrix = None
        
        if not text_preprocessor.is_built:
            raise ValueError("Text preprocessor vocabulary not built")
    
    def create_tfidf_embeddings(
        self,
        captions: List[str],
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Create TF-IDF based embeddings.
        
        Process:
        1. Build TF-IDF vectors for all words in vocabulary
        2. Apply dimensionality reduction (SVD) to target embedding_dim
        3. Initialize special tokens with small random values
        
        Args:
            captions: List of caption strings for computing TF-IDF
            n_components: Target embedding dimension (default: self.embedding_dim)
        
        Returns:
            Embedding matrix (vocab_size, embedding_dim)
        """
        if n_components is None:
            n_components = self.embedding_dim
        
        print("\n" + "=" * 70)
        print("CREATING TF-IDF EMBEDDINGS")
        print("=" * 70)
        
        print(f"\n✓ Parameters:")
        print(f"  - Vocabulary size: {self.vocab_size:,}")
        print(f"  - Embedding dimension: {n_components}")
        print(f"  - Number of captions: {len(captions):,}")
        
        # Clean captions
        cleaned_captions = [
            self.text_preprocessor.clean_caption(caption)
            for caption in captions
        ]
        
        # Create TF-IDF vectorizer
        print(f"\n✓ Building TF-IDF vectors...")
        vectorizer = TfidfVectorizer(
            max_features=self.vocab_size - 4,  # Exclude special tokens
            lowercase=True,
            stop_words=None  # Keep all words
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(cleaned_captions)
        print(f"  - TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Get vocabulary from vectorizer
        tfidf_vocab = vectorizer.vocabulary_
        
        # Apply SVD for dimensionality reduction
        print(f"\n✓ Applying dimensionality reduction (SVD)...")
        n_components = min(n_components, tfidf_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_matrix = svd.fit_transform(tfidf_matrix.T)  # Transpose to get word embeddings
        
        print(f"  - Reduced shape: {reduced_matrix.shape}")
        print(f"  - Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
        
        # Initialize embedding matrix
        embedding_matrix = np.random.randn(self.vocab_size, n_components) * 0.01
        
        # Map TF-IDF embeddings to vocabulary indices
        print(f"\n✓ Mapping embeddings to vocabulary...")
        mapped_count = 0
        for word, tfidf_idx in tfidf_vocab.items():
            if word in self.text_preprocessor.word2idx:
                vocab_idx = self.text_preprocessor.word2idx[word]
                embedding_matrix[vocab_idx] = reduced_matrix[tfidf_idx]
                mapped_count += 1
        
        print(f"  - Mapped {mapped_count:,} words to vocabulary")
        
        # Initialize special tokens with zeros
        embedding_matrix[self.text_preprocessor.PAD_IDX] = 0.0
        embedding_matrix[self.text_preprocessor.SOS_IDX] *= 0.1
        embedding_matrix[self.text_preprocessor.EOS_IDX] *= 0.1
        embedding_matrix[self.text_preprocessor.UNK_IDX] *= 0.1
        
        self.embedding_matrix = embedding_matrix
        
        print(f"\n✓ TF-IDF embeddings created:")
        print(f"  - Shape: {embedding_matrix.shape}")
        print(f"  - Mean: {embedding_matrix.mean():.4f}")
        print(f"  - Std: {embedding_matrix.std():.4f}")
        
        return embedding_matrix
    
    def create_word2vec_embeddings(
        self,
        model_name: str = 'word2vec-google-news-300'
    ) -> np.ndarray:
        """
        Create Word2Vec embeddings using pre-trained model.
        
        Requires gensim-data package:
        pip install gensim
        python -m gensim.downloader --download word2vec-google-news-300
        
        Args:
            model_name: Name of pre-trained Word2Vec model
        
        Returns:
            Embedding matrix (vocab_size, 300)
        """
        print("\n" + "=" * 70)
        print("CREATING WORD2VEC EMBEDDINGS")
        print("=" * 70)
        
        try:
            import gensim.downloader as api
            
            print(f"\n✓ Loading pre-trained Word2Vec model: {model_name}")
            print("  (This may take a few minutes on first run...)")
            
            # Load pre-trained Word2Vec model
            w2v_model = api.load(model_name)
            w2v_dim = w2v_model.vector_size
            
            print(f"  - Model loaded successfully")
            print(f"  - Embedding dimension: {w2v_dim}")
            
            # Initialize embedding matrix with small random values
            embedding_matrix = np.random.randn(self.vocab_size, w2v_dim) * 0.01
            
            # Map Word2Vec embeddings to vocabulary
            print(f"\n✓ Mapping embeddings to vocabulary...")
            mapped_count = 0
            for word, idx in self.text_preprocessor.word2idx.items():
                if idx < 4:  # Skip special tokens
                    continue
                
                try:
                    # Get Word2Vec embedding
                    embedding_matrix[idx] = w2v_model[word]
                    mapped_count += 1
                except KeyError:
                    # Word not in Word2Vec vocabulary, keep random initialization
                    pass
            
            print(f"  - Mapped {mapped_count:,}/{self.vocab_size - 4:,} words ({100*mapped_count/(self.vocab_size-4):.1f}%)")
            
            # Initialize special tokens
            embedding_matrix[self.text_preprocessor.PAD_IDX] = 0.0
            embedding_matrix[self.text_preprocessor.SOS_IDX] *= 0.1
            embedding_matrix[self.text_preprocessor.EOS_IDX] *= 0.1
            embedding_matrix[self.text_preprocessor.UNK_IDX] *= 0.1
            
            self.embedding_matrix = embedding_matrix
            
            print(f"\n✓ Word2Vec embeddings created:")
            print(f"  - Shape: {embedding_matrix.shape}")
            print(f"  - Mean: {embedding_matrix.mean():.4f}")
            print(f"  - Std: {embedding_matrix.std():.4f}")
            
            return embedding_matrix
            
        except ImportError:
            print("\n❌ Error: gensim not installed")
            print("   Install with: pip install gensim")
            raise
        except Exception as e:
            print(f"\n❌ Error loading Word2Vec: {e}")
            raise
    
    def create_glove_embeddings(
        self,
        glove_file: Optional[Path] = None,
        embedding_dim: int = 300
    ) -> np.ndarray:
        """
        Create GloVe embeddings using pre-trained vectors.
        
        Download GloVe from: https://nlp.stanford.edu/projects/glove/
        Example: glove.6B.300d.txt (Common Crawl 840B tokens)
        
        Args:
            glove_file: Path to GloVe text file (e.g., glove.6B.300d.txt)
            embedding_dim: Dimension of GloVe embeddings
        
        Returns:
            Embedding matrix (vocab_size, embedding_dim)
        """
        print("\n" + "=" * 70)
        print("CREATING GLOVE EMBEDDINGS")
        print("=" * 70)
        
        if glove_file is None:
            glove_file = EMBEDDINGS_DIR / f"glove.6B.{embedding_dim}d.txt"
        
        glove_file = Path(glove_file)
        
        if not glove_file.exists():
            print(f"\n❌ Error: GloVe file not found: {glove_file}")
            print("\nDownload GloVe embeddings from:")
            print("  https://nlp.stanford.edu/projects/glove/")
            print(f"\nPlace the file at: {glove_file}")
            raise FileNotFoundError(f"GloVe file not found: {glove_file}")
        
        print(f"\n✓ Loading GloVe embeddings from: {glove_file.name}")
        
        # Load GloVe embeddings into dictionary
        glove_embeddings = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                glove_embeddings[word] = vector
        
        print(f"  - Loaded {len(glove_embeddings):,} GloVe vectors")
        print(f"  - Embedding dimension: {embedding_dim}")
        
        # Initialize embedding matrix
        embedding_matrix = np.random.randn(self.vocab_size, embedding_dim) * 0.01
        
        # Map GloVe embeddings to vocabulary
        print(f"\n✓ Mapping embeddings to vocabulary...")
        mapped_count = 0
        for word, idx in self.text_preprocessor.word2idx.items():
            if idx < 4:  # Skip special tokens
                continue
            
            if word in glove_embeddings:
                embedding_matrix[idx] = glove_embeddings[word]
                mapped_count += 1
        
        print(f"  - Mapped {mapped_count:,}/{self.vocab_size - 4:,} words ({100*mapped_count/(self.vocab_size-4):.1f}%)")
        
        # Initialize special tokens
        embedding_matrix[self.text_preprocessor.PAD_IDX] = 0.0
        embedding_matrix[self.text_preprocessor.SOS_IDX] *= 0.1
        embedding_matrix[self.text_preprocessor.EOS_IDX] *= 0.1
        embedding_matrix[self.text_preprocessor.UNK_IDX] *= 0.1
        
        self.embedding_matrix = embedding_matrix
        
        print(f"\n✓ GloVe embeddings created:")
        print(f"  - Shape: {embedding_matrix.shape}")
        print(f"  - Mean: {embedding_matrix.mean():.4f}")
        print(f"  - Std: {embedding_matrix.std():.4f}")
        
        return embedding_matrix
    
    def save_embeddings(
        self,
        filepath: Path,
        embedding_type: str = 'custom'
    ) -> None:
        """
        Save embedding matrix to file.
        
        Args:
            filepath: Path to save embeddings
            embedding_type: Type of embeddings ('tfidf', 'word2vec', 'glove', 'custom')
        """
        if self.embedding_matrix is None:
            raise ValueError("No embeddings to save. Create embeddings first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy array
        np.save(filepath, self.embedding_matrix)
        
        # Save metadata
        metadata = {
            'embedding_type': embedding_type,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_matrix.shape[1],
            'shape': self.embedding_matrix.shape,
            'mean': float(self.embedding_matrix.mean()),
            'std': float(self.embedding_matrix.std())
        }
        
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Embeddings saved:")
        print(f"  - Matrix: {filepath}")
        print(f"  - Metadata: {metadata_file}")
    
    def load_embeddings(self, filepath: Path) -> np.ndarray:
        """
        Load embedding matrix from file.
        
        Args:
            filepath: Path to embedding file (.npy)
        
        Returns:
            Loaded embedding matrix
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        
        self.embedding_matrix = np.load(filepath)
        
        print(f"\n✓ Embeddings loaded from: {filepath}")
        print(f"  - Shape: {self.embedding_matrix.shape}")
        
        return self.embedding_matrix
    
    def get_pytorch_embedding_layer(
        self,
        freeze: bool = False
    ) -> nn.Embedding:
        """
        Create PyTorch Embedding layer from embedding matrix.
        
        Args:
            freeze: If True, freeze embeddings (no gradient updates)
        
        Returns:
            PyTorch Embedding layer
        """
        if self.embedding_matrix is None:
            raise ValueError("No embeddings available. Create or load embeddings first.")
        
        # Create embedding layer
        vocab_size, embedding_dim = self.embedding_matrix.shape
        embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initialize with pre-trained weights
        embedding_layer.weight.data.copy_(torch.from_numpy(self.embedding_matrix))
        
        # Freeze if requested
        if freeze:
            embedding_layer.weight.requires_grad = False
        
        return embedding_layer


def create_all_embeddings(
    text_preprocessor: TextPreprocessor,
    captions: List[str],
    output_dir: Path = EMBEDDINGS_DIR,
    create_word2vec: bool = False,
    create_glove: bool = False
) -> Dict[str, np.ndarray]:
    """
    Create all embedding types and save to disk.
    
    Args:
        text_preprocessor: TextPreprocessor with built vocabulary
        captions: List of captions for TF-IDF computation
        output_dir: Directory to save embeddings
        create_word2vec: Whether to create Word2Vec embeddings (requires gensim)
        create_glove: Whether to create GloVe embeddings (requires GloVe file)
    
    Returns:
        Dictionary mapping embedding type to embedding matrix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings = {}
    
    # Create TF-IDF embeddings (always available)
    print("\n" + "=" * 70)
    print("CREATING EMBEDDINGS")
    print("=" * 70)
    
    text_embeddings = TextEmbeddings(text_preprocessor, embedding_dim=EMBEDDING_DIM)
    
    # 1. TF-IDF
    tfidf_matrix = text_embeddings.create_tfidf_embeddings(captions)
    tfidf_path = output_dir / "tfidf_embeddings.npy"
    text_embeddings.save_embeddings(tfidf_path, embedding_type='tfidf')
    embeddings['tfidf'] = tfidf_matrix
    
    # 2. Word2Vec (optional)
    if create_word2vec:
        try:
            w2v_matrix = text_embeddings.create_word2vec_embeddings()
            w2v_path = output_dir / "word2vec_embeddings.npy"
            text_embeddings.save_embeddings(w2v_path, embedding_type='word2vec')
            embeddings['word2vec'] = w2v_matrix
        except Exception as e:
            print(f"\n⚠ Warning: Could not create Word2Vec embeddings: {e}")
    
    # 3. GloVe (optional)
    if create_glove:
        try:
            glove_matrix = text_embeddings.create_glove_embeddings()
            glove_path = output_dir / "glove_embeddings.npy"
            text_embeddings.save_embeddings(glove_path, embedding_type='glove')
            embeddings['glove'] = glove_matrix
        except Exception as e:
            print(f"\n⚠ Warning: Could not create GloVe embeddings: {e}")
    
    return embeddings


def main():
    """
    Main function to create embeddings.
    """
    from utils.constants import PROCESSED_DIR, ARTEMIS_CSV
    import pandas as pd
    
    # Load vocabulary
    vocab_path = PROCESSED_DIR / "vocabulary.json"
    if not vocab_path.exists():
        print(f"❌ Error: {vocab_path} not found")
        print("   Run utils/text_preprocessing.py first")
        return
    
    print("Loading vocabulary...")
    text_preprocessor = TextPreprocessor()
    text_preprocessor.load_vocabulary(vocab_path)
    
    # Load captions for TF-IDF
    selected_images_path = PROCESSED_DIR / "selected_images.json"
    with open(selected_images_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    selected_paintings = {item['painting'] for item in data['selected_paintings']}
    
    print(f"\nLoading captions from ArtEmis CSV...")
    df = pd.read_csv(ARTEMIS_CSV)
    captions = df[df['painting'].isin(selected_paintings)]['utterance'].tolist()
    
    print(f"  - Loaded {len(captions):,} captions")
    
    # Create embeddings
    embeddings = create_all_embeddings(
        text_preprocessor=text_preprocessor,
        captions=captions,
        output_dir=EMBEDDINGS_DIR,
        create_word2vec=False,  # Set to True if gensim is installed
        create_glove=False  # Set to True if GloVe file is available
    )
    
    print("\n" + "=" * 70)
    print("✅ EMBEDDINGS CREATION COMPLETE")
    print("=" * 70)
    print(f"\nCreated embeddings:")
    for emb_type, matrix in embeddings.items():
        print(f"  - {emb_type}: {matrix.shape}")


if __name__ == "__main__":
    main()
