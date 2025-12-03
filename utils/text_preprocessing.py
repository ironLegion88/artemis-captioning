"""
Text Preprocessing Module for ArtEmis Caption Generation

This module handles text preprocessing, vocabulary building, and tokenization
for the ArtEmis dataset captions.

Key Features:
- Build vocabulary from training captions
- Tokenize captions with special tokens (<PAD>, <SOS>, <EOS>, <UNK>)
- Encode/decode between text and token IDs
- Caption cleaning and normalization
- Vocabulary statistics and analysis

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import pickle

from utils.constants import (
    VOCAB_SIZE,
    MAX_CAPTION_LENGTH,
    PROCESSED_DIR
)


class TextPreprocessor:
    """
    Handles text preprocessing and vocabulary management for captions.
    
    Features:
    - Builds vocabulary from training captions
    - Tokenizes text with special tokens
    - Encodes text to token IDs
    - Decodes token IDs back to text
    - Handles unknown words with <UNK> token
    - Pads/truncates to MAX_CAPTION_LENGTH
    
    Special Tokens:
        <PAD>: Padding token (ID: 0)
        <SOS>: Start of sequence token (ID: 1)
        <EOS>: End of sequence token (ID: 2)
        <UNK>: Unknown word token (ID: 3)
    
    Attributes:
        vocab_size (int): Maximum vocabulary size
        max_length (int): Maximum caption length
        word2idx (Dict[str, int]): Word to index mapping
        idx2word (Dict[int, str]): Index to word mapping
        word_freq (Counter): Word frequency counts
    """
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'
    
    # Special token IDs (fixed positions)
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_length: int = MAX_CAPTION_LENGTH
    ):
        """
        Initialize the TextPreprocessor.
        
        Args:
            vocab_size: Maximum vocabulary size (including special tokens)
            max_length: Maximum caption length (for padding/truncation)
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Initialize vocabulary dictionaries
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        
        # Flag to track if vocabulary has been built
        self.is_built = False
    
    def clean_caption(self, caption: str) -> str:
        """
        Clean and normalize a caption string.
        
        Operations:
        - Convert to lowercase
        - Remove extra whitespace
        - Keep only alphanumeric, spaces, and basic punctuation
        - Strip leading/trailing whitespace
        
        Args:
            caption: Raw caption text
        
        Returns:
            Cleaned caption text
        """
        # Convert to lowercase
        caption = caption.lower()
        
        # Remove special characters but keep basic punctuation
        caption = re.sub(r'[^a-z0-9\s.,!?\'-]', '', caption)
        
        # Replace multiple spaces with single space
        caption = re.sub(r'\s+', ' ', caption)
        
        # Strip leading/trailing whitespace
        caption = caption.strip()
        
        return caption
    
    def tokenize(self, caption: str) -> List[str]:
        """
        Tokenize a caption into words.
        
        Args:
            caption: Caption text (should be cleaned first)
        
        Returns:
            List of word tokens
        """
        # Simple whitespace tokenization
        tokens = caption.split()
        return tokens
    
    def build_vocabulary(
        self,
        captions: List[str],
        min_freq: int = 2
    ) -> None:
        """
        Build vocabulary from a list of captions.
        
        Process:
        1. Clean all captions
        2. Tokenize and count word frequencies
        3. Select top (vocab_size - 4) most frequent words
        4. Add special tokens at fixed positions
        5. Create word2idx and idx2word mappings
        
        Args:
            captions: List of caption strings
            min_freq: Minimum word frequency to include (default: 2)
        """
        print(f"\nBuilding vocabulary from {len(captions)} captions...")
        
        # Count word frequencies
        word_counts = Counter()
        
        for caption in captions:
            # Clean and tokenize
            cleaned = self.clean_caption(caption)
            tokens = self.tokenize(cleaned)
            word_counts.update(tokens)
        
        print(f"  - Total unique words: {len(word_counts)}")
        print(f"  - Total word occurrences: {sum(word_counts.values())}")
        
        # Filter by minimum frequency
        filtered_words = {
            word: count 
            for word, count in word_counts.items() 
            if count >= min_freq
        }
        
        print(f"  - Words with freq >= {min_freq}: {len(filtered_words)}")
        
        # Select top (vocab_size - 4) most frequent words
        # -4 because we reserve 4 slots for special tokens
        num_regular_words = self.vocab_size - 4
        most_common = Counter(filtered_words).most_common(num_regular_words)
        
        # Initialize vocabulary with special tokens at fixed positions
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
            self.UNK_TOKEN: self.UNK_IDX
        }
        
        # Add regular words starting from index 4
        for idx, (word, count) in enumerate(most_common, start=4):
            self.word2idx[word] = idx
            self.word_freq[word] = count
        
        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Update actual vocabulary size
        actual_vocab_size = len(self.word2idx)
        
        print(f"  - Final vocabulary size: {actual_vocab_size}")
        print(f"  - Special tokens: {self.PAD_TOKEN}, {self.SOS_TOKEN}, {self.EOS_TOKEN}, {self.UNK_TOKEN}")
        
        self.is_built = True
    
    def encode(
        self,
        caption: str,
        add_special_tokens: bool = True,
        pad: bool = True
    ) -> List[int]:
        """
        Encode a caption to a list of token IDs.
        
        Args:
            caption: Caption text to encode
            add_special_tokens: If True, add <SOS> and <EOS> tokens
            pad: If True, pad to max_length
        
        Returns:
            List of token IDs
        
        Raises:
            ValueError: If vocabulary has not been built
        """
        if not self.is_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        # Clean and tokenize
        cleaned = self.clean_caption(caption)
        tokens = self.tokenize(cleaned)
        
        # Convert tokens to IDs
        token_ids = []
        
        # Add <SOS> token
        if add_special_tokens:
            token_ids.append(self.SOS_IDX)
        
        # Add word tokens
        for token in tokens:
            # Use word ID if in vocabulary, otherwise use <UNK>
            token_id = self.word2idx.get(token, self.UNK_IDX)
            token_ids.append(token_id)
            
            # Stop if we reach max length (leaving room for <EOS>)
            if add_special_tokens and len(token_ids) >= self.max_length - 1:
                break
        
        # Add <EOS> token
        if add_special_tokens:
            token_ids.append(self.EOS_IDX)
        
        # Pad to max_length if requested
        if pad:
            while len(token_ids) < self.max_length:
                token_ids.append(self.PAD_IDX)
            
            # Truncate if too long
            token_ids = token_ids[:self.max_length]
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode a list of token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: If True, skip special tokens in output
        
        Returns:
            Decoded caption text
        """
        if not self.is_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        words = []
        
        for token_id in token_ids:
            # Get word from vocabulary
            word = self.idx2word.get(token_id, self.UNK_TOKEN)
            
            # Skip special tokens if requested
            if skip_special_tokens:
                if word in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                    continue
            
            words.append(word)
        
        # Join words into sentence
        caption = ' '.join(words)
        
        return caption
    
    def encode_batch(
        self,
        captions: List[str],
        add_special_tokens: bool = True,
        pad: bool = True
    ) -> List[List[int]]:
        """
        Encode a batch of captions.
        
        Args:
            captions: List of caption strings
            add_special_tokens: If True, add <SOS> and <EOS> tokens
            pad: If True, pad to max_length
        
        Returns:
            List of token ID lists
        """
        return [
            self.encode(caption, add_special_tokens, pad)
            for caption in captions
        ]
    
    def decode_batch(
        self,
        token_id_lists: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode a batch of token ID lists.
        
        Args:
            token_id_lists: List of token ID lists
            skip_special_tokens: If True, skip special tokens in output
        
        Returns:
            List of decoded captions
        """
        return [
            self.decode(token_ids, skip_special_tokens)
            for token_ids in token_id_lists
        ]
    
    def save_vocabulary(self, filepath: Path) -> None:
        """
        Save vocabulary to a JSON file.
        
        Args:
            filepath: Path to save the vocabulary file
        """
        if not self.is_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        vocab_data = {
            'vocab_size': len(self.word2idx),
            'max_length': self.max_length,
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()},  # JSON keys must be strings
            'word_freq': dict(self.word_freq),
            'special_tokens': {
                'PAD': self.PAD_IDX,
                'SOS': self.SOS_IDX,
                'EOS': self.EOS_IDX,
                'UNK': self.UNK_IDX
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Vocabulary saved to: {filepath}")
    
    def load_vocabulary(self, filepath: Path) -> None:
        """
        Load vocabulary from a JSON file.
        
        Args:
            filepath: Path to the vocabulary file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab_size = vocab_data['vocab_size']
        self.max_length = vocab_data['max_length']
        self.word2idx = vocab_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        self.word_freq = Counter(vocab_data['word_freq'])
        
        self.is_built = True
        
        print(f"\n✓ Vocabulary loaded from: {filepath}")
        print(f"  - Vocabulary size: {self.vocab_size}")
        print(f"  - Max caption length: {self.max_length}")
    
    def get_vocabulary_stats(self) -> Dict:
        """
        Get statistics about the vocabulary.
        
        Returns:
            Dictionary with vocabulary statistics
        """
        if not self.is_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        stats = {
            'vocab_size': len(self.word2idx),
            'max_length': self.max_length,
            'num_special_tokens': 4,
            'num_regular_words': len(self.word2idx) - 4,
            'most_common_words': self.word_freq.most_common(10),
            'total_word_occurrences': sum(self.word_freq.values())
        }
        
        return stats


def build_vocabulary_from_json(
    json_path: Path,
    vocab_size: int = VOCAB_SIZE,
    max_length: int = MAX_CAPTION_LENGTH,
    min_freq: int = 2
) -> TextPreprocessor:
    """
    Build vocabulary from selected_images.json file.
    
    Args:
        json_path: Path to selected_images.json
        vocab_size: Maximum vocabulary size
        max_length: Maximum caption length
        min_freq: Minimum word frequency to include
    
    Returns:
        TextPreprocessor with built vocabulary
    """
    import pandas as pd
    from utils.constants import ARTEMIS_CSV
    
    print("=" * 70)
    print("BUILDING VOCABULARY FROM SELECTED IMAGES")
    print("=" * 70)
    
    # Load selected images data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n✓ Loaded data from: {json_path}")
    print(f"  - Number of images: {data['total_selected']}")
    
    # Get list of selected painting names
    selected_paintings = {item['painting'] for item in data['selected_paintings']}
    print(f"  - Extracted {len(selected_paintings)} painting names")
    
    # Load ArtEmis CSV to get captions
    print(f"\n✓ Loading captions from: {ARTEMIS_CSV}")
    df = pd.read_csv(ARTEMIS_CSV)
    
    # Filter to only selected paintings
    selected_df = df[df['painting'].isin(selected_paintings)]
    print(f"  - Found {len(selected_df)} captions for selected paintings")
    
    # Extract all captions
    all_captions = selected_df['utterance'].tolist()
    print(f"  - Extracted {len(all_captions)} captions")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        vocab_size=vocab_size,
        max_length=max_length
    )
    
    # Build vocabulary
    preprocessor.build_vocabulary(all_captions, min_freq=min_freq)
    
    # Get and display statistics
    stats = preprocessor.get_vocabulary_stats()
    print(f"\nVocabulary Statistics:")
    print(f"  - Vocabulary size: {stats['vocab_size']}")
    print(f"  - Regular words: {stats['num_regular_words']}")
    print(f"  - Special tokens: {stats['num_special_tokens']}")
    print(f"  - Total word occurrences: {stats['total_word_occurrences']}")
    
    print(f"\nTop 10 most common words:")
    for word, count in stats['most_common_words']:
        print(f"  - '{word}': {count}")
    
    return preprocessor


def test_text_preprocessing():
    """
    Test the text preprocessing pipeline.
    """
    print("=" * 70)
    print("TESTING TEXT PREPROCESSING")
    print("=" * 70)
    
    # Sample captions for testing
    sample_captions = [
        "This painting evokes a sense of melancholy and introspection.",
        "The vibrant colors create an atmosphere of joy and celebration.",
        "I feel a deep sadness when looking at this artwork.",
        "The artist's brushstrokes convey movement and energy.",
        "This piece makes me contemplate the beauty of nature."
    ]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(vocab_size=100, max_length=20)
    
    # Build vocabulary from sample captions
    preprocessor.build_vocabulary(sample_captions, min_freq=1)
    
    # Test encoding
    test_caption = "This painting evokes joy and sadness."
    print(f"\nOriginal caption: '{test_caption}'")
    
    encoded = preprocessor.encode(test_caption, add_special_tokens=True, pad=True)
    print(f"Encoded (with padding): {encoded[:15]}... (length: {len(encoded)})")
    
    # Test decoding
    decoded = preprocessor.decode(encoded, skip_special_tokens=True)
    print(f"Decoded: '{decoded}'")
    
    # Test batch encoding/decoding
    batch_encoded = preprocessor.encode_batch(sample_captions[:3])
    print(f"\n✓ Batch encoding successful (3 captions)")
    
    batch_decoded = preprocessor.decode_batch(batch_encoded)
    print(f"✓ Batch decoding successful")
    for i, caption in enumerate(batch_decoded):
        print(f"  {i+1}. {caption}")
    
    # Display vocabulary stats
    stats = preprocessor.get_vocabulary_stats()
    print(f"\nVocabulary Statistics:")
    print(f"  - Vocabulary size: {stats['vocab_size']}")
    print(f"  - Top 5 words: {stats['most_common_words'][:5]}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    # Check if selected_images.json exists
    selected_images_path = PROCESSED_DIR / "selected_images.json"
    
    if selected_images_path.exists():
        # Build vocabulary from actual data
        preprocessor = build_vocabulary_from_json(
            selected_images_path,
            vocab_size=VOCAB_SIZE,
            max_length=MAX_CAPTION_LENGTH,
            min_freq=2
        )
        
        # Save vocabulary
        vocab_path = PROCESSED_DIR / "vocabulary.json"
        preprocessor.save_vocabulary(vocab_path)
        
        # Test encoding/decoding with real captions
        import pandas as pd
        from utils.constants import ARTEMIS_CSV
        
        with open(selected_images_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get first painting name
        first_painting = data['selected_paintings'][0]['painting']
        
        # Load ArtEmis CSV to get caption
        df = pd.read_csv(ARTEMIS_CSV)
        test_caption = df[df['painting'] == first_painting]['utterance'].iloc[0]
        
        print("\n" + "=" * 70)
        print("TESTING WITH REAL CAPTION")
        print("=" * 70)
        print(f"\nOriginal: '{test_caption}'")
        
        encoded = preprocessor.encode(test_caption)
        print(f"Encoded: {encoded[:20]}...")
        
        decoded = preprocessor.decode(encoded)
        print(f"Decoded: '{decoded}'")
        
        print("\n" + "=" * 70)
        print("✅ VOCABULARY BUILD COMPLETE")
        print("=" * 70)
        
    else:
        # Run basic tests
        print(f"⚠ File not found: {selected_images_path}")
        print("Running basic tests instead...\n")
        test_text_preprocessing()
