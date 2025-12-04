"""
Evaluation Metrics for Image Captioning

This module implements comprehensive evaluation metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4 (n-gram precision)
- METEOR (semantic similarity)
- ROUGE-L (longest common subsequence)
- CIDEr (consensus-based evaluation)

Author: ArtEmis Caption Generation Project
Date: December 2025
"""

import os
import sys
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BLEUScore:
    """
    BLEU score calculator for image captioning evaluation.
    
    Used by training scripts to compute BLEU scores during validation.
    """
    
    def __init__(self, idx_to_word: Dict[int, str], word_to_idx: Dict[str, int]):
        """
        Initialize BLEU scorer.
        
        Args:
            idx_to_word: Mapping from token indices to words
            word_to_idx: Mapping from words to token indices
        """
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx
        self.pad_idx = word_to_idx.get('<PAD>', 0)
        self.start_idx = word_to_idx.get('<START>', 1)
        self.end_idx = word_to_idx.get('<END>', 2)
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs to text."""
        words = []
        for idx in token_ids:
            if idx in [self.pad_idx, self.start_idx, self.end_idx]:
                continue
            word = self.idx_to_word.get(idx, '<UNK>')
            words.append(word)
        return ' '.join(words)
    
    def compute_bleu(self, references: List[List[int]], hypothesis: List[int], n: int = 4) -> float:
        """
        Compute BLEU-n score.
        
        Args:
            references: List of reference token ID sequences
            hypothesis: Hypothesis token ID sequence
            n: Maximum n-gram size
        
        Returns:
            BLEU-n score
        """
        # Decode to text
        ref_texts = [self.decode(ref) for ref in references]
        hyp_text = self.decode(hypothesis)
        
        # Tokenize
        ref_tokens = [text.split() for text in ref_texts]
        hyp_tokens = hyp_text.split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        # Compute n-gram precisions
        precisions = []
        for i in range(1, n + 1):
            hyp_ngrams = self._get_ngrams(hyp_tokens, i)
            if not hyp_ngrams:
                precisions.append(0.0)
                continue
            
            # Count matches across all references
            max_counts = Counter()
            for ref in ref_tokens:
                ref_ngrams = self._get_ngrams(ref, i)
                for ngram, count in ref_ngrams.items():
                    max_counts[ngram] = max(max_counts[ngram], count)
            
            # Clipped counts
            clipped = sum(min(count, max_counts.get(ngram, 0)) 
                         for ngram, count in hyp_ngrams.items())
            total = sum(hyp_ngrams.values())
            
            precisions.append(clipped / total if total > 0 else 0.0)
        
        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            return 0.0
        
        log_precisions = [math.log(p) for p in precisions]
        avg_log_precision = sum(log_precisions) / len(log_precisions)
        
        # Brevity penalty
        ref_len = min(len(ref) for ref in ref_tokens)
        hyp_len = len(hyp_tokens)
        
        if hyp_len >= ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / hyp_len)
        
        return bp * math.exp(avg_log_precision)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens."""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def __call__(self, references: List[List[int]], hypothesis: List[int]) -> float:
        """Compute BLEU-4 score (default)."""
        return self.compute_bleu(references, hypothesis, n=4)


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Extract n-grams from a list of tokens.
    
    Args:
        tokens: List of word tokens
        n: Size of n-grams
    
    Returns:
        Counter of n-gram frequencies
    """
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] += 1
    return ngrams


def compute_bleu_score(
    references: List[List[str]],
    hypothesis: List[str],
    max_n: int = 4,
    smoothing: bool = True
) -> Dict[str, float]:
    """
    Compute BLEU scores (1 through max_n) for a single hypothesis.
    
    Args:
        references: List of reference captions (each is list of tokens)
        hypothesis: Hypothesis caption (list of tokens)
        max_n: Maximum n-gram size (default: 4 for BLEU-4)
        smoothing: Whether to apply smoothing for zero counts
    
    Returns:
        Dictionary with BLEU-1 through BLEU-n scores
    """
    scores = {}
    precisions = []
    
    # Compute precision for each n-gram size
    for n in range(1, max_n + 1):
        hyp_ngrams = get_ngrams(hypothesis, n)
        
        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count matches against all references (take max per n-gram)
        max_ref_counts = Counter()
        for ref in references:
            ref_ngrams = get_ngrams(ref, n)
            for ngram in hyp_ngrams:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
        
        # Clipped counts
        clipped_count = sum(min(hyp_ngrams[ngram], max_ref_counts[ngram]) 
                          for ngram in hyp_ngrams)
        total_count = sum(hyp_ngrams.values())
        
        # Precision with optional smoothing
        if total_count == 0:
            precision = 0.0
        elif clipped_count == 0 and smoothing:
            precision = 1.0 / (total_count + 1)  # Add-1 smoothing
        else:
            precision = clipped_count / total_count
        
        precisions.append(precision)
    
    # Brevity penalty
    hyp_len = len(hypothesis)
    ref_lens = [len(ref) for ref in references]
    
    # Choose closest reference length
    closest_ref_len = min(ref_lens, key=lambda x: (abs(x - hyp_len), x))
    
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len >= closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / hyp_len)
    
    # Compute BLEU scores
    for n in range(1, max_n + 1):
        if n == 1:
            score = bp * precisions[0]
        else:
            # Geometric mean of precisions up to n
            log_precisions = [math.log(p) if p > 0 else -float('inf') 
                            for p in precisions[:n]]
            if all(lp > -float('inf') for lp in log_precisions):
                avg_log_precision = sum(log_precisions) / n
                score = bp * math.exp(avg_log_precision)
            else:
                score = 0.0
        
        scores[f'BLEU-{n}'] = score
    
    return scores


def compute_corpus_bleu(
    all_references: List[List[List[str]]],
    all_hypotheses: List[List[str]],
    max_n: int = 4
) -> Dict[str, float]:
    """
    Compute corpus-level BLEU scores.
    
    Args:
        all_references: List of reference lists for each sample
        all_hypotheses: List of hypothesis captions
        max_n: Maximum n-gram size
    
    Returns:
        Dictionary with corpus BLEU-1 through BLEU-n scores
    """
    # Aggregate counts
    total_clipped = [0] * max_n
    total_count = [0] * max_n
    total_hyp_len = 0
    total_ref_len = 0
    
    for references, hypothesis in zip(all_references, all_hypotheses):
        total_hyp_len += len(hypothesis)
        
        # Choose closest reference length
        ref_lens = [len(ref) for ref in references]
        closest_ref_len = min(ref_lens, key=lambda x: (abs(x - len(hypothesis)), x))
        total_ref_len += closest_ref_len
        
        for n in range(1, max_n + 1):
            hyp_ngrams = get_ngrams(hypothesis, n)
            
            # Max counts from references
            max_ref_counts = Counter()
            for ref in references:
                ref_ngrams = get_ngrams(ref, n)
                for ngram in hyp_ngrams:
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
            
            clipped = sum(min(hyp_ngrams[ngram], max_ref_counts[ngram]) 
                         for ngram in hyp_ngrams)
            total_clipped[n-1] += clipped
            total_count[n-1] += sum(hyp_ngrams.values())
    
    # Compute precisions
    precisions = []
    for n in range(max_n):
        if total_count[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(total_clipped[n] / total_count[n])
    
    # Brevity penalty
    if total_hyp_len == 0:
        bp = 0.0
    elif total_hyp_len >= total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_len / total_hyp_len)
    
    # Compute BLEU scores
    scores = {}
    for n in range(1, max_n + 1):
        log_precisions = [math.log(p) if p > 0 else -float('inf') 
                        for p in precisions[:n]]
        if all(lp > -float('inf') for lp in log_precisions):
            avg_log_precision = sum(log_precisions) / n
            score = bp * math.exp(avg_log_precision)
        else:
            score = 0.0
        scores[f'BLEU-{n}'] = score
    
    return scores


def compute_rouge_l(reference: List[str], hypothesis: List[str]) -> float:
    """
    Compute ROUGE-L (Longest Common Subsequence) F1 score.
    
    Args:
        reference: Reference caption tokens
        hypothesis: Hypothesis caption tokens
    
    Returns:
        ROUGE-L F1 score
    """
    if len(reference) == 0 or len(hypothesis) == 0:
        return 0.0
    
    # Compute LCS length using dynamic programming
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    # Compute precision, recall, F1
    precision = lcs_length / len(hypothesis)
    recall = lcs_length / len(reference)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_meteor(
    reference: List[str],
    hypothesis: List[str],
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5
) -> float:
    """
    Compute simplified METEOR score (exact match only).
    
    Full METEOR includes stemming and synonyms (requires WordNet).
    This implementation uses exact word matching.
    
    Args:
        reference: Reference caption tokens
        hypothesis: Hypothesis caption tokens
        alpha: Precision weight (default: 0.9)
        beta: Fragmentation penalty (default: 3.0)
        gamma: Fragmentation weight (default: 0.5)
    
    Returns:
        METEOR score
    """
    if len(hypothesis) == 0 or len(reference) == 0:
        return 0.0
    
    # Find matches (exact)
    ref_set = set(reference)
    hyp_set = set(hypothesis)
    matches = ref_set & hyp_set
    
    if len(matches) == 0:
        return 0.0
    
    # Precision and recall
    precision = len(matches) / len(hypothesis)
    recall = len(matches) / len(reference)
    
    # F-mean with alpha weighting
    if precision == 0 or recall == 0:
        return 0.0
    
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    
    # Fragmentation penalty (simplified)
    # Count chunks of consecutive matches
    chunks = 0
    in_chunk = False
    hyp_positions = {w: i for i, w in enumerate(hypothesis)}
    
    prev_pos = -2
    for word in reference:
        if word in hyp_set:
            pos = hyp_positions.get(word, -1)
            if pos != prev_pos + 1:
                chunks += 1
            prev_pos = pos
    
    if len(matches) > 0:
        fragmentation = chunks / len(matches)
    else:
        fragmentation = 0
    
    penalty = gamma * (fragmentation ** beta)
    
    meteor = f_mean * (1 - penalty)
    return max(0.0, meteor)


def compute_cider_score(
    references: List[List[str]],
    hypothesis: List[str],
    n: int = 4
) -> float:
    """
    Compute simplified CIDEr score for a single sample.
    
    Full CIDEr requires TF-IDF computation across corpus.
    This is a simplified version using n-gram overlap.
    
    Args:
        references: List of reference captions
        hypothesis: Hypothesis caption
        n: Maximum n-gram size
    
    Returns:
        CIDEr-like score
    """
    if len(hypothesis) == 0:
        return 0.0
    
    scores = []
    
    for ng in range(1, n + 1):
        hyp_ngrams = get_ngrams(hypothesis, ng)
        
        if len(hyp_ngrams) == 0:
            scores.append(0.0)
            continue
        
        # Average similarity across references
        ref_scores = []
        for ref in references:
            ref_ngrams = get_ngrams(ref, ng)
            
            if len(ref_ngrams) == 0:
                ref_scores.append(0.0)
                continue
            
            # Cosine similarity of n-gram vectors
            common = set(hyp_ngrams.keys()) & set(ref_ngrams.keys())
            
            if len(common) == 0:
                ref_scores.append(0.0)
                continue
            
            dot_product = sum(hyp_ngrams[k] * ref_ngrams[k] for k in common)
            hyp_norm = math.sqrt(sum(v ** 2 for v in hyp_ngrams.values()))
            ref_norm = math.sqrt(sum(v ** 2 for v in ref_ngrams.values()))
            
            if hyp_norm * ref_norm == 0:
                ref_scores.append(0.0)
            else:
                ref_scores.append(dot_product / (hyp_norm * ref_norm))
        
        scores.append(np.mean(ref_scores) if ref_scores else 0.0)
    
    # Average across n-gram sizes
    return np.mean(scores) if scores else 0.0


class CaptionEvaluator:
    """
    Comprehensive evaluator for image captioning.
    
    Computes multiple metrics and provides detailed analysis.
    """
    
    def __init__(self, tokenize: bool = True):
        """
        Initialize evaluator.
        
        Args:
            tokenize: Whether to tokenize strings (False if already tokenized)
        """
        self.tokenize = tokenize
        self.results = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def evaluate_single(
        self,
        references: List[str],
        hypothesis: str
    ) -> Dict[str, float]:
        """
        Evaluate a single hypothesis against references.
        
        Args:
            references: List of reference caption strings
            hypothesis: Generated caption string
        
        Returns:
            Dictionary of metric scores
        """
        # Tokenize if needed
        if self.tokenize:
            ref_tokens = [self._tokenize(ref) for ref in references]
            hyp_tokens = self._tokenize(hypothesis)
        else:
            ref_tokens = references
            hyp_tokens = hypothesis
        
        # Compute all metrics
        bleu_scores = compute_bleu_score(ref_tokens, hyp_tokens)
        rouge_l = compute_rouge_l(ref_tokens[0], hyp_tokens)  # Use first reference
        meteor = compute_meteor(ref_tokens[0], hyp_tokens)
        cider = compute_cider_score(ref_tokens, hyp_tokens)
        
        metrics = {
            **bleu_scores,
            'ROUGE-L': rouge_l,
            'METEOR': meteor,
            'CIDEr': cider
        }
        
        return metrics
    
    def evaluate_corpus(
        self,
        all_references: List[List[str]],
        all_hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate entire corpus.
        
        Args:
            all_references: List of reference lists for each sample
            all_hypotheses: List of generated captions
        
        Returns:
            Dictionary of corpus-level metric scores
        """
        # Tokenize
        if self.tokenize:
            all_ref_tokens = [[self._tokenize(ref) for ref in refs] 
                             for refs in all_references]
            all_hyp_tokens = [self._tokenize(hyp) for hyp in all_hypotheses]
        else:
            all_ref_tokens = all_references
            all_hyp_tokens = all_hypotheses
        
        # Corpus BLEU
        bleu_scores = compute_corpus_bleu(all_ref_tokens, all_hyp_tokens)
        
        # Average other metrics
        rouge_scores = []
        meteor_scores = []
        cider_scores = []
        
        for refs, hyp in zip(all_ref_tokens, all_hyp_tokens):
            rouge_scores.append(compute_rouge_l(refs[0], hyp))
            meteor_scores.append(compute_meteor(refs[0], hyp))
            cider_scores.append(compute_cider_score(refs, hyp))
        
        metrics = {
            **bleu_scores,
            'ROUGE-L': np.mean(rouge_scores),
            'METEOR': np.mean(meteor_scores),
            'CIDEr': np.mean(cider_scores)
        }
        
        return metrics
    
    def detailed_report(
        self,
        all_references: List[List[str]],
        all_hypotheses: List[str],
        top_k: int = 5
    ) -> str:
        """
        Generate detailed evaluation report.
        
        Args:
            all_references: List of reference lists
            all_hypotheses: List of generated captions
            top_k: Number of best/worst examples to show
        
        Returns:
            Formatted report string
        """
        # Evaluate each sample
        sample_scores = []
        for i, (refs, hyp) in enumerate(zip(all_references, all_hypotheses)):
            scores = self.evaluate_single(refs, hyp)
            sample_scores.append({
                'index': i,
                'hypothesis': hyp,
                'references': refs,
                'scores': scores,
                'avg_bleu': np.mean([scores[f'BLEU-{n}'] for n in range(1, 5)])
            })
        
        # Sort by average BLEU
        sample_scores.sort(key=lambda x: x['avg_bleu'], reverse=True)
        
        # Corpus metrics
        corpus_metrics = self.evaluate_corpus(all_references, all_hypotheses)
        
        # Build report
        report = []
        report.append("=" * 70)
        report.append("CAPTION EVALUATION REPORT")
        report.append("=" * 70)
        
        report.append("\n📊 CORPUS-LEVEL METRICS:")
        report.append("-" * 40)
        for metric, value in corpus_metrics.items():
            report.append(f"  {metric:12s}: {value:.4f}")
        
        report.append(f"\n🏆 TOP {top_k} BEST CAPTIONS:")
        report.append("-" * 40)
        for sample in sample_scores[:top_k]:
            report.append(f"\n  Sample {sample['index']}:")
            report.append(f"    Hypothesis: {sample['hypothesis']}")
            report.append(f"    Reference:  {sample['references'][0]}")
            report.append(f"    BLEU-4: {sample['scores']['BLEU-4']:.4f}")
        
        report.append(f"\n❌ TOP {top_k} WORST CAPTIONS:")
        report.append("-" * 40)
        for sample in sample_scores[-top_k:]:
            report.append(f"\n  Sample {sample['index']}:")
            report.append(f"    Hypothesis: {sample['hypothesis']}")
            report.append(f"    Reference:  {sample['references'][0]}")
            report.append(f"    BLEU-4: {sample['scores']['BLEU-4']:.4f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def test_metrics():
    """Test evaluation metrics with sample data."""
    print("=" * 70)
    print("TESTING EVALUATION METRICS")
    print("=" * 70)
    
    # Sample data
    references = [
        ["the cat sat on the mat"],
        ["a dog is running in the park"],
        ["beautiful sunset over the ocean"]
    ]
    
    hypotheses = [
        "the cat is on the mat",
        "a dog runs in park",
        "sunset on ocean"
    ]
    
    evaluator = CaptionEvaluator(tokenize=True)
    
    print("\n✓ Testing single sample evaluation...")
    single_scores = evaluator.evaluate_single(references[0], hypotheses[0])
    print(f"  Reference: {references[0][0]}")
    print(f"  Hypothesis: {hypotheses[0]}")
    for metric, score in single_scores.items():
        print(f"    {metric}: {score:.4f}")
    
    print("\n✓ Testing corpus evaluation...")
    corpus_scores = evaluator.evaluate_corpus(references, hypotheses)
    for metric, score in corpus_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n✓ Testing detailed report...")
    report = evaluator.detailed_report(references, hypotheses, top_k=2)
    print(report)
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION METRICS TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_metrics()
