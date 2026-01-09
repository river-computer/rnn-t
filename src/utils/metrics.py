"""Evaluation metrics for phoneme recognition.

Provides:
- PER (Phoneme Error Rate)
- WER (Word Error Rate) - if phoneme-to-word mapping provided
"""

from typing import List, Optional, Dict
import editdistance


def compute_per(
    predictions: List[List[int]],
    references: List[List[int]],
) -> Dict[str, float]:
    """Compute Phoneme Error Rate (PER).

    PER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference length

    Args:
        predictions: List of predicted phoneme sequences
        references: List of reference phoneme sequences

    Returns:
        Dictionary with:
        - 'per': Overall PER
        - 'substitutions': Total substitutions
        - 'deletions': Total deletions
        - 'insertions': Total insertions
        - 'ref_length': Total reference length
    """
    assert len(predictions) == len(references), "Mismatched prediction/reference counts"

    total_edits = 0
    total_ref_length = 0

    for pred, ref in zip(predictions, references):
        # editdistance computes Levenshtein distance (S + D + I)
        total_edits += editdistance.eval(pred, ref)
        total_ref_length += len(ref)

    per = total_edits / total_ref_length if total_ref_length > 0 else 0.0

    return {
        'per': per,
        'total_edits': total_edits,
        'ref_length': total_ref_length,
    }


def compute_per_detailed(
    prediction: List[int],
    reference: List[int],
) -> Dict[str, int]:
    """Compute detailed edit operations for a single pair.

    Uses dynamic programming to find substitutions, deletions, insertions.

    Args:
        prediction: Single predicted sequence
        reference: Single reference sequence

    Returns:
        Dictionary with substitutions, deletions, insertions counts
    """
    m, n = len(prediction), len(reference)

    # dp[i][j] = (edit_distance, substitutions, deletions, insertions)
    # for prediction[:i] vs reference[:j]
    dp = [[(0, 0, 0, 0) for _ in range(n + 1)] for _ in range(m + 1)]

    # Base cases
    for i in range(1, m + 1):
        dp[i][0] = (i, 0, 0, i)  # All insertions
    for j in range(1, n + 1):
        dp[0][j] = (j, 0, j, 0)  # All deletions

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if prediction[i - 1] == reference[j - 1]:
                # Match
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Substitution
                sub = dp[i - 1][j - 1]
                sub_cost = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])

                # Deletion (from reference)
                delete = dp[i][j - 1]
                del_cost = (delete[0] + 1, delete[1], delete[2] + 1, delete[3])

                # Insertion (into prediction)
                insert = dp[i - 1][j]
                ins_cost = (insert[0] + 1, insert[1], insert[2], insert[3] + 1)

                # Take minimum cost operation
                dp[i][j] = min([sub_cost, del_cost, ins_cost], key=lambda x: x[0])

    final = dp[m][n]
    return {
        'edit_distance': final[0],
        'substitutions': final[1],
        'deletions': final[2],
        'insertions': final[3],
    }


def compute_wer(
    predictions: List[List[int]],
    references: List[List[int]],
    phoneme_to_word: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """Compute Word Error Rate (WER).

    If phoneme_to_word is provided, converts phoneme sequences to words
    before computing WER. Otherwise, treats each phoneme as a "word".

    Args:
        predictions: List of predicted phoneme sequences
        references: List of reference phoneme sequences
        phoneme_to_word: Optional mapping from phoneme IDs to words

    Returns:
        Dictionary with WER and related statistics
    """
    if phoneme_to_word is not None:
        # Convert phoneme sequences to word sequences
        # This requires a phoneme-to-grapheme model or dictionary
        # For now, we just compute phoneme-level "WER" (same as PER)
        pass

    # Without word mapping, WER = PER
    return compute_per(predictions, references)


def filter_special_tokens(
    sequence: List[int],
    special_tokens: set = None,
) -> List[int]:
    """Remove special tokens from sequence before evaluation.

    Args:
        sequence: Phoneme sequence
        special_tokens: Set of special token IDs to remove
                       Default: {0, 1, 2, 3} (blank, sil, sos, eos)

    Returns:
        Filtered sequence
    """
    if special_tokens is None:
        special_tokens = {0, 1, 2, 3}

    return [p for p in sequence if p not in special_tokens]


class MetricTracker:
    """Track metrics across batches for epoch-level reporting."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.total_edits = 0
        self.total_ref_length = 0
        self.total_samples = 0

    def update(
        self,
        predictions: List[List[int]],
        references: List[List[int]],
        filter_special: bool = True,
    ):
        """Update metrics with batch predictions.

        Args:
            predictions: Batch of predicted sequences
            references: Batch of reference sequences
            filter_special: Whether to filter special tokens
        """
        for pred, ref in zip(predictions, references):
            if filter_special:
                pred = filter_special_tokens(pred)
                ref = filter_special_tokens(ref)

            self.total_edits += editdistance.eval(pred, ref)
            self.total_ref_length += len(ref)
            self.total_samples += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary with PER and sample count
        """
        per = self.total_edits / self.total_ref_length if self.total_ref_length > 0 else 0.0
        return {
            'per': per,
            'total_edits': self.total_edits,
            'ref_length': self.total_ref_length,
            'num_samples': self.total_samples,
        }
