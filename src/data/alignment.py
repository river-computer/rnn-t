"""TextGrid alignment parsing and phoneme label generation.

Loads pre-computed phoneme alignments from Montreal Forced Aligner
and converts them to frame-level and sequence-level labels.
"""

import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np

try:
    from praatio import textgrid
except ImportError:
    textgrid = None


@dataclass
class PhonemeInterval:
    """A single phoneme interval from alignment."""
    start_sec: float
    end_sec: float
    phoneme: str


@dataclass
class Alignment:
    """Full alignment for an utterance."""
    utterance_id: str
    intervals: list[PhonemeInterval]
    text: str


class Vocabulary:
    """Phoneme vocabulary with special tokens."""

    def __init__(self, vocab_path: str):
        with open(vocab_path) as f:
            data = json.load(f)

        self.special_tokens = data["special_tokens"]
        self.phonemes = data["phonemes"]
        self.vocab_size = data["vocab_size"]
        self.blank_id = data["blank_id"]
        self.sil_id = data["sil_id"]
        self.sos_id = data["sos_id"]
        self.eos_id = data["eos_id"]

        # Build reverse mapping
        self.id_to_token = {}
        for token, idx in self.special_tokens.items():
            self.id_to_token[idx] = token
        for token, idx in self.phonemes.items():
            self.id_to_token[idx] = token

        # Build forward mapping
        self.token_to_id = {**self.special_tokens, **self.phonemes}

    def encode(self, phoneme: str) -> int:
        """Convert phoneme to ID."""
        # Handle stress markers (e.g., "AH0", "AH1", "AH2" -> "AH")
        base_phoneme = ''.join(c for c in phoneme if not c.isdigit())

        if base_phoneme in self.token_to_id:
            return self.token_to_id[base_phoneme]
        elif phoneme.upper() in self.token_to_id:
            return self.token_to_id[phoneme.upper()]
        else:
            # Unknown phoneme -> silence
            return self.sil_id

    def decode(self, idx: int) -> str:
        """Convert ID to phoneme."""
        return self.id_to_token.get(idx, "<unk>")

    def decode_sequence(self, ids: list[int], remove_special: bool = True) -> list[str]:
        """Decode a sequence of IDs to phonemes."""
        phonemes = []
        for idx in ids:
            token = self.decode(idx)
            if remove_special and token in self.special_tokens:
                continue
            phonemes.append(token)
        return phonemes


class AlignmentLoader:
    """Loads and processes phoneme alignments."""

    def __init__(
        self,
        alignments_dir: str,
        vocab: Vocabulary,
        sample_rate: int = 689,
        frame_shift_ms: float = 4.0,
    ):
        self.alignments_dir = Path(alignments_dir)
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.frame_shift_ms = frame_shift_ms

        # Frame shift in seconds
        self.frame_shift_sec = frame_shift_ms / 1000

    def load_textgrid(self, path: str) -> Optional[Alignment]:
        """Load alignment from TextGrid file.

        Args:
            path: Path to TextGrid file

        Returns:
            Alignment object or None if loading fails
        """
        if textgrid is None:
            raise ImportError("praatio is required for TextGrid parsing")

        try:
            tg = textgrid.openTextgrid(path, includeEmptyIntervals=True)
        except Exception as e:
            print(f"Failed to load TextGrid {path}: {e}")
            return None

        # Get the phoneme tier (usually named "phones" or "phonemes")
        phone_tier = None
        for tier_name in ["phones", "phonemes", "phone"]:
            if tier_name in tg.tierNames:
                phone_tier = tg.getTier(tier_name)
                break

        if phone_tier is None:
            # Try first tier
            phone_tier = tg.getTier(tg.tierNames[0])

        intervals = []
        for interval in phone_tier.entries:
            phoneme = interval.label.strip()
            if phoneme:  # Skip empty intervals
                intervals.append(PhonemeInterval(
                    start_sec=interval.start,
                    end_sec=interval.end,
                    phoneme=phoneme
                ))

        utterance_id = Path(path).stem
        return Alignment(
            utterance_id=utterance_id,
            intervals=intervals,
            text=""
        )

    def alignment_to_frame_labels(
        self,
        alignment: Alignment,
        num_frames: int
    ) -> np.ndarray:
        """Convert alignment to frame-level labels for CTC.

        Args:
            alignment: Phoneme alignment
            num_frames: Number of frames in preprocessed EMG

        Returns:
            Frame-level labels, shape (num_frames,)
        """
        labels = np.full(num_frames, self.vocab.sil_id, dtype=np.int64)

        for interval in alignment.intervals:
            start_frame = int(interval.start_sec / self.frame_shift_sec)
            end_frame = int(interval.end_sec / self.frame_shift_sec)

            # Clamp to valid range
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames))

            phoneme_id = self.vocab.encode(interval.phoneme)
            labels[start_frame:end_frame] = phoneme_id

        return labels

    def alignment_to_sequence(
        self,
        alignment: Alignment,
        add_sos: bool = False,
        add_eos: bool = True
    ) -> np.ndarray:
        """Convert alignment to collapsed phoneme sequence for RNN-T.

        Args:
            alignment: Phoneme alignment
            add_sos: Whether to prepend <sos> token
            add_eos: Whether to append <eos> token

        Returns:
            Phoneme sequence (no repeats, no blanks)
        """
        sequence = []

        if add_sos:
            sequence.append(self.vocab.sos_id)

        prev_phoneme = None
        for interval in alignment.intervals:
            phoneme_id = self.vocab.encode(interval.phoneme)

            # Skip silence tokens in sequence
            if phoneme_id == self.vocab.sil_id:
                continue

            # Skip repeated phonemes
            if phoneme_id != prev_phoneme:
                sequence.append(phoneme_id)
                prev_phoneme = phoneme_id

        if add_eos:
            sequence.append(self.vocab.eos_id)

        return np.array(sequence, dtype=np.int64)

    def find_alignment(self, utterance_id: str, session_id: str) -> Optional[str]:
        """Find TextGrid file for an utterance.

        Args:
            utterance_id: Utterance identifier (e.g., "0", "123")
            session_id: Session identifier (e.g., "5-4")

        Returns:
            Path to TextGrid file or None if not found
        """
        # Try various naming conventions
        patterns = [
            f"{session_id}/{session_id}_{utterance_id}_audio.TextGrid",  # Actual format in dataset
            f"{session_id}/{utterance_id}.TextGrid",
            f"{session_id}_{utterance_id}.TextGrid",
            f"{utterance_id}.TextGrid",
        ]

        for pattern in patterns:
            path = self.alignments_dir / pattern
            if path.exists():
                return str(path)

        return None


def create_ctc_targets(
    frame_labels: np.ndarray,
    blank_id: int = 0
) -> tuple[np.ndarray, int]:
    """Create CTC targets from frame-level labels.

    For CTC, we need the collapsed sequence without blanks,
    but the loss function handles alignment internally.

    Args:
        frame_labels: Frame-level labels, shape (T,)
        blank_id: ID of blank token

    Returns:
        (targets, target_length) for CTC loss
    """
    # Remove consecutive duplicates and blanks
    targets = []
    prev = None
    for label in frame_labels:
        if label != blank_id and label != prev:
            targets.append(label)
            prev = label

    return np.array(targets, dtype=np.int64), len(targets)
