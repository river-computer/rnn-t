"""PyTorch Dataset for EMG data with phoneme labels."""

import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .preprocessing import EMGPreprocessor, SessionStats
from .alignment import AlignmentLoader, Vocabulary, Alignment


@dataclass
class EMGSample:
    """A single EMG sample with labels."""
    utterance_id: str
    session_id: str
    emg: np.ndarray           # (num_frames, num_channels)
    frame_labels: np.ndarray  # (num_frames,) for CTC
    sequence: np.ndarray      # (seq_len,) for RNN-T
    text: str


class EMGDataset(Dataset):
    """Dataset for EMG silent speech data."""

    def __init__(
        self,
        data_dir: str,
        alignments_dir: str,
        vocab_path: str,
        sessions: list[str],
        split: str = "voiced",  # "voiced", "silent", "closed_vocab"
        preprocessor: Optional[EMGPreprocessor] = None,
        session_stats: Optional[dict[str, SessionStats]] = None,
        max_samples: Optional[int] = None,
    ):
        """Initialize the dataset.

        Args:
            data_dir: Path to extracted EMG data directory
            alignments_dir: Path to extracted TextGrid alignments
            vocab_path: Path to vocab.json
            sessions: List of session IDs to include (e.g., ["5-4", "5-5"])
            split: Data split ("voiced", "silent", "closed_vocab")
            preprocessor: EMG preprocessor (created if None)
            session_stats: Pre-computed session normalization stats
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.vocab = Vocabulary(vocab_path)
        self.sessions = sessions
        self.split = split

        # Initialize preprocessor
        self.preprocessor = preprocessor or EMGPreprocessor()

        # Initialize alignment loader
        self.alignment_loader = AlignmentLoader(
            alignments_dir=alignments_dir,
            vocab=self.vocab,
        )

        # Load or compute session stats
        self.session_stats = session_stats or {}

        # Discover samples
        self.samples = self._discover_samples(max_samples)

        # Map session IDs to indices for embedding
        self.session_to_idx = {s: i for i, s in enumerate(sorted(set(
            s.session_id for s in self.samples
        )))}

    def _discover_samples(self, max_samples: Optional[int]) -> list[EMGSample]:
        """Discover all valid samples in the dataset."""
        samples = []

        # Determine subdirectory based on split
        if self.split == "voiced":
            split_dir = self.data_dir / "voiced_parallel_data"
        elif self.split == "silent":
            split_dir = self.data_dir / "silent_parallel_data"
        else:
            split_dir = self.data_dir / "closed_vocab" / "voiced"

        for session_id in self.sessions:
            session_dir = split_dir / session_id
            if self.split == "silent":
                session_dir = split_dir / f"{session_id}_silent"

            if not session_dir.exists():
                print(f"Session directory not found: {session_dir}")
                continue

            # Find all EMG files in session
            emg_files = list(session_dir.glob("*_emg.npy"))

            # Collect all EMG for session stats if not pre-computed
            if session_id not in self.session_stats:
                session_emg_list = []
                for emg_file in emg_files:
                    emg = np.load(emg_file)
                    session_emg_list.append(emg)

                if session_emg_list:
                    self.session_stats[session_id] = self.preprocessor.compute_session_stats(
                        session_emg_list, session_id
                    )

            for emg_file in emg_files:
                utterance_id = emg_file.stem.replace("_emg", "")
                info_file = session_dir / f"{utterance_id}_info.json"

                if not info_file.exists():
                    continue

                # Load metadata
                with open(info_file) as f:
                    info = json.load(f)

                text = info.get("text", "")

                # Skip empty transcripts for voiced data
                if self.split == "voiced" and not text:
                    continue

                samples.append(EMGSample(
                    utterance_id=utterance_id,
                    session_id=session_id,
                    emg=None,  # Loaded on demand
                    frame_labels=None,
                    sequence=None,
                    text=text,
                ))

                if max_samples and len(samples) >= max_samples:
                    return samples

        print(f"Discovered {len(samples)} samples for split={self.split}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample by index.

        Returns:
            Dictionary with:
                - emg: (num_frames, num_channels) tensor
                - frame_labels: (num_frames,) tensor for CTC
                - sequence: (seq_len,) tensor for RNN-T
                - sequence_length: int
                - session_idx: int
                - utterance_id: str
        """
        sample = self.samples[idx]

        # Determine file paths
        if self.split == "voiced":
            split_dir = self.data_dir / "voiced_parallel_data"
        elif self.split == "silent":
            split_dir = self.data_dir / "silent_parallel_data"
        else:
            split_dir = self.data_dir / "closed_vocab" / "voiced"

        session_dir = split_dir / sample.session_id
        if self.split == "silent":
            session_dir = split_dir / f"{sample.session_id}_silent"

        emg_path = session_dir / f"{sample.utterance_id}_emg.npy"

        # Load and preprocess EMG
        raw_emg = np.load(emg_path)
        stats = self.session_stats.get(sample.session_id)
        emg = self.preprocessor.process_for_model(raw_emg, stats)
        num_frames = len(emg)

        # Get alignment
        alignment_path = self.alignment_loader.find_alignment(
            sample.utterance_id, sample.session_id
        )

        if alignment_path:
            alignment = self.alignment_loader.load_textgrid(alignment_path)
            if alignment:
                frame_labels = self.alignment_loader.alignment_to_frame_labels(
                    alignment, num_frames
                )
                sequence = self.alignment_loader.alignment_to_sequence(alignment)
            else:
                # Fallback: all silence
                frame_labels = np.full(num_frames, self.vocab.sil_id, dtype=np.int64)
                sequence = np.array([self.vocab.eos_id], dtype=np.int64)
        else:
            # No alignment available
            frame_labels = np.full(num_frames, self.vocab.sil_id, dtype=np.int64)
            sequence = np.array([self.vocab.eos_id], dtype=np.int64)

        return {
            "emg": torch.from_numpy(emg).float(),
            "frame_labels": torch.from_numpy(frame_labels).long(),
            "sequence": torch.from_numpy(sequence).long(),
            "sequence_length": len(sequence),
            "session_idx": self.session_to_idx.get(sample.session_id, 0),
            "utterance_id": sample.utterance_id,
            "num_frames": num_frames,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for DataLoader.

    Pads sequences to the maximum length in the batch.

    Returns:
        Dictionary with batched tensors:
            - emg: (batch, max_frames, num_channels)
            - emg_lengths: (batch,)
            - frame_labels: (batch, max_frames)
            - sequences: (batch, max_seq_len)
            - sequence_lengths: (batch,)
            - session_indices: (batch,)
    """
    # Sort by EMG length (descending) for efficient packing
    batch = sorted(batch, key=lambda x: x["num_frames"], reverse=True)

    # Pad EMG
    emg_list = [item["emg"] for item in batch]
    emg_padded = pad_sequence(emg_list, batch_first=True, padding_value=0.0)
    emg_lengths = torch.tensor([item["num_frames"] for item in batch])

    # Pad frame labels
    frame_labels_list = [item["frame_labels"] for item in batch]
    frame_labels_padded = pad_sequence(
        frame_labels_list, batch_first=True, padding_value=0
    )

    # Pad sequences
    sequence_list = [item["sequence"] for item in batch]
    sequences_padded = pad_sequence(
        sequence_list, batch_first=True, padding_value=0
    )
    sequence_lengths = torch.tensor([item["sequence_length"] for item in batch])

    # Session indices
    session_indices = torch.tensor([item["session_idx"] for item in batch])

    return {
        "emg": emg_padded,
        "emg_lengths": emg_lengths,
        "frame_labels": frame_labels_padded,
        "sequences": sequences_padded,
        "sequence_lengths": sequence_lengths,
        "session_indices": session_indices,
    }
