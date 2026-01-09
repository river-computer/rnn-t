"""EMG signal preprocessing pipeline.

Based on Gaddy et al. preprocessing:
1. Notch filter at 60 Hz (power line interference)
2. Bandpass filter 20-450 Hz (muscle activity band)
3. Downsample to 689 Hz
4. Frame into 8ms windows with 50% overlap
5. Per-session z-score normalization
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SessionStats:
    """Per-session normalization statistics."""
    mean: np.ndarray  # (8,)
    std: np.ndarray   # (8,)
    session_id: str


class EMGPreprocessor:
    """Preprocesses raw EMG signals for model input."""

    def __init__(
        self,
        source_sample_rate: int = 1000,
        target_sample_rate: int = 689,
        notch_freq: float = 60.0,
        bandpass_low: float = 20.0,
        bandpass_high: float = 450.0,
        frame_length_ms: float = 8.0,
        frame_shift_ms: float = 4.0,
        num_channels: int = 8,
    ):
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.notch_freq = notch_freq
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.num_channels = num_channels

        # Pre-compute filter coefficients
        self._init_filters()

        # Frame parameters at target sample rate
        self.frame_length = int(self.target_sample_rate * frame_length_ms / 1000)
        self.frame_shift = int(self.target_sample_rate * frame_shift_ms / 1000)

    def _init_filters(self):
        """Initialize filter coefficients."""
        # Notch filter for 60 Hz power line interference
        # Q factor of 30 gives narrow notch
        self.notch_b, self.notch_a = signal.iirnotch(
            self.notch_freq,
            Q=30.0,
            fs=self.source_sample_rate
        )

        # Bandpass filter for muscle activity
        # Use 4th order Butterworth
        nyquist = self.source_sample_rate / 2
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        self.bandpass_b, self.bandpass_a = signal.butter(
            4, [low, high], btype='band'
        )

    def filter(self, emg: np.ndarray) -> np.ndarray:
        """Apply notch and bandpass filters.

        Args:
            emg: Raw EMG signal, shape (T, num_channels)

        Returns:
            Filtered EMG signal, shape (T, num_channels)
        """
        # Apply filters channel-by-channel
        filtered = np.zeros_like(emg)
        for ch in range(emg.shape[1]):
            # Notch filter
            notched = signal.filtfilt(self.notch_b, self.notch_a, emg[:, ch])
            # Bandpass filter
            filtered[:, ch] = signal.filtfilt(
                self.bandpass_b, self.bandpass_a, notched
            )
        return filtered

    def resample(self, emg: np.ndarray) -> np.ndarray:
        """Resample from source to target sample rate.

        Args:
            emg: EMG signal at source_sample_rate, shape (T, num_channels)

        Returns:
            Resampled EMG signal, shape (T', num_channels)
        """
        if self.source_sample_rate == self.target_sample_rate:
            return emg

        # Use scipy resample for each channel
        num_output_samples = int(
            len(emg) * self.target_sample_rate / self.source_sample_rate
        )
        resampled = signal.resample(emg, num_output_samples, axis=0)
        return resampled.astype(np.float32)

    def frame(self, emg: np.ndarray) -> np.ndarray:
        """Split EMG into overlapping frames.

        Args:
            emg: EMG signal at target_sample_rate, shape (T, num_channels)

        Returns:
            Framed EMG, shape (num_frames, frame_length * num_channels)
        """
        T, C = emg.shape

        # Calculate number of frames
        num_frames = max(1, (T - self.frame_length) // self.frame_shift + 1)

        # Extract frames
        frames = np.zeros((num_frames, self.frame_length * C), dtype=np.float32)
        for i in range(num_frames):
            start = i * self.frame_shift
            end = start + self.frame_length
            if end <= T:
                frame = emg[start:end, :]  # (frame_length, C)
                frames[i] = frame.flatten()  # (frame_length * C,)

        return frames

    def normalize(
        self,
        emg: np.ndarray,
        stats: Optional[SessionStats] = None
    ) -> np.ndarray:
        """Z-score normalize EMG signal.

        Args:
            emg: EMG signal, shape (T, C) or (num_frames, frame_length * C)
            stats: Pre-computed session statistics. If None, compute from input.

        Returns:
            Normalized EMG signal, same shape as input
        """
        if stats is not None:
            # Use pre-computed statistics
            if emg.ndim == 2 and emg.shape[1] == self.num_channels:
                # Shape (T, C)
                return (emg - stats.mean) / (stats.std + 1e-8)
            else:
                # Shape (num_frames, frame_length * C) - normalize per channel
                normalized = emg.copy()
                for ch in range(self.num_channels):
                    # Each channel appears at positions ch, ch+C, ch+2C, ...
                    for offset in range(0, emg.shape[1], self.num_channels):
                        idx = offset + ch
                        if idx < emg.shape[1]:
                            normalized[:, idx] = (
                                emg[:, idx] - stats.mean[ch]
                            ) / (stats.std[ch] + 1e-8)
                return normalized
        else:
            # Compute statistics from input
            mean = emg.mean(axis=0)
            std = emg.std(axis=0)
            return (emg - mean) / (std + 1e-8)

    def compute_session_stats(
        self,
        emg_list: list[np.ndarray],
        session_id: str
    ) -> SessionStats:
        """Compute normalization statistics for a session.

        Args:
            emg_list: List of raw EMG arrays, each shape (T_i, num_channels)
            session_id: Session identifier

        Returns:
            SessionStats with mean and std per channel
        """
        # Concatenate all EMG for the session
        all_emg = np.concatenate(emg_list, axis=0)

        # Compute per-channel statistics
        mean = all_emg.mean(axis=0)  # (C,)
        std = all_emg.std(axis=0)    # (C,)

        return SessionStats(mean=mean, std=std, session_id=session_id)

    def process(
        self,
        emg: np.ndarray,
        stats: Optional[SessionStats] = None
    ) -> np.ndarray:
        """Full preprocessing pipeline.

        Args:
            emg: Raw EMG signal, shape (T, num_channels)
            stats: Session normalization statistics

        Returns:
            Preprocessed EMG, shape (num_frames, frame_length * num_channels)
        """
        # 1. Filter
        filtered = self.filter(emg)

        # 2. Resample
        resampled = self.resample(filtered)

        # 3. Normalize (before framing, per channel)
        normalized = self.normalize(resampled, stats)

        # 4. Frame
        framed = self.frame(normalized)

        return framed

    def process_for_model(
        self,
        emg: np.ndarray,
        stats: Optional[SessionStats] = None
    ) -> np.ndarray:
        """Process EMG for model input (mean across frame instead of flatten).

        This version computes mean across each frame, resulting in (num_frames, 8)
        which is the format expected by the encoder.

        Args:
            emg: Raw EMG signal, shape (T, num_channels)
            stats: Session normalization statistics

        Returns:
            Preprocessed EMG, shape (num_frames, num_channels)
        """
        # 1. Filter
        filtered = self.filter(emg)

        # 2. Resample
        resampled = self.resample(filtered)

        # 3. Normalize
        normalized = self.normalize(resampled, stats)

        # 4. Frame with mean pooling
        T, C = normalized.shape
        num_frames = max(1, (T - self.frame_length) // self.frame_shift + 1)

        frames = np.zeros((num_frames, C), dtype=np.float32)
        for i in range(num_frames):
            start = i * self.frame_shift
            end = start + self.frame_length
            if end <= T:
                frames[i] = normalized[start:end, :].mean(axis=0)

        return frames


def ms_to_frames(
    ms: float,
    sample_rate: int = 689,
    frame_shift_ms: float = 4.0
) -> int:
    """Convert milliseconds to frame index.

    Args:
        ms: Time in milliseconds
        sample_rate: Sample rate after resampling
        frame_shift_ms: Frame shift in milliseconds

    Returns:
        Frame index
    """
    samples = ms * sample_rate / 1000
    frame_shift_samples = sample_rate * frame_shift_ms / 1000
    return int(samples / frame_shift_samples)


def frames_to_ms(
    frame_idx: int,
    sample_rate: int = 689,
    frame_shift_ms: float = 4.0
) -> float:
    """Convert frame index to milliseconds.

    Args:
        frame_idx: Frame index
        sample_rate: Sample rate after resampling
        frame_shift_ms: Frame shift in milliseconds

    Returns:
        Time in milliseconds
    """
    frame_shift_samples = sample_rate * frame_shift_ms / 1000
    samples = frame_idx * frame_shift_samples
    return samples * 1000 / sample_rate
