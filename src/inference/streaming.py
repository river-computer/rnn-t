"""Streaming inference for real-time EMG-to-phoneme decoding.

Provides:
- StreamingDecoder: Processes EMG chunks and emits phonemes in real-time
- StreamingState: Maintains decoder state across chunks
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import json

import numpy as np
import torch
from torch import Tensor

from ..data import EMGPreprocessor
from ..model import Transducer, build_transducer


@dataclass
class StreamingState:
    """State maintained across streaming chunks."""

    # Encoder KV cache (one tuple per transformer layer)
    encoder_kv_cache: List[Optional[Tuple[Tensor, Tensor]]] = field(default_factory=list)

    # Predictor LSTM hidden state
    predictor_hidden: Optional[Tuple[Tensor, Tensor]] = None

    # Last emitted token (for predictor input)
    last_token: int = 2  # <sos> token

    # Full sequence of emitted phonemes
    emitted_sequence: List[int] = field(default_factory=list)

    # Buffer for incomplete frames (less than frame_length)
    emg_buffer: Optional[np.ndarray] = None

    def reset(self):
        """Reset state for new utterance."""
        self.encoder_kv_cache = []
        self.predictor_hidden = None
        self.last_token = 2
        self.emitted_sequence = []
        self.emg_buffer = None


class StreamingDecoder:
    """Real-time streaming decoder for EMG-to-phoneme.

    Usage:
        decoder = StreamingDecoder.from_checkpoint('model.pt')
        state = decoder.init_state()

        # Process EMG chunks (e.g., 160ms each)
        for chunk in emg_stream:
            phonemes, state = decoder.process_chunk(chunk, state)
            print(f"New phonemes: {phonemes}")
    """

    def __init__(
        self,
        model: Transducer,
        preprocessor: EMGPreprocessor,
        vocab: Dict[str, int],
        session_id: int = 0,
        device: torch.device = None,
        chunk_size_ms: float = 160.0,
        max_symbols_per_step: int = 10,
    ):
        """Initialize streaming decoder.

        Args:
            model: Trained Transducer model
            preprocessor: EMG signal preprocessor
            vocab: Phoneme vocabulary mapping
            session_id: Recording session ID (for session embedding)
            device: Inference device (CPU/MPS/CUDA)
            chunk_size_ms: Expected chunk size in milliseconds
            max_symbols_per_step: Max symbols to emit per encoder frame
        """
        self.device = device or torch.device('cpu')
        self.model = model.to(self.device)
        self.model.eval()

        self.preprocessor = preprocessor
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.session_id = session_id
        self.chunk_size_ms = chunk_size_ms
        self.max_symbols_per_step = max_symbols_per_step

        # Session ID tensor (reused)
        self._session_tensor = torch.tensor([session_id], device=self.device)

        # Special token IDs
        self.blank_id = vocab.get('<blank>', 0)
        self.sos_id = vocab.get('<sos>', 2)
        self.eos_id = vocab.get('<eos>', 3)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        vocab_path: str = None,
        device: torch.device = None,
        **kwargs,
    ) -> 'StreamingDecoder':
        """Create decoder from saved checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            vocab_path: Path to vocabulary JSON (optional if in checkpoint)
            device: Inference device
            **kwargs: Additional arguments for StreamingDecoder

        Returns:
            Initialized StreamingDecoder
        """
        device = device or torch.device('cpu')

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})

        # Build model
        model = build_transducer(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Build preprocessor
        data_config = config.get('data', {})
        preprocessor = EMGPreprocessor(
            sample_rate=data_config.get('emg_sample_rate', 1000),
            target_rate=data_config.get('target_sample_rate', 689),
            notch_freq=data_config.get('notch_freq', 60),
            bandpass_low=data_config.get('bandpass_low', 20),
            bandpass_high=data_config.get('bandpass_high', 450),
            frame_length_ms=data_config.get('frame_length_ms', 8),
            frame_shift_ms=data_config.get('frame_shift_ms', 4),
        )

        # Load vocab
        if vocab_path:
            with open(vocab_path) as f:
                vocab_data = json.load(f)
            vocab = {**vocab_data['special_tokens'], **vocab_data['phonemes']}
        else:
            # Default vocab
            vocab = {'<blank>': 0, '<sil>': 1, '<sos>': 2, '<eos>': 3}
            phonemes = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
                       'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
                       'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
                       'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
            for i, p in enumerate(phonemes):
                vocab[p] = i + 4

        return cls(
            model=model,
            preprocessor=preprocessor,
            vocab=vocab,
            device=device,
            **kwargs,
        )

    def init_state(self) -> StreamingState:
        """Initialize fresh state for new utterance.

        Returns:
            Initial StreamingState
        """
        state = StreamingState()
        state.encoder_kv_cache = [None] * self.model.encoder.num_layers
        state.predictor_hidden = self.model.predictor.init_hidden(1, self.device)
        state.last_token = self.sos_id
        return state

    @torch.no_grad()
    def process_chunk(
        self,
        emg_chunk: np.ndarray,
        state: StreamingState,
        session_id: Optional[int] = None,
    ) -> Tuple[List[str], StreamingState]:
        """Process a chunk of raw EMG and return new phonemes.

        Args:
            emg_chunk: (samples, channels) raw EMG at original sample rate
            state: Current streaming state
            session_id: Optional session ID override

        Returns:
            new_phonemes: List of phoneme strings emitted in this chunk
            state: Updated streaming state
        """
        # Handle buffering for incomplete frames
        if state.emg_buffer is not None:
            emg_chunk = np.concatenate([state.emg_buffer, emg_chunk], axis=0)
            state.emg_buffer = None

        # Preprocess chunk
        session = session_id if session_id is not None else self.session_id
        session_name = f"5-{4 + session}"  # Map to session name

        try:
            processed = self.preprocessor.process(emg_chunk, session_name)
        except ValueError:
            # Chunk too short, buffer it
            state.emg_buffer = emg_chunk
            return [], state

        # If chunk is too short after processing, buffer remaining
        if processed.shape[0] < 4:  # Need at least 4 frames for subsample
            state.emg_buffer = emg_chunk
            return [], state

        # Convert to tensor
        emg_tensor = torch.from_numpy(processed).float().unsqueeze(0).to(self.device)

        # Get session tensor
        session_tensor = self._session_tensor if session_id is None else torch.tensor([session_id], device=self.device)

        # Run encoder with KV cache
        encoder_out, new_kv_cache = self.model.encoder.forward_streaming(
            emg_tensor, session_tensor, state.encoder_kv_cache
        )
        state.encoder_kv_cache = new_kv_cache

        # Decode each encoder frame
        new_phoneme_ids = []
        T = encoder_out.shape[1]

        for t in range(T):
            encoder_frame = encoder_out[:, t:t+1, :]  # (1, 1, dim)

            for _ in range(self.max_symbols_per_step):
                # Get predictor output
                last_token_tensor = torch.tensor([[state.last_token]], device=self.device)
                predictor_out, state.predictor_hidden = self.model.predictor.forward_step(
                    last_token_tensor.squeeze(1), state.predictor_hidden
                )

                # Compute joint logits
                logits = self.model.joiner.forward_single(encoder_frame, predictor_out)

                # Greedy selection
                token = logits.argmax(dim=-1).item()

                if token == self.blank_id:
                    # Blank - move to next encoder frame
                    break

                # Emit phoneme
                new_phoneme_ids.append(token)
                state.emitted_sequence.append(token)
                state.last_token = token

        # Convert IDs to phoneme strings
        new_phonemes = [self.reverse_vocab.get(pid, f'<UNK:{pid}>') for pid in new_phoneme_ids]

        return new_phonemes, state

    def get_full_sequence(self, state: StreamingState) -> List[str]:
        """Get full emitted sequence as phoneme strings.

        Args:
            state: Current streaming state

        Returns:
            List of phoneme strings
        """
        return [self.reverse_vocab.get(pid, f'<UNK:{pid}>') for pid in state.emitted_sequence]

    def reset(self, state: StreamingState) -> StreamingState:
        """Reset state for new utterance.

        Args:
            state: State to reset

        Returns:
            Reset state
        """
        state.reset()
        state.encoder_kv_cache = [None] * self.model.encoder.num_layers
        state.predictor_hidden = self.model.predictor.init_hidden(1, self.device)
        state.last_token = self.sos_id
        return state


def simulate_streaming(
    decoder: StreamingDecoder,
    emg: np.ndarray,
    chunk_samples: int = 160,
    sample_rate: int = 1000,
) -> List[str]:
    """Simulate streaming inference on full EMG sequence.

    Useful for testing streaming decoder behavior.

    Args:
        decoder: StreamingDecoder instance
        emg: (samples, channels) full EMG sequence
        chunk_samples: Samples per chunk
        sample_rate: EMG sample rate

    Returns:
        Full decoded phoneme sequence
    """
    state = decoder.init_state()
    all_phonemes = []

    # Process in chunks
    for start in range(0, emg.shape[0], chunk_samples):
        end = min(start + chunk_samples, emg.shape[0])
        chunk = emg[start:end]

        phonemes, state = decoder.process_chunk(chunk, state)
        all_phonemes.extend(phonemes)

        if phonemes:
            print(f"Chunk [{start}:{end}]: {' '.join(phonemes)}")

    return all_phonemes
