"""Full RNN-Transducer model combining encoder, predictor, and joiner.

Provides:
1. Training forward (full lattice for RNN-T loss)
2. Greedy decoding
3. Streaming inference with state management
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import EMGEncoder
from .predictor import Predictor
from .joiner import Joiner


@dataclass
class StreamingState:
    """State for streaming inference."""
    encoder_kv_cache: List[Optional[Tuple[Tensor, Tensor]]] = field(default_factory=list)
    predictor_hidden: Optional[Tuple[Tensor, Tensor]] = None
    last_token: int = 2  # <sos> token index
    emitted_sequence: List[int] = field(default_factory=list)


class Transducer(nn.Module):
    """Full RNN-Transducer model.

    Combines EMGEncoder, Predictor, and Joiner for end-to-end
    EMG-to-phoneme transduction.
    """

    def __init__(
        self,
        encoder: EMGEncoder,
        predictor: Predictor,
        joiner: Joiner,
        blank_id: int = 0,
        sos_id: int = 2,
        eos_id: int = 3,
    ):
        """Initialize transducer.

        Args:
            encoder: EMG encoder module
            predictor: Prediction network module
            joiner: Joint network module
            blank_id: Index of blank token
            sos_id: Index of start-of-sequence token
            eos_id: Index of end-of-sequence token
        """
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.joiner = joiner

        self.blank_id = blank_id
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(
        self,
        emg: Tensor,
        session_id: Tensor,
        targets: Tensor,
        emg_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass for training.

        Args:
            emg: (batch, time, emg_channels) - preprocessed EMG
            session_id: (batch,) - session indices
            targets: (batch, max_target_len) - target phoneme sequences (no <sos>)
            emg_lengths: (batch,) - EMG sequence lengths
            target_lengths: (batch,) - target sequence lengths

        Returns:
            logits: (batch, T, U+1, vocab_size) - joint logits
            encoder_lengths: (batch,) - encoder output lengths
            predictor_input: (batch, U+1) - predictor input (<sos> + targets)
        """
        batch_size = emg.shape[0]
        device = emg.device

        # Encode EMG
        encoder_out, encoder_lengths = self.encoder(emg, session_id, emg_lengths)

        # Prepare predictor input: [<sos>] + targets (without last token)
        # For RNN-T, predictor sees [sos, y1, y2, ..., y_{U-1}] to predict [y1, y2, ..., yU]
        max_target_len = targets.shape[1]
        predictor_input = torch.full(
            (batch_size, max_target_len + 1),
            self.sos_id,
            dtype=torch.long,
            device=device,
        )
        predictor_input[:, 1:] = targets

        # Run predictor
        predictor_out, _ = self.predictor(predictor_input)

        # Compute joint logits
        logits = self.joiner(encoder_out, predictor_out)

        return logits, encoder_lengths, target_lengths

    @torch.no_grad()
    def decode_greedy(
        self,
        emg: Tensor,
        session_id: Tensor,
        max_symbols_per_step: int = 10,
    ) -> List[List[int]]:
        """Greedy decoding for batch of EMG sequences.

        Args:
            emg: (batch, time, emg_channels) - preprocessed EMG
            session_id: (batch,) - session indices
            max_symbols_per_step: Maximum symbols to emit per encoder frame

        Returns:
            List of decoded phoneme sequences (one per batch item)
        """
        self.eval()
        batch_size = emg.shape[0]
        device = emg.device

        # Encode
        encoder_out, encoder_lengths = self.encoder(emg, session_id)
        T = encoder_out.shape[1]

        # Initialize predictor
        predictor_hidden = self.predictor.init_hidden(batch_size, device)

        # Start with <sos>
        last_tokens = torch.full((batch_size,), self.sos_id, dtype=torch.long, device=device)

        # Get initial predictor output
        predictor_out, predictor_hidden = self.predictor.forward_step(last_tokens, predictor_hidden)

        # Decode
        decoded = [[] for _ in range(batch_size)]

        for t in range(T):
            encoder_frame = encoder_out[:, t:t+1, :]  # (batch, 1, dim)

            for _ in range(max_symbols_per_step):
                # Compute joint
                logits = self.joiner.forward_single(encoder_frame, predictor_out)  # (batch, vocab)

                # Greedy selection
                tokens = logits.argmax(dim=-1)  # (batch,)

                # Check for blanks
                non_blank_mask = tokens != self.blank_id

                if not non_blank_mask.any():
                    # All blanks, move to next encoder frame
                    break

                # Emit non-blank tokens
                for b in range(batch_size):
                    if t < encoder_lengths[b] and tokens[b] != self.blank_id:
                        decoded[b].append(tokens[b].item())

                # Update predictor for non-blank emissions
                if non_blank_mask.any():
                    last_tokens = tokens.clone()
                    predictor_out, predictor_hidden = self.predictor.forward_step(
                        last_tokens, predictor_hidden
                    )

        return decoded

    def init_streaming_state(self, batch_size: int = 1, device: torch.device = None) -> StreamingState:
        """Initialize state for streaming inference.

        Args:
            batch_size: Batch size (usually 1 for streaming)
            device: Device for tensors

        Returns:
            Initial StreamingState
        """
        if device is None:
            device = next(self.parameters()).device

        return StreamingState(
            encoder_kv_cache=[None] * self.encoder.num_layers,
            predictor_hidden=self.predictor.init_hidden(batch_size, device),
            last_token=self.sos_id,
            emitted_sequence=[],
        )

    @torch.no_grad()
    def decode_streaming(
        self,
        emg_chunk: Tensor,
        session_id: Tensor,
        state: StreamingState,
        max_symbols_per_step: int = 10,
    ) -> Tuple[List[int], StreamingState]:
        """Process a chunk of EMG for streaming inference.

        Args:
            emg_chunk: (1, chunk_time, emg_channels) - EMG chunk
            session_id: (1,) - session index
            state: Current streaming state
            max_symbols_per_step: Max symbols per encoder frame

        Returns:
            new_phonemes: List of phonemes emitted in this chunk
            state: Updated streaming state
        """
        self.eval()
        device = emg_chunk.device

        # Encode chunk with KV cache
        encoder_out, new_kv_cache = self.encoder.forward_streaming(
            emg_chunk, session_id, state.encoder_kv_cache
        )
        state.encoder_kv_cache = new_kv_cache

        T = encoder_out.shape[1]
        new_phonemes = []

        # Get current predictor output
        last_token_tensor = torch.tensor([[state.last_token]], device=device)
        predictor_out, state.predictor_hidden = self.predictor.forward_step(
            last_token_tensor.squeeze(1), state.predictor_hidden
        )

        for t in range(T):
            encoder_frame = encoder_out[:, t:t+1, :]  # (1, 1, dim)

            for _ in range(max_symbols_per_step):
                # Compute joint
                logits = self.joiner.forward_single(encoder_frame, predictor_out)  # (1, vocab)

                # Greedy selection
                token = logits.argmax(dim=-1).item()

                if token == self.blank_id:
                    # Blank, move to next encoder frame
                    break

                # Emit phoneme
                new_phonemes.append(token)
                state.emitted_sequence.append(token)
                state.last_token = token

                # Update predictor
                token_tensor = torch.tensor([token], device=device)
                predictor_out, state.predictor_hidden = self.predictor.forward_step(
                    token_tensor, state.predictor_hidden
                )

        return new_phonemes, state


def build_transducer(config: dict) -> Transducer:
    """Build transducer model from config.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        Initialized Transducer model
    """
    model_config = config.get('model', {})
    encoder_config = model_config.get('encoder', {})
    predictor_config = model_config.get('predictor', {})
    joiner_config = model_config.get('joiner', {})

    encoder = EMGEncoder(
        emg_channels=encoder_config.get('emg_channels', 8),
        num_sessions=encoder_config.get('num_sessions', 8),
        session_embed_dim=encoder_config.get('session_embed_dim', 32),
        conv_dim=encoder_config.get('conv_dim', 768),
        num_conv_blocks=encoder_config.get('num_conv_blocks', 3),
        d_model=encoder_config.get('d_model', 768),
        nhead=encoder_config.get('nhead', 8),
        num_layers=encoder_config.get('num_layers', 6),
        dim_feedforward=encoder_config.get('dim_feedforward', 2048),
        output_dim=encoder_config.get('output_dim', 128),
        dropout=encoder_config.get('dropout', 0.1),
    )

    predictor = Predictor(
        vocab_size=predictor_config.get('vocab_size', 43),
        embed_dim=predictor_config.get('embed_dim', 128),
        hidden_dim=predictor_config.get('hidden_dim', 320),
        output_dim=predictor_config.get('output_dim', 128),
        num_layers=predictor_config.get('num_layers', 1),
        dropout=predictor_config.get('dropout', 0.1),
    )

    joiner = Joiner(
        input_dim=joiner_config.get('input_dim', 128),
        vocab_size=joiner_config.get('vocab_size', 43),
    )

    return Transducer(
        encoder=encoder,
        predictor=predictor,
        joiner=joiner,
        blank_id=config.get('blank_id', 0),
        sos_id=config.get('sos_id', 2),
        eos_id=config.get('eos_id', 3),
    )
