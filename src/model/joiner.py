"""Joint network (joiner) for RNN-T.

The joiner combines encoder and predictor outputs to produce logits
over the vocabulary at each (time, label) position.
"""

import torch
import torch.nn as nn
from torch import Tensor


class Joiner(nn.Module):
    """Joint network that combines encoder and predictor outputs.

    Architecture:
    1. Add encoder and predictor outputs (broadcast)
    2. Tanh activation
    3. Linear projection to vocabulary size
    """

    def __init__(
        self,
        input_dim: int = 128,
        vocab_size: int = 43,
    ):
        """Initialize joiner.

        Args:
            input_dim: Dimension of encoder/predictor outputs (must match)
            vocab_size: Size of output vocabulary (including blank)
        """
        super().__init__()

        self.input_dim = input_dim
        self.vocab_size = vocab_size

        # Output projection
        self.output_proj = nn.Linear(input_dim, vocab_size)

    def forward(
        self,
        encoder_out: Tensor,
        predictor_out: Tensor,
    ) -> Tensor:
        """Forward pass for training.

        Creates the full (B, T, U, V) tensor for RNN-T loss computation.

        Args:
            encoder_out: (batch, T, input_dim) - encoder outputs
            predictor_out: (batch, U, input_dim) - predictor outputs

        Returns:
            logits: (batch, T, U, vocab_size) - joint logits
        """
        # Expand dimensions for broadcasting
        # encoder: (batch, T, 1, input_dim)
        # predictor: (batch, 1, U, input_dim)
        enc_expanded = encoder_out.unsqueeze(2)
        pred_expanded = predictor_out.unsqueeze(1)

        # Broadcast add: (batch, T, U, input_dim)
        joint = enc_expanded + pred_expanded

        # Activation
        joint = torch.tanh(joint)

        # Project to vocabulary: (batch, T, U, vocab_size)
        logits = self.output_proj(joint)

        return logits

    def forward_single(
        self,
        encoder_frame: Tensor,
        predictor_out: Tensor,
    ) -> Tensor:
        """Forward pass for single encoder frame (streaming inference).

        Args:
            encoder_frame: (batch, 1, input_dim) or (batch, input_dim)
            predictor_out: (batch, 1, input_dim) or (batch, input_dim)

        Returns:
            logits: (batch, vocab_size)
        """
        # Ensure proper shape
        if encoder_frame.dim() == 2:
            encoder_frame = encoder_frame.unsqueeze(1)
        if predictor_out.dim() == 2:
            predictor_out = predictor_out.unsqueeze(1)

        # Add: (batch, 1, input_dim)
        joint = encoder_frame + predictor_out

        # Activation
        joint = torch.tanh(joint)

        # Project: (batch, 1, vocab_size)
        logits = self.output_proj(joint)

        # Squeeze time dimension: (batch, vocab_size)
        return logits.squeeze(1)
