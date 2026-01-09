"""Phoneme predictor (prediction network) for RNN-T.

The predictor is an autoregressive model that predicts the next phoneme
based on the previously emitted phonemes. During training, it receives
ground truth previous tokens (teacher forcing). During inference, it
receives the last emitted token.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Predictor(nn.Module):
    """LSTM-based prediction network for RNN-T.

    Architecture:
    1. Embedding layer for input tokens
    2. Single LSTM layer for sequence modeling
    3. Linear projection to joint dimension
    """

    def __init__(
        self,
        vocab_size: int = 43,
        embed_dim: int = 128,
        hidden_dim: int = 320,
        output_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """Initialize predictor.

        Args:
            vocab_size: Number of phoneme tokens (including special tokens)
            embed_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            output_dim: Output projection dimension (must match encoder output)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass for training.

        Args:
            tokens: (batch, seq_len) - input token sequence
                   For RNN-T training: this is [<sos>] + targets[:-1]
            hidden: Optional (h, c) LSTM hidden state tuple
                   Each of shape (num_layers, batch, hidden_dim)

        Returns:
            output: (batch, seq_len, output_dim)
            hidden: (h, c) final hidden state tuple
        """
        batch_size, seq_len = tokens.shape

        # Embed tokens
        x = self.embedding(tokens)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, tokens.device)

        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)  # (batch, seq_len, hidden_dim)

        # Output projection
        output = self.output_proj(lstm_out)  # (batch, seq_len, output_dim)

        return output, hidden

    def forward_step(
        self,
        token: Tensor,
        hidden: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Single step forward for streaming inference.

        Args:
            token: (batch,) or (batch, 1) - single token
            hidden: (h, c) LSTM hidden state tuple

        Returns:
            output: (batch, 1, output_dim)
            hidden: Updated (h, c) hidden state
        """
        # Ensure token has sequence dimension
        if token.dim() == 1:
            token = token.unsqueeze(1)  # (batch, 1)

        # Embed
        x = self.embedding(token)  # (batch, 1, embed_dim)

        # LSTM step
        lstm_out, hidden = self.lstm(x, hidden)  # (batch, 1, hidden_dim)

        # Project
        output = self.output_proj(lstm_out)  # (batch, 1, output_dim)

        return output, hidden

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Initialize LSTM hidden state.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            (h_0, c_0) tuple of zero-initialized hidden states
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
