"""Gaddy encoder architecture for loading pretrained CTC weights.

Matches the architecture from https://github.com/dgaddy/silent_speech
Paper: "Digital Voicing of Silent Speech" (EMNLP 2020)
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock(nn.Module):
    """Residual block matching Gaddy's architecture.

    Structure: conv1 -> bn1 -> relu -> conv2 -> bn2 -> + residual -> relu
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual path for dimension/stride mismatch
        if stride != 1 or in_channels != out_channels:
            self.residual_path = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(out_channels)
        else:
            self.residual_path = None
            self.res_norm = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, channels, time)

        Returns:
            (batch, out_channels, time // stride)
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        if self.dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual_path is not None:
            residual = self.residual_path(residual)
            residual = self.res_norm(residual)

        out = out + residual
        out = F.relu(out)

        return out


class RelativePositionalEmbedding(nn.Module):
    """Learned relative positional embeddings for attention."""

    def __init__(self, num_heads: int, max_distance: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.head_dim = head_dim

        # Embeddings shape: (num_heads, 2*max_distance - 1, head_dim, 1)
        num_positions = 2 * max_distance - 1
        self.embeddings = nn.Parameter(
            torch.randn(num_heads, num_positions, head_dim, 1) * 0.02
        )

    def forward(self, length: int) -> Tensor:
        """Get relative position biases for attention.

        Args:
            length: sequence length

        Returns:
            (num_heads, length, length) position bias matrix
        """
        # Create relative position indices
        positions = torch.arange(length, device=self.embeddings.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Clamp to valid range and shift to positive indices
        relative_positions = relative_positions.clamp(
            -self.max_distance + 1, self.max_distance - 1
        )
        relative_positions = relative_positions + self.max_distance - 1

        # Gather embeddings: (num_heads, 2*max_dist-1, head_dim, 1)
        # We want to compute attention bias from these
        # For each head, for each position pair, we get a scalar bias

        # Get embeddings for all relative positions
        # Shape: (num_heads, length, length, head_dim)
        pos_emb = self.embeddings[:, relative_positions, :, 0]  # (heads, L, L, head_dim)

        # Sum over head_dim to get scalar bias per position pair
        # This is a simplification - actual impl may differ slightly
        return pos_emb.sum(dim=-1)  # (heads, L, L)


class GaddyMultiheadAttention(nn.Module):
    """Multi-head attention matching Gaddy's implementation.

    Uses separate per-head projections and relative positional encoding.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.2,
        relative_positional_distance: int = 100,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Per-head projections (different from standard PyTorch MHA)
        # Shape: (num_heads, embed_dim, head_dim)
        self.w_q = nn.Parameter(torch.randn(num_heads, embed_dim, self.head_dim) * 0.02)
        self.w_k = nn.Parameter(torch.randn(num_heads, embed_dim, self.head_dim) * 0.02)
        self.w_v = nn.Parameter(torch.randn(num_heads, embed_dim, self.head_dim) * 0.02)
        self.w_o = nn.Parameter(torch.randn(num_heads, self.head_dim, embed_dim) * 0.02)

        # Relative positional encoding
        self.relative_positional = RelativePositionalEmbedding(
            num_heads, relative_positional_distance, self.head_dim
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, embed_dim)
            mask: optional attention mask

        Returns:
            (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V per head
        # x: (batch, seq, embed) -> (batch, seq, heads, head_dim)
        # Using einsum for per-head projections
        q = torch.einsum('bse,hed->bhsd', x, self.w_q)  # (batch, heads, seq, head_dim)
        k = torch.einsum('bse,hed->bhsd', x, self.w_k)
        v = torch.einsum('bse,hed->bhsd', x, self.w_v)

        # Attention scores
        scale = self.head_dim ** -0.5
        attn_logits = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale

        # Add relative positional bias
        pos_bias = self.relative_positional(seq_len)  # (heads, seq, seq)
        attn_logits = attn_logits + pos_bias.unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Project back with per-head output projection
        output = torch.einsum('bhsd,hde->bse', attn_output, self.w_o)

        return output


class GaddyTransformerLayer(nn.Module):
    """Transformer encoder layer matching Gaddy's architecture."""

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        dim_feedforward: int = 3072,
        dropout: float = 0.2,
        relative_positional_distance: int = 100,
    ):
        super().__init__()

        self.self_attn = GaddyMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            relative_positional_distance=relative_positional_distance,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask

        Returns:
            (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_out = self.self_attn(x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # FFN with residual
        ffn_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)

        return x


class GaddyEncoder(nn.Module):
    """EMG encoder matching Gaddy's pretrained model architecture.

    Architecture:
    - 3 ResBlocks with stride=2 (8x downsampling)
    - Linear projection (w_raw_in)
    - 6 Transformer layers with relative positional encoding
    - CTC output head (w_out)
    """

    def __init__(
        self,
        emg_channels: int = 8,
        conv_dim: int = 768,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 3072,
        dropout: float = 0.2,
        vocab_size: int = 38,
        relative_positional_distance: int = 100,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.downsample_factor = 8  # 2^3 from 3 stride-2 conv blocks

        # Conv blocks with downsampling
        self.conv_blocks = nn.ModuleList([
            ResBlock(emg_channels, conv_dim, stride=2, dropout=dropout),
            ResBlock(conv_dim, conv_dim, stride=2, dropout=dropout),
            ResBlock(conv_dim, conv_dim, stride=2, dropout=dropout),
        ])

        # Linear projection before transformer
        self.w_raw_in = nn.Linear(conv_dim, d_model)

        # Transformer layers
        self.transformer = nn.ModuleDict({
            'layers': nn.ModuleList([
                GaddyTransformerLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    relative_positional_distance=relative_positional_distance,
                )
                for _ in range(num_layers)
            ])
        })

        # CTC output head
        self.w_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        emg: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            emg: (batch, time, emg_channels) - raw EMG input
            lengths: (batch,) - sequence lengths

        Returns:
            logits: (batch, time // 8, vocab_size) - CTC logits
            output_lengths: (batch,) - output sequence lengths
        """
        batch_size, time, _ = emg.shape

        # Transpose for conv: (batch, channels, time)
        x = emg.transpose(1, 2)

        # Conv blocks with downsampling
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Transpose back: (batch, time, channels)
        x = x.transpose(1, 2)

        # Linear projection
        x = self.w_raw_in(x)

        # Transformer layers
        for layer in self.transformer['layers']:
            x = layer(x)

        # CTC output
        logits = self.w_out(x)

        # Compute output lengths
        if lengths is not None:
            output_lengths = lengths // self.downsample_factor
        else:
            output_lengths = torch.full(
                (batch_size,), x.shape[1], device=emg.device, dtype=torch.long
            )

        return logits, output_lengths

    def get_encoder_output(
        self,
        emg: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Get encoder output before CTC head (for RNN-T).

        Args:
            emg: (batch, time, emg_channels)
            lengths: (batch,)

        Returns:
            encoder_out: (batch, time // 8, d_model)
            output_lengths: (batch,)
        """
        batch_size, time, _ = emg.shape

        # Transpose for conv
        x = emg.transpose(1, 2)

        # Conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Transpose back
        x = x.transpose(1, 2)

        # Linear projection
        x = self.w_raw_in(x)

        # Transformer
        for layer in self.transformer['layers']:
            x = layer(x)

        # Compute output lengths
        if lengths is not None:
            output_lengths = lengths // self.downsample_factor
        else:
            output_lengths = torch.full(
                (batch_size,), x.shape[1], device=emg.device, dtype=torch.long
            )

        return x, output_lengths


class GaddyEncoderForRNNT(nn.Module):
    """Wrapper around GaddyEncoder for RNN-T training.

    Adds a projection layer to match joiner dimensions and freezes
    pretrained weights optionally.
    """

    def __init__(
        self,
        pretrained_path: str,
        output_dim: int = 128,
        freeze_encoder: bool = False,
        device: str = 'cpu',
    ):
        """Initialize wrapper.

        Args:
            pretrained_path: Path to Gaddy checkpoint
            output_dim: Output dimension for joiner (default 128)
            freeze_encoder: Whether to freeze pretrained weights
            device: Device to load to
        """
        super().__init__()

        # Load pretrained encoder
        self.encoder = GaddyEncoder.from_pretrained(pretrained_path, device)
        self.d_model = self.encoder.d_model  # 768
        self.output_dim = output_dim
        self.downsample_factor = self.encoder.downsample_factor  # 8

        # Projection to joiner dimension
        self.output_proj = nn.Linear(self.d_model, output_dim)

        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        emg: Tensor,
        session_id: Tensor,  # Ignored - Gaddy doesn't use session embedding
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass compatible with EMGEncoder interface.

        Args:
            emg: (batch, time, emg_channels) - raw EMG
            session_id: (batch,) - ignored (Gaddy doesn't use session)
            lengths: (batch,) - sequence lengths

        Returns:
            output: (batch, time // 8, output_dim)
            output_lengths: (batch,)
        """
        # Get encoder output (768-dim)
        encoder_out, output_lengths = self.encoder.get_encoder_output(emg, lengths)

        # Project to joiner dimension
        output = self.output_proj(encoder_out)

        return output, output_lengths

    def get_output_length(self, input_length: int) -> int:
        """Calculate output length (8x downsample)."""
        return input_length // self.downsample_factor

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = 'cpu') -> 'GaddyEncoder':
        """Load pretrained weights from Gaddy checkpoint.

        Args:
            checkpoint_path: path to model.pt
            device: device to load to

        Returns:
            GaddyEncoder with loaded weights
        """
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Infer vocab size from w_out
        vocab_size = state_dict['w_out.weight'].shape[0]

        # Create model
        model = cls(vocab_size=vocab_size)

        # Load weights
        model.load_state_dict(state_dict, strict=True)

        return model
