"""EMG Encoder with causal attention for streaming inference.

Architecture:
- Session embedding for electrode placement variation
- Conv blocks with residual connections
- 4x subsampling via strided convolutions
- Causal Transformer encoder (6 layers)
- Output projection to joint dimension
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBlock(nn.Module):
    """Convolutional block with residual connection and GELU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,  # Same padding
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, channels, time)

        Returns:
            (batch, out_channels, time)
        """
        residual = self.residual(x)
        x = self.conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class CausalConv1d(nn.Module):
    """Causal 1D convolution that only looks at past frames."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        # Left padding only for causal
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with causal padding.

        Args:
            x: (batch, channels, time)

        Returns:
            (batch, out_channels, time // stride)
        """
        # Pad only on the left side
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class SubsampleBlock(nn.Module):
    """Subsampling block that reduces time dimension by 4x."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        # Two stride-2 convolutions for 4x reduction
        self.conv1 = CausalConv1d(in_dim, out_dim, kernel_size=3, stride=2)
        self.norm1 = nn.BatchNorm1d(out_dim)
        self.conv2 = CausalConv1d(out_dim, out_dim, kernel_size=3, stride=2)
        self.norm2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, channels, time)

        Returns:
            (batch, out_dim, time // 4)
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        return x


class CausalMultiheadAttention(nn.Module):
    """Multi-head attention with causal masking for streaming."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass with optional KV caching for streaming.

        Args:
            x: (batch, seq_len, embed_dim)
            kv_cache: Optional tuple of (K, V) from previous chunks
            use_cache: Whether to return KV cache for next chunk

        Returns:
            output: (batch, seq_len, embed_dim)
            new_cache: Optional (K, V) for next chunk if use_cache=True
        """
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # If we have cached KV, prepend them
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        # Reshape for multi-head attention
        # (batch, seq, embed) -> (batch, heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        kv_len = k.shape[2]

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Create causal mask
        # Query positions can only attend to key positions <= their absolute position
        # For streaming: query position i (in current chunk) can attend to
        # key positions 0...(cache_len + i)
        if kv_cache is not None:
            cache_len = kv_cache[0].shape[1]
        else:
            cache_len = 0

        # Create causal mask for current queries against all keys
        causal_mask = torch.ones(seq_len, kv_len, dtype=torch.bool, device=x.device)
        for i in range(seq_len):
            # Query i can attend to keys 0 through (cache_len + i)
            causal_mask[i, :cache_len + i + 1] = False

        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        # Prepare cache for next chunk
        new_cache = None
        if use_cache:
            # Cache the full K, V including current chunk
            new_cache = (
                k.transpose(1, 2).contiguous().view(batch_size, kv_len, self.embed_dim),
                v.transpose(1, 2).contiguous().view(batch_size, kv_len, self.embed_dim),
            )

        return output, new_cache


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with causal attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = CausalMultiheadAttention(d_model, nhead, dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            kv_cache: Optional KV cache from previous chunk
            use_cache: Whether to return cache

        Returns:
            output: (batch, seq_len, d_model)
            new_cache: Optional KV cache for next chunk
        """
        # Self-attention with residual
        attn_out, new_cache = self.self_attn(x, kv_cache, use_cache)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # FFN with residual
        ffn_out = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)

        return x, new_cache


class EMGEncoder(nn.Module):
    """EMG encoder for RNN-T with causal attention.

    Processes EMG signals through:
    1. Session embedding (handles electrode placement variation)
    2. Conv blocks with residual connections
    3. 4x subsampling
    4. Causal Transformer layers
    5. Output projection
    """

    def __init__(
        self,
        emg_channels: int = 8,
        num_sessions: int = 8,
        session_embed_dim: int = 32,
        conv_dim: int = 768,
        num_conv_blocks: int = 3,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.emg_channels = emg_channels
        self.num_layers = num_layers
        self.d_model = d_model
        self.output_dim = output_dim

        # Session embedding
        self.session_embedding = nn.Embedding(num_sessions, session_embed_dim)

        # Input dimension after concatenating EMG + session
        input_dim = emg_channels + session_embed_dim

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_dim = input_dim
        for i in range(num_conv_blocks):
            out_dim = conv_dim
            self.conv_blocks.append(ConvBlock(in_dim, out_dim, dropout=dropout))
            in_dim = out_dim

        # Subsampling (4x reduction)
        self.subsample = SubsampleBlock(conv_dim, d_model, dropout=dropout)

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(
        self,
        emg: Tensor,
        session_id: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for training.

        Args:
            emg: (batch, time, emg_channels) - preprocessed EMG
            session_id: (batch,) - session indices
            lengths: (batch,) - original lengths for masking (optional)

        Returns:
            output: (batch, time//4, output_dim)
            output_lengths: (batch,) - output sequence lengths
        """
        batch_size, time, _ = emg.shape

        # Get session embedding and expand to time dimension
        session_embed = self.session_embedding(session_id)  # (batch, session_dim)
        session_embed = session_embed.unsqueeze(1).expand(-1, time, -1)  # (batch, time, session_dim)

        # Concatenate EMG and session embedding
        x = torch.cat([emg, session_embed], dim=-1)  # (batch, time, emg + session)

        # Transpose for conv: (batch, channels, time)
        x = x.transpose(1, 2)

        # Conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Subsample (4x reduction)
        x = self.subsample(x)

        # Transpose back: (batch, time, channels)
        x = x.transpose(1, 2)

        # Transformer layers
        for layer in self.transformer_layers:
            x, _ = layer(x, kv_cache=None, use_cache=False)

        # Output projection
        output = self.output_proj(x)

        # Compute output lengths if input lengths provided
        if lengths is not None:
            output_lengths = (lengths // 4).long()
        else:
            output_lengths = torch.full((batch_size,), output.shape[1], device=emg.device)

        return output, output_lengths

    def forward_streaming(
        self,
        emg: Tensor,
        session_id: Tensor,
        kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """Forward pass for streaming inference.

        Args:
            emg: (batch, chunk_time, emg_channels) - preprocessed EMG chunk
            session_id: (batch,) - session indices
            kv_cache: List of (K, V) tuples per layer from previous chunks

        Returns:
            output: (batch, chunk_time//4, output_dim)
            new_cache: Updated KV cache for next chunk
        """
        batch_size, time, _ = emg.shape

        # Initialize cache if needed
        if kv_cache is None:
            kv_cache = [None] * self.num_layers

        # Session embedding
        session_embed = self.session_embedding(session_id)
        session_embed = session_embed.unsqueeze(1).expand(-1, time, -1)

        # Concatenate
        x = torch.cat([emg, session_embed], dim=-1)
        x = x.transpose(1, 2)

        # Conv blocks (stateless with causal padding)
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Subsample
        x = self.subsample(x)
        x = x.transpose(1, 2)

        # Transformer with KV caching
        new_cache = []
        for i, layer in enumerate(self.transformer_layers):
            x, layer_cache = layer(x, kv_cache=kv_cache[i], use_cache=True)
            new_cache.append(layer_cache)

        # Output projection
        output = self.output_proj(x)

        return output, new_cache

    def get_output_length(self, input_length: int) -> int:
        """Calculate output length given input length (4x subsample)."""
        # Each subsample layer does stride=2, total 4x reduction
        return input_length // 4
