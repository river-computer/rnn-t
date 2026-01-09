"""Loss functions for RNN-T training.

Provides:
1. CTCLoss - wrapper for PyTorch CTC loss (Stage 1 pretraining)
2. RNNTLoss - wrapper for k2 pruned RNN-T loss (Stage 2+ training)
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

# Try to import k2 for pruned RNN-T loss
try:
    import k2
    K2_AVAILABLE = True
except ImportError:
    K2_AVAILABLE = False

# Fallback to torchaudio RNN-T loss
try:
    from torchaudio.transforms import RNNTLoss as TorchaudioRNNTLoss
    TORCHAUDIO_RNNT_AVAILABLE = True
except ImportError:
    TORCHAUDIO_RNNT_AVAILABLE = False


class CTCLoss(nn.Module):
    """CTC loss wrapper for encoder pretraining.

    Uses PyTorch's built-in CTCLoss with log_softmax applied to logits.
    """

    def __init__(self, blank_id: int = 0, reduction: str = 'mean'):
        """Initialize CTC loss.

        Args:
            blank_id: Index of blank token
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.blank_id = blank_id
        self.ctc = nn.CTCLoss(blank=blank_id, reduction=reduction, zero_infinity=True)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Compute CTC loss.

        Args:
            logits: (batch, time, vocab_size) - encoder output + CTC head
            targets: (batch, max_target_len) - target sequences
            input_lengths: (batch,) - input sequence lengths
            target_lengths: (batch,) - target sequence lengths

        Returns:
            loss: Scalar CTC loss
        """
        # CTC expects (time, batch, vocab)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

        # Flatten targets for CTC
        # CTC expects 1D target tensor with all sequences concatenated
        targets_flat = targets[targets != -1]  # Remove padding if any

        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        # Just return CTC loss - let's see if vanilla CTC can learn
        return ctc_loss


class RNNTLoss(nn.Module):
    """RNN-T loss wrapper supporting both k2 pruned loss and torchaudio.

    Prefers k2 pruned loss for 10x memory reduction.
    Falls back to torchaudio if k2 is not available.
    """

    def __init__(
        self,
        blank_id: int = 0,
        reduction: str = 'mean',
        use_pruned: bool = True,
        prune_range: int = 5,
    ):
        """Initialize RNN-T loss.

        Args:
            blank_id: Index of blank token
            reduction: 'mean' or 'sum'
            use_pruned: Whether to use k2 pruned loss (recommended)
            prune_range: Pruning range for k2 loss
        """
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction
        self.use_pruned = use_pruned and K2_AVAILABLE
        self.prune_range = prune_range

        if not self.use_pruned and not TORCHAUDIO_RNNT_AVAILABLE:
            raise ImportError(
                "Neither k2 nor torchaudio RNN-T loss is available. "
                "Install k2 (recommended) or torchaudio."
            )

        if not self.use_pruned:
            self.torchaudio_loss = TorchaudioRNNTLoss(
                blank=blank_id,
                reduction=reduction,
            )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Compute RNN-T loss.

        Args:
            logits: (batch, T, U+1, vocab_size) - joiner output
            targets: (batch, max_target_len) - target sequences (no sos/eos)
            input_lengths: (batch,) - encoder output lengths
            target_lengths: (batch,) - target lengths

        Returns:
            loss: Scalar RNN-T loss
        """
        if self.use_pruned:
            return self._k2_pruned_loss(logits, targets, input_lengths, target_lengths)
        else:
            return self._torchaudio_loss(logits, targets, input_lengths, target_lengths)

    def _k2_pruned_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Compute k2 pruned RNN-T loss.

        k2's pruned loss only computes paths near the alignment boundary,
        reducing memory by ~10x.
        """
        batch_size = logits.shape[0]
        device = logits.device

        # k2 expects log probabilities
        log_probs = logits.log_softmax(dim=-1)

        # Create supervision segments for k2
        # Each segment: (sequence_index, start_frame, duration)
        supervision_segments = torch.stack([
            torch.arange(batch_size, device=device),
            torch.zeros(batch_size, device=device, dtype=torch.int32),
            input_lengths.int(),
        ], dim=1)

        # Compute boundary for pruning
        # The boundary defines the region around the expected alignment path
        # where we compute the loss
        boundary = k2.get_rnnt_prune_ranges(
            lm=log_probs[:, :, :, self.blank_id + 1:].sum(dim=-1),  # Non-blank probs
            am=log_probs[:, :, :, self.blank_id],  # Blank probs
            symbols=targets,
            ranges=self.prune_range,
        )

        # Compute pruned loss
        loss = k2.rnnt_loss_pruned(
            logits=log_probs,
            symbols=targets,
            ranges=boundary,
            termination_symbol=self.blank_id,
            boundary=torch.stack([
                torch.zeros(batch_size, device=device, dtype=torch.int64),
                torch.zeros(batch_size, device=device, dtype=torch.int64),
                input_lengths.long(),
                target_lengths.long(),
            ], dim=1),
            reduction=self.reduction,
        )

        return loss

    def _torchaudio_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Compute torchaudio RNN-T loss (standard, no pruning)."""
        # torchaudio expects log_softmax applied
        log_probs = logits.log_softmax(dim=-1)

        # Debug: Check for length constraint violations
        B, T, U, V = log_probs.shape
        max_input_len = input_lengths.max().item()
        max_target_len = target_lengths.max().item()

        if max_input_len > T:
            raise ValueError(
                f"input_lengths max ({max_input_len}) > logits T dim ({T}). "
                f"Logits shape: {log_probs.shape}, input_lengths: {input_lengths.tolist()}"
            )

        if max_target_len > U - 1:
            raise ValueError(
                f"target_lengths max ({max_target_len}) > logits U-1 dim ({U-1}). "
                f"Logits shape: {log_probs.shape}, target_lengths: {target_lengths.tolist()}"
            )

        # Check for RNN-T constraint: encoder_length >= target_length for each sample
        violations = input_lengths < target_lengths
        if violations.any():
            bad_idx = violations.nonzero().squeeze(-1).tolist()
            raise ValueError(
                f"RNN-T constraint violated: encoder_length < target_length. "
                f"Bad samples: {bad_idx}, "
                f"input_lengths: {input_lengths[violations].tolist()}, "
                f"target_lengths: {target_lengths[violations].tolist()}"
            )

        return self.torchaudio_loss(
            log_probs,
            targets.int(),
            input_lengths.int(),
            target_lengths.int(),
        )


class SimplifiedRNNTLoss(nn.Module):
    """Simplified RNN-T loss using torchaudio.

    Use this if k2 installation fails. Less memory efficient but works.
    """

    def __init__(self, blank_id: int = 0, reduction: str = 'mean'):
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction

        if not TORCHAUDIO_RNNT_AVAILABLE:
            raise ImportError("torchaudio is required for SimplifiedRNNTLoss")

        self.loss_fn = TorchaudioRNNTLoss(blank=blank_id, reduction=reduction)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Compute RNN-T loss.

        Args:
            logits: (batch, T, U+1, vocab_size) - joiner output
            targets: (batch, max_target_len) - target sequences
            input_lengths: (batch,) - encoder output lengths
            target_lengths: (batch,) - target lengths

        Returns:
            loss: Scalar RNN-T loss
        """
        log_probs = logits.log_softmax(dim=-1)
        return self.loss_fn(
            log_probs,
            targets.int(),
            input_lengths.int(),
            target_lengths.int(),
        )
