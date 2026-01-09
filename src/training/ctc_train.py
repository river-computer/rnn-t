"""Stage 1: CTC Pretraining for encoder initialization.

Trains the encoder with CTC loss before RNN-T training.
This provides better initialization and faster convergence.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..data import EMGDataset, collate_fn
from ..model import EMGEncoder
from ..utils import CTCLoss, MetricTracker, S3Checkpoint


class CTCHead(nn.Module):
    """CTC output head for encoder pretraining."""

    def __init__(self, input_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, vocab_size)

    def forward(self, x):
        return self.proj(x)


class CTCTrainer:
    """Trainer for CTC pretraining stage."""

    def __init__(
        self,
        encoder: EMGEncoder,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """Initialize trainer.

        Args:
            encoder: EMG encoder module
            config: Training configuration
            device: Training device
        """
        self.encoder = encoder.to(device)
        self.config = config
        self.device = device

        # CTC output head
        vocab_size = config['data']['vocab_size']
        encoder_output_dim = config['model']['encoder']['output_dim']
        self.ctc_head = CTCHead(encoder_output_dim, vocab_size).to(device)

        # Loss
        self.criterion = CTCLoss(blank_id=config.get('blank_id', 0))

        # Optimizer
        train_config = config['training']['ctc']
        self.optimizer = AdamW(
            list(self.encoder.parameters()) + list(self.ctc_head.parameters()),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=train_config['lr_factor'],
            patience=train_config['lr_patience'],
        )

        # Mixed precision
        self.use_amp = train_config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.max_grad_norm = train_config.get('max_grad_norm', 5.0)

        # Checkpointing
        s3_config = config.get('s3', {})
        self.checkpoint_manager = S3Checkpoint(
            bucket=s3_config.get('bucket', 'river-emg-speech'),
            prefix=s3_config.get('ctc_prefix', 'checkpoints/ctc'),
            local_dir=config.get('checkpoint_dir', './checkpoints/ctc'),
        )

        # Metrics
        self.metric_tracker = MetricTracker()

        # Best validation PER for checkpointing
        self.best_val_per = float('inf')

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.encoder.train()
        self.ctc_head.train()

        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            # Move to device
            emg = batch['emg'].to(self.device)
            session_id = batch['session_id'].to(self.device)
            targets = batch['frame_labels'].to(self.device)  # CTC uses frame labels
            emg_lengths = batch['emg_lengths'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            with autocast(enabled=self.use_amp):
                encoder_out, encoder_lengths = self.encoder(emg, session_id, emg_lengths)
                logits = self.ctc_head(encoder_out)
                loss = self.criterion(logits, targets, encoder_lengths, target_lengths)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.ctc_head.parameters()),
                    self.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.ctc_head.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {
            'train_loss': total_loss / num_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on held-out data.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.encoder.eval()
        self.ctc_head.eval()

        total_loss = 0
        num_batches = 0
        self.metric_tracker.reset()

        for batch in dataloader:
            emg = batch['emg'].to(self.device)
            session_id = batch['session_id'].to(self.device)
            targets = batch['frame_labels'].to(self.device)
            emg_lengths = batch['emg_lengths'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)

            # Forward
            encoder_out, encoder_lengths = self.encoder(emg, session_id, emg_lengths)
            logits = self.ctc_head(encoder_out)
            loss = self.criterion(logits, targets, encoder_lengths, target_lengths)

            total_loss += loss.item()
            num_batches += 1

            # Decode for PER
            log_probs = logits.log_softmax(dim=-1)
            predictions = self._ctc_greedy_decode(log_probs, encoder_lengths)
            references = [t[:l].tolist() for t, l in zip(targets, target_lengths)]

            self.metric_tracker.update(predictions, references)

        metrics = self.metric_tracker.compute()
        metrics['val_loss'] = total_loss / num_batches

        return metrics

    def _ctc_greedy_decode(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> list:
        """Greedy CTC decoding.

        Args:
            log_probs: (batch, time, vocab) log probabilities
            lengths: (batch,) sequence lengths

        Returns:
            List of decoded sequences (collapsed, no blanks)
        """
        predictions = []
        batch_size = log_probs.shape[0]

        for b in range(batch_size):
            # Get best path
            best_path = log_probs[b, :lengths[b]].argmax(dim=-1).tolist()

            # Collapse repeated tokens and remove blanks
            collapsed = []
            prev = None
            for token in best_path:
                if token != 0 and token != prev:  # 0 = blank
                    collapsed.append(token)
                prev = token

            predictions.append(collapsed)

        return predictions

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint to S3.

        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'ctc_head_state_dict': self.ctc_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }

        # Save epoch checkpoint
        self.checkpoint_manager.save(checkpoint, f'epoch_{epoch}', metadata=metrics)

        # Save best checkpoint
        if is_best:
            self.checkpoint_manager.save(checkpoint, 'best', metadata=metrics)
            print(f"New best checkpoint (PER: {metrics['per']:.4f})")

    def load_checkpoint(self, name: str = 'best') -> int:
        """Load checkpoint from S3.

        Args:
            name: Checkpoint name to load

        Returns:
            Epoch number of loaded checkpoint
        """
        if not self.checkpoint_manager.exists(name):
            return 0

        checkpoint = self.checkpoint_manager.load(name, map_location=self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.ctc_head.load_state_dict(checkpoint['ctc_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch']

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        resume: bool = True,
    ) -> Dict[str, float]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            resume: Whether to resume from checkpoint

        Returns:
            Final metrics
        """
        start_epoch = 0
        if resume:
            start_epoch = self.load_checkpoint('latest')
            if start_epoch > 0:
                print(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_metrics['val_loss'])

            # Check if best
            is_best = val_metrics['per'] < self.best_val_per
            if is_best:
                self.best_val_per = val_metrics['per']

            # Save checkpoint
            all_metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(epoch, all_metrics, is_best)

            # Log
            print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"PER={val_metrics['per']:.4f}")

            if WANDB_AVAILABLE and wandb.run:
                wandb.log(all_metrics, step=epoch)

        return val_metrics


def train_ctc(config: Dict[str, Any], epochs: int = 50, target_per: float = 0.35) -> str:
    """Train CTC model and return checkpoint path.

    Args:
        config: Training configuration
        epochs: Number of epochs
        target_per: Target PER to stop early

    Returns:
        S3 path to best checkpoint
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build encoder
    encoder_config = config['model']['encoder']
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

    # Build datasets
    train_dataset = EMGDataset(
        data_dir=config['data']['local_data_dir'],
        split='train',
        data_type='voiced',
    )
    val_dataset = EMGDataset(
        data_dir=config['data']['local_data_dir'],
        split='val',
        data_type='voiced',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['ctc']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['ctc']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Train
    trainer = CTCTrainer(encoder, config, device)
    trainer.train(train_loader, val_loader, epochs)

    # Return best checkpoint path
    s3_config = config.get('s3', {})
    bucket = s3_config.get('bucket', 'river-emg-speech')
    prefix = s3_config.get('ctc_prefix', 'checkpoints/ctc')
    return f"s3://{bucket}/{prefix}/best.pt"


def main():
    parser = argparse.ArgumentParser(description='CTC pretraining for EMG encoder')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project='emg-rnnt', name='ctc-pretrain', config=config)

    # Train
    checkpoint_path = train_ctc(config, epochs=args.epochs)
    print(f"Training complete. Best checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    main()
