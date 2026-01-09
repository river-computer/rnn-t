"""Stage 2-3: RNN-T Training.

Trains the full RNN-Transducer model with k2 pruned loss.
Initializes encoder from CTC pretrained checkpoint.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..data import EMGDataset, collate_fn
from ..model import Transducer, build_transducer
from ..utils import RNNTLoss, SimplifiedRNNTLoss, MetricTracker, S3Checkpoint, download_checkpoint


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr

    def _get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * progress))

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']


class RNNTTrainer:
    """Trainer for RNN-T model."""

    def __init__(
        self,
        model: Transducer,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """Initialize trainer.

        Args:
            model: Transducer model
            config: Training configuration
            device: Training device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Loss
        train_config = config['training']['rnnt']
        try:
            self.criterion = RNNTLoss(
                blank_id=config.get('blank_id', 0),
                use_pruned=train_config.get('use_pruned_loss', True),
                prune_range=train_config.get('prune_range', 5),
            )
        except ImportError:
            print("Warning: k2 not available, using torchaudio RNN-T loss")
            self.criterion = SimplifiedRNNTLoss(blank_id=config.get('blank_id', 0))

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
        )

        # Scheduler
        total_steps = train_config['epochs'] * train_config.get('steps_per_epoch', 1000)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=train_config.get('warmup_steps', 2000),
            total_steps=total_steps,
        )

        # Mixed precision
        self.use_amp = train_config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.accum_steps = train_config.get('gradient_accumulation_steps', 4)
        self.max_grad_norm = train_config.get('max_grad_norm', 5.0)

        # Checkpointing
        s3_config = config.get('s3', {})
        self.checkpoint_manager = S3Checkpoint(
            bucket=s3_config.get('bucket', 'river-emg-speech'),
            prefix=s3_config.get('rnnt_prefix', 'checkpoints/rnnt'),
            local_dir=config.get('checkpoint_dir', './checkpoints/rnnt'),
        )

        # Metrics
        self.metric_tracker = MetricTracker()
        self.best_val_per = float('inf')

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0
        num_batches = 0
        accumulated_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            emg = batch['emg'].to(self.device)
            session_id = batch['session_id'].to(self.device)
            targets = batch['targets'].to(self.device)  # RNN-T uses sequence targets
            emg_lengths = batch['emg_lengths'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)

            # Forward pass
            with autocast(enabled=self.use_amp):
                logits, encoder_lengths, _ = self.model(
                    emg, session_id, targets, emg_lengths, target_lengths
                )
                loss = self.criterion(logits, targets, encoder_lengths, target_lengths)
                loss = loss / self.accum_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

                total_loss += accumulated_loss * self.accum_steps
                accumulated_loss = 0
                num_batches += 1

        return {
            'train_loss': total_loss / max(num_batches, 1),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on held-out data.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0
        num_batches = 0
        self.metric_tracker.reset()

        for batch in dataloader:
            emg = batch['emg'].to(self.device)
            session_id = batch['session_id'].to(self.device)
            targets = batch['targets'].to(self.device)
            emg_lengths = batch['emg_lengths'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)

            # Forward (compute loss)
            logits, encoder_lengths, _ = self.model(
                emg, session_id, targets, emg_lengths, target_lengths
            )
            loss = self.criterion(logits, targets, encoder_lengths, target_lengths)

            total_loss += loss.item()
            num_batches += 1

            # Greedy decode for PER
            predictions = self.model.decode_greedy(emg, session_id)
            references = [t[:l].tolist() for t, l in zip(targets, target_lengths)]

            self.metric_tracker.update(predictions, references)

        metrics = self.metric_tracker.compute()
        metrics['val_loss'] = total_loss / num_batches

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint to S3."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }

        self.checkpoint_manager.save(checkpoint, f'epoch_{epoch}', metadata=metrics)

        if is_best:
            self.checkpoint_manager.save(checkpoint, 'best', metadata=metrics)
            print(f"New best checkpoint (PER: {metrics['per']:.4f})")

    def load_checkpoint(self, name: str = 'best') -> int:
        """Load checkpoint from S3."""
        if not self.checkpoint_manager.exists(name):
            return 0

        checkpoint = self.checkpoint_manager.load(name, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch']

    def load_encoder_from_ctc(self, ctc_checkpoint_path: str):
        """Load encoder weights from CTC checkpoint.

        Args:
            ctc_checkpoint_path: S3 or local path to CTC checkpoint
        """
        checkpoint = download_checkpoint(ctc_checkpoint_path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"Loaded encoder from CTC checkpoint: {ctc_checkpoint_path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        encoder_init: Optional[str] = None,
        resume: bool = True,
    ) -> Dict[str, float]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            encoder_init: Path to CTC checkpoint for encoder initialization
            resume: Whether to resume from checkpoint

        Returns:
            Final metrics
        """
        start_epoch = 0

        # Try to resume first
        if resume:
            start_epoch = self.load_checkpoint('latest')
            if start_epoch > 0:
                print(f"Resuming from epoch {start_epoch}")

        # Initialize encoder from CTC if not resuming
        if start_epoch == 0 and encoder_init:
            self.load_encoder_from_ctc(encoder_init)

        for epoch in range(start_epoch, epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

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
                  f"PER={val_metrics['per']:.4f}, "
                  f"LR={train_metrics['learning_rate']:.2e}")

            if WANDB_AVAILABLE and wandb.run:
                wandb.log(all_metrics, step=epoch)

        return val_metrics


def train_rnnt(
    config: Dict[str, Any],
    encoder_init: Optional[str] = None,
    epochs: int = 100,
) -> str:
    """Train RNN-T model and return checkpoint path.

    Args:
        config: Training configuration
        encoder_init: Path to CTC checkpoint for encoder initialization
        epochs: Number of epochs

    Returns:
        S3 path to best checkpoint
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = build_transducer(config)

    # Build paths
    data_dir = Path(config['data']['local_data_dir'])
    alignments_dir = data_dir / 'text_alignments'
    vocab_path = Path('data/vocab.json')

    # Build datasets
    train_dataset = EMGDataset(
        data_dir=str(data_dir),
        alignments_dir=str(alignments_dir),
        vocab_path=str(vocab_path),
        sessions=config['data']['train_sessions'],
        split='voiced',
    )
    val_dataset = EMGDataset(
        data_dir=str(data_dir),
        alignments_dir=str(alignments_dir),
        vocab_path=str(vocab_path),
        sessions=config['data']['val_sessions'],
        split='voiced',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['rnnt']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['rnnt']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Train
    trainer = RNNTTrainer(model, config, device)
    trainer.train(train_loader, val_loader, epochs, encoder_init=encoder_init)

    # Return best checkpoint path
    s3_config = config.get('s3', {})
    bucket = s3_config.get('bucket', 'river-emg-speech')
    prefix = s3_config.get('rnnt_prefix', 'checkpoints/rnnt')
    return f"s3://{bucket}/{prefix}/best.pt"


def main():
    parser = argparse.ArgumentParser(description='RNN-T training for EMG')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    parser.add_argument('--encoder-init', type=str, default=None,
                        help='Path to CTC checkpoint for encoder initialization')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project='emg-rnnt', name='rnnt-train', config=config)

    # Train
    checkpoint_path = train_rnnt(config, encoder_init=args.encoder_init, epochs=args.epochs)
    print(f"Training complete. Best checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    main()
