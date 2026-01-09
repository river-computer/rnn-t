"""Stage 4: Silent Speech Adaptation using DTW.

Adapts the RNN-T model to silent EMG using Dynamic Time Warping
to transfer labels from vocalized to silent recordings.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import yaml

try:
    from tslearn.metrics import dtw_path
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..data import EMGDataset, EMGPreprocessor, collate_fn
from ..model import Transducer, build_transducer
from ..utils import RNNTLoss, SimplifiedRNNTLoss, MetricTracker, S3Checkpoint, download_checkpoint


class SilentVocalizedPairDataset(Dataset):
    """Dataset for silent/vocalized EMG pairs."""

    def __init__(
        self,
        data_dir: str,
        alignments_dir: str,
        preprocessor: EMGPreprocessor,
        split: str = 'train',
    ):
        """Initialize dataset.

        Args:
            data_dir: Path to EMG data directory
            alignments_dir: Path to alignments directory
            preprocessor: EMG preprocessor
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.alignments_dir = Path(alignments_dir)
        self.preprocessor = preprocessor
        self.split = split

        # Find all silent/vocalized pairs
        self.pairs = self._find_pairs()

    def _find_pairs(self) -> List[Dict[str, str]]:
        """Find matching silent/vocalized recording pairs."""
        pairs = []

        silent_dir = self.data_dir / 'silent_parallel_data'
        voiced_dir = self.data_dir / 'voiced_parallel_data'

        if not silent_dir.exists() or not voiced_dir.exists():
            return pairs

        # Map session IDs
        session_map = {
            '5-4_silent': '5-4',
            '5-5_silent': '5-5',
            '5-6_silent': '5-6',
            '5-7_silent': '5-7',
            '5-8_silent': '5-8',
            '5-9_silent': '5-9',
            '5-10_silent': '5-10',
            '5-11_silent': '5-11',
        }

        for silent_session, voiced_session in session_map.items():
            silent_session_dir = silent_dir / silent_session
            voiced_session_dir = voiced_dir / voiced_session

            if not silent_session_dir.exists() or not voiced_session_dir.exists():
                continue

            # Find matching samples by index
            silent_samples = {}
            for f in silent_session_dir.glob('*_emg.npy'):
                idx = int(f.stem.split('_')[0])
                silent_samples[idx] = f

            for f in voiced_session_dir.glob('*_emg.npy'):
                idx = int(f.stem.split('_')[0])
                if idx in silent_samples:
                    pairs.append({
                        'silent_emg': str(silent_samples[idx]),
                        'voiced_emg': str(f),
                        'silent_session': silent_session,
                        'voiced_session': voiced_session,
                        'idx': idx,
                    })

        # Split into train/val (80/20)
        n_train = int(len(pairs) * 0.8)
        if self.split == 'train':
            pairs = pairs[:n_train]
        else:
            pairs = pairs[n_train:]

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        # Load EMG
        silent_emg = np.load(pair['silent_emg'])
        voiced_emg = np.load(pair['voiced_emg'])

        # Preprocess
        silent_emg = self.preprocessor.process(silent_emg, pair['silent_session'])
        voiced_emg = self.preprocessor.process(voiced_emg, pair['voiced_session'])

        # Get session IDs
        session_map = {'5-4': 0, '5-5': 1, '5-6': 2, '5-7': 3, '5-8': 4, '5-9': 5, '5-10': 6, '5-11': 7}
        silent_session_id = session_map.get(pair['voiced_session'].replace('_silent', ''), 0)

        return {
            'silent_emg': torch.from_numpy(silent_emg).float(),
            'voiced_emg': torch.from_numpy(voiced_emg).float(),
            'session_id': torch.tensor(silent_session_id, dtype=torch.long),
            'pair_idx': idx,
        }


def compute_dtw_alignment(
    silent_features: np.ndarray,
    voiced_features: np.ndarray,
) -> List[Tuple[int, int]]:
    """Compute DTW alignment between silent and voiced features.

    Args:
        silent_features: (T_s, D) silent encoder features
        voiced_features: (T_v, D) voiced encoder features

    Returns:
        List of (silent_idx, voiced_idx) alignment pairs
    """
    if not DTW_AVAILABLE:
        raise ImportError("tslearn is required for DTW. Install with: pip install tslearn")

    # DTW path returns aligned indices
    path, _ = dtw_path(silent_features, voiced_features)
    return path


def transfer_labels_via_dtw(
    alignment: List[Tuple[int, int]],
    voiced_labels: List[int],
    voiced_length: int,
    silent_length: int,
) -> List[int]:
    """Transfer labels from voiced to silent via DTW alignment.

    Args:
        alignment: List of (silent_idx, voiced_idx) pairs from DTW
        voiced_labels: Frame-level labels for voiced sequence
        voiced_length: Length of voiced encoder output
        silent_length: Length of silent encoder output

    Returns:
        Transferred labels for silent sequence
    """
    # Create mapping from voiced frames to labels
    # After 4x subsample, need to map encoder frames to original labels
    voiced_frame_labels = voiced_labels[:voiced_length]

    # Transfer labels via alignment
    silent_labels = [0] * silent_length  # Initialize with blanks

    for s_idx, v_idx in alignment:
        if s_idx < silent_length and v_idx < len(voiced_frame_labels):
            silent_labels[s_idx] = voiced_frame_labels[v_idx]

    return silent_labels


class SilentAdaptTrainer:
    """Trainer for silent speech adaptation."""

    def __init__(
        self,
        model: Transducer,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """Initialize trainer.

        Args:
            model: Transducer model (from RNN-T training)
            config: Training configuration
            device: Training device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Freeze encoder initially (or use very low LR)
        train_config = config['training']['silent']
        self.freeze_encoder = train_config.get('freeze_encoder', True)

        if self.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Loss
        try:
            self.criterion = RNNTLoss(
                blank_id=config.get('blank_id', 0),
                use_pruned=train_config.get('use_pruned_loss', True),
            )
        except ImportError:
            self.criterion = SimplifiedRNNTLoss(blank_id=config.get('blank_id', 0))

        # Optimizer (only non-frozen parameters)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            params,
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        # Mixed precision
        self.use_amp = train_config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        s3_config = config.get('s3', {})
        self.checkpoint_manager = S3Checkpoint(
            bucket=s3_config.get('bucket', 'river-emg-speech'),
            prefix=s3_config.get('silent_prefix', 'checkpoints/silent'),
            local_dir=config.get('checkpoint_dir', './checkpoints/silent'),
        )

        # Metrics
        self.metric_tracker = MetricTracker()
        self.best_val_per = float('inf')

    def _compute_pseudo_labels(
        self,
        silent_emg: torch.Tensor,
        voiced_emg: torch.Tensor,
        session_id: torch.Tensor,
        voiced_labels: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute pseudo-labels for silent EMG using DTW.

        Args:
            silent_emg: (1, T_s, 8) silent EMG
            voiced_emg: (1, T_v, 8) voiced EMG
            session_id: (1,) session index
            voiced_labels: Target labels for voiced

        Returns:
            pseudo_targets: Pseudo-labels for silent
            target_lengths: Length of pseudo-labels
        """
        with torch.no_grad():
            # Encode both
            silent_enc, silent_lengths = self.model.encoder(silent_emg, session_id)
            voiced_enc, voiced_lengths = self.model.encoder(voiced_emg, session_id)

            # Get features as numpy
            silent_feat = silent_enc[0, :silent_lengths[0]].cpu().numpy()
            voiced_feat = voiced_enc[0, :voiced_lengths[0]].cpu().numpy()

            # DTW alignment
            alignment = compute_dtw_alignment(silent_feat, voiced_feat)

            # Transfer labels
            pseudo_labels = transfer_labels_via_dtw(
                alignment,
                voiced_labels,
                voiced_lengths[0].item(),
                silent_lengths[0].item(),
            )

            # Collapse to sequence (remove blanks and repeats)
            collapsed = []
            prev = None
            for label in pseudo_labels:
                if label != 0 and label != prev:  # 0 = blank
                    collapsed.append(label)
                prev = label

            return (
                torch.tensor([collapsed], device=self.device),
                torch.tensor([len(collapsed)], device=self.device),
            )

    def train_epoch(
        self,
        dataloader: DataLoader,
        voiced_dataset: EMGDataset,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch with DTW pseudo-labeling.

        Args:
            dataloader: Silent/voiced pair data loader
            voiced_dataset: Voiced dataset for getting labels
            epoch: Current epoch

        Returns:
            Training metrics
        """
        self.model.train()
        if self.freeze_encoder:
            self.model.encoder.eval()

        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            silent_emg = batch['silent_emg'].to(self.device)
            voiced_emg = batch['voiced_emg'].to(self.device)
            session_id = batch['session_id'].to(self.device)

            # Process one sample at a time for DTW
            batch_size = silent_emg.shape[0]

            for i in range(batch_size):
                # Get voiced labels from dataset
                pair_idx = batch['pair_idx'][i].item()
                voiced_labels = voiced_dataset.get_labels(pair_idx)

                # Compute pseudo-labels
                pseudo_targets, target_lengths = self._compute_pseudo_labels(
                    silent_emg[i:i+1],
                    voiced_emg[i:i+1],
                    session_id[i:i+1],
                    voiced_labels,
                )

                if target_lengths[0] == 0:
                    continue

                # Forward
                self.optimizer.zero_grad()

                with autocast(enabled=self.use_amp):
                    logits, encoder_lengths, _ = self.model(
                        silent_emg[i:i+1],
                        session_id[i:i+1],
                        pseudo_targets,
                        torch.tensor([silent_emg.shape[1]], device=self.device),
                        target_lengths,
                    )
                    loss = self.criterion(logits, pseudo_targets, encoder_lengths, target_lengths)

                # Backward
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            'train_loss': total_loss / max(num_batches, 1),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on silent data.

        Note: Since we don't have ground truth for silent, we report
        loss only. PER is computed on vocalized validation set.
        """
        self.model.eval()

        # For silent validation, we decode and report consistency
        all_predictions = []

        for batch in dataloader:
            silent_emg = batch['silent_emg'].to(self.device)
            session_id = batch['session_id'].to(self.device)

            predictions = self.model.decode_greedy(silent_emg, session_id)
            all_predictions.extend(predictions)

        # Report average prediction length as proxy metric
        avg_length = sum(len(p) for p in all_predictions) / len(all_predictions) if all_predictions else 0

        return {
            'avg_pred_length': avg_length,
            'num_predictions': len(all_predictions),
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
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
        self.checkpoint_manager.save(checkpoint, 'best', metadata=metrics)

    def load_from_rnnt(self, rnnt_checkpoint_path: str):
        """Load model from RNN-T checkpoint.

        Args:
            rnnt_checkpoint_path: Path to RNN-T checkpoint
        """
        checkpoint = download_checkpoint(rnnt_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from RNN-T checkpoint: {rnnt_checkpoint_path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        voiced_dataset: EMGDataset,
        epochs: int,
        init_checkpoint: Optional[str] = None,
    ) -> Dict[str, float]:
        """Full training loop.

        Args:
            train_loader: Training pair data loader
            val_loader: Validation pair data loader
            voiced_dataset: Voiced dataset for labels
            epochs: Number of epochs
            init_checkpoint: Path to RNN-T checkpoint

        Returns:
            Final metrics
        """
        if init_checkpoint:
            self.load_from_rnnt(init_checkpoint)

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, voiced_dataset, epoch)
            val_metrics = self.validate(val_loader)

            self.scheduler.step(train_metrics['train_loss'])

            all_metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(epoch, all_metrics)

            print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                  f"avg_pred_length={val_metrics['avg_pred_length']:.1f}")

            if WANDB_AVAILABLE and wandb.run:
                wandb.log(all_metrics, step=epoch)

        return val_metrics


def train_silent(
    config: Dict[str, Any],
    init_checkpoint: Optional[str] = None,
    epochs: int = 25,
) -> str:
    """Train silent adaptation and return checkpoint path.

    Args:
        config: Training configuration
        init_checkpoint: Path to RNN-T checkpoint
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

    # Build preprocessor
    preprocessor = EMGPreprocessor(
        sample_rate=config['data']['emg_sample_rate'],
        target_rate=config['data']['target_sample_rate'],
        notch_freq=config['data']['notch_freq'],
        bandpass_low=config['data']['bandpass_low'],
        bandpass_high=config['data']['bandpass_high'],
        frame_length_ms=config['data']['frame_length_ms'],
        frame_shift_ms=config['data']['frame_shift_ms'],
    )

    # Build datasets
    train_dataset = SilentVocalizedPairDataset(
        data_dir=str(data_dir),
        alignments_dir=str(alignments_dir),
        preprocessor=preprocessor,
        split='train',
    )
    val_dataset = SilentVocalizedPairDataset(
        data_dir=str(data_dir),
        alignments_dir=str(alignments_dir),
        preprocessor=preprocessor,
        split='val',
    )

    # Voiced dataset for labels
    voiced_dataset = EMGDataset(
        data_dir=str(data_dir),
        alignments_dir=str(alignments_dir),
        vocab_path=str(vocab_path),
        sessions=config['data']['train_sessions'],
        split='voiced',
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Train
    trainer = SilentAdaptTrainer(model, config, device)
    trainer.train(train_loader, val_loader, voiced_dataset, epochs, init_checkpoint)

    # Return checkpoint path
    s3_config = config.get('s3', {})
    bucket = s3_config.get('bucket', 'river-emg-speech')
    prefix = s3_config.get('silent_prefix', 'checkpoints/silent')
    return f"s3://{bucket}/{prefix}/best.pt"


def main():
    parser = argparse.ArgumentParser(description='Silent speech adaptation')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    parser.add_argument('--init', type=str, default=None,
                        help='Path to RNN-T checkpoint')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project='emg-rnnt', name='silent-adapt', config=config)

    # Train
    checkpoint_path = train_silent(config, init_checkpoint=args.init, epochs=args.epochs)
    print(f"Training complete. Best checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    main()
