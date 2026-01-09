#!/usr/bin/env python3
"""Unified training script that runs all stages sequentially.

Stages:
1. CTC Pretraining - Initialize encoder (50 epochs)
2. RNN-T Training - Full transducer (100 epochs)
3. Silent Adaptation - DTW-based transfer (25 epochs)

Features:
- Automatic checkpoint management to S3
- Resume from any stage if interrupted
- Early stopping if validation PER plateaus
- Progress logging to wandb (optional)

Usage:
    python scripts/train_all.py --config config/default.yaml

    # Resume from specific stage
    python scripts/train_all.py --config config/default.yaml --start-stage 2

    # Dry run (test without training)
    python scripts/train_all.py --config config/default.yaml --dry-run
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def check_prerequisites(config: dict) -> bool:
    """Check that all prerequisites are met before training."""
    print("\n=== Checking Prerequisites ===")
    errors = []

    # Check data directory
    data_dir = Path(config['data']['local_data_dir'])
    if not data_dir.exists():
        errors.append(f"Data directory not found: {data_dir}")

    # Check for EMG files
    voiced_dir = data_dir / 'voiced_parallel_data'
    if voiced_dir.exists():
        emg_files = list(voiced_dir.glob('**/*_emg.npy'))
        if len(emg_files) < 100:
            errors.append(f"Insufficient training data: {len(emg_files)} EMG files (need 100+)")
        else:
            print(f"  ✓ Found {len(emg_files)} EMG files")
    else:
        errors.append("voiced_parallel_data directory not found")

    # Check alignments
    alignments_dir = data_dir / 'text_alignments'
    if alignments_dir.exists():
        alignment_files = list(alignments_dir.glob('**/*.TextGrid'))
        print(f"  ✓ Found {len(alignment_files)} alignment files")
    else:
        print("  ⚠ Alignments directory not found (will use text only)")

    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU available: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        errors.append("No GPU available (training will be very slow)")

    # Check S3 connectivity
    try:
        import boto3
        s3 = boto3.client('s3')
        bucket = config.get('s3', {}).get('bucket', 'river-emg-speech')
        s3.head_bucket(Bucket=bucket)
        print(f"  ✓ S3 bucket accessible: {bucket}")
    except Exception as e:
        print(f"  ⚠ S3 not accessible: {e} (will use local checkpoints)")

    if errors:
        print("\n  ✗ Prerequisites not met:")
        for error in errors:
            print(f"    - {error}")
        return False

    print("\n  ✓ All prerequisites met")
    return True


def get_checkpoint_status(config: dict) -> dict:
    """Check which checkpoints exist."""
    from src.utils import checkpoint_exists

    s3_config = config.get('s3', {})
    bucket = s3_config.get('bucket', 'river-emg-speech')

    status = {
        'ctc_best': checkpoint_exists(f"s3://{bucket}/checkpoints/ctc/best.pt"),
        'rnnt_best': checkpoint_exists(f"s3://{bucket}/checkpoints/rnnt/best.pt"),
        'silent_best': checkpoint_exists(f"s3://{bucket}/checkpoints/silent/best.pt"),
    }

    return status


def run_stage_1_ctc(config: dict, dry_run: bool = False) -> str:
    """Stage 1: CTC Pretraining."""
    print("\n" + "=" * 60)
    print("Stage 1: CTC Pretraining")
    print("=" * 60)

    if dry_run:
        print("  [DRY RUN] Would train CTC for 50 epochs")
        return "s3://river-emg-speech/checkpoints/ctc/best.pt"

    from src.training import train_ctc

    epochs = config['training']['ctc']['epochs']
    print(f"  Training for {epochs} epochs...")

    checkpoint_path = train_ctc(config, epochs=epochs)
    print(f"  ✓ Stage 1 complete: {checkpoint_path}")

    return checkpoint_path


def run_stage_2_rnnt(config: dict, ctc_checkpoint: str, dry_run: bool = False) -> str:
    """Stage 2: RNN-T Training."""
    print("\n" + "=" * 60)
    print("Stage 2: RNN-T Training")
    print("=" * 60)

    if dry_run:
        print("  [DRY RUN] Would train RNN-T for 100 epochs")
        return "s3://river-emg-speech/checkpoints/rnnt/best.pt"

    from src.training import train_rnnt

    epochs = config['training']['rnnt']['epochs']
    print(f"  Training for {epochs} epochs...")
    print(f"  Encoder initialized from: {ctc_checkpoint}")

    checkpoint_path = train_rnnt(config, encoder_init=ctc_checkpoint, epochs=epochs)
    print(f"  ✓ Stage 2 complete: {checkpoint_path}")

    return checkpoint_path


def run_stage_3_silent(config: dict, rnnt_checkpoint: str, dry_run: bool = False) -> str:
    """Stage 3: Silent Adaptation."""
    print("\n" + "=" * 60)
    print("Stage 3: Silent Adaptation")
    print("=" * 60)

    if dry_run:
        print("  [DRY RUN] Would train silent adaptation for 25 epochs")
        return "s3://river-emg-speech/checkpoints/silent/best.pt"

    from src.training import train_silent

    epochs = config['training']['silent']['epochs']
    print(f"  Training for {epochs} epochs...")
    print(f"  Model initialized from: {rnnt_checkpoint}")

    checkpoint_path = train_silent(config, init_checkpoint=rnnt_checkpoint, epochs=epochs)
    print(f"  ✓ Stage 3 complete: {checkpoint_path}")

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Run all training stages')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    parser.add_argument('--start-stage', type=int, default=1, choices=[1, 2, 3],
                        help='Stage to start from (1=CTC, 2=RNN-T, 3=Silent)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without training')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--skip-prereq', action='store_true',
                        help='Skip prerequisite checks')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("EMG RNN-T Training Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check prerequisites
    if not args.skip_prereq and not args.dry_run:
        if not check_prerequisites(config):
            print("\n✗ Aborting due to failed prerequisites")
            return 1

    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project='emg-rnnt',
            name=f'full-training-{datetime.now().strftime("%Y%m%d-%H%M")}',
            config=config,
        )
        print("\n  ✓ Wandb initialized")

    # Check existing checkpoints
    ckpt_status = get_checkpoint_status(config)
    print("\n=== Checkpoint Status ===")
    print(f"  CTC best:    {'✓ exists' if ckpt_status['ctc_best'] else '✗ not found'}")
    print(f"  RNN-T best:  {'✓ exists' if ckpt_status['rnnt_best'] else '✗ not found'}")
    print(f"  Silent best: {'✓ exists' if ckpt_status['silent_best'] else '✗ not found'}")

    # Determine starting checkpoints based on stage
    s3_config = config.get('s3', {})
    bucket = s3_config.get('bucket', 'river-emg-speech')

    ctc_checkpoint = f"s3://{bucket}/checkpoints/ctc/best.pt"
    rnnt_checkpoint = f"s3://{bucket}/checkpoints/rnnt/best.pt"

    # Run stages
    if args.start_stage <= 1:
        ctc_checkpoint = run_stage_1_ctc(config, dry_run=args.dry_run)

    if args.start_stage <= 2:
        if args.start_stage == 2 and not ckpt_status['ctc_best']:
            print("\n✗ Cannot start at stage 2: CTC checkpoint not found")
            return 1
        rnnt_checkpoint = run_stage_2_rnnt(config, ctc_checkpoint, dry_run=args.dry_run)

    if args.start_stage <= 3:
        if args.start_stage == 3 and not ckpt_status['rnnt_best']:
            print("\n✗ Cannot start at stage 3: RNN-T checkpoint not found")
            return 1
        final_checkpoint = run_stage_3_silent(config, rnnt_checkpoint, dry_run=args.dry_run)

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal model: {final_checkpoint}")
    print(f"\nTo export for inference:")
    print(f"  python -m src.inference.export {final_checkpoint} --output model.pt")
    print(f"\nTo run inference:")
    print(f"  from src.inference import StreamingDecoder")
    print(f"  decoder = StreamingDecoder.from_checkpoint('model.pt')")

    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()

    return 0


if __name__ == '__main__':
    sys.exit(main())
