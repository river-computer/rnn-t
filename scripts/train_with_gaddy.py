#!/usr/bin/env python3
"""Train RNN-T using Gaddy's pretrained encoder.

This skips CTC pretraining entirely by using pretrained weights from:
https://zenodo.org/records/7183877

Usage:
    python scripts/train_with_gaddy.py --gaddy-checkpoint checkpoints/gaddy_ctc.pt

    # With frozen encoder (faster convergence, less flexibility)
    python scripts/train_with_gaddy.py --gaddy-checkpoint checkpoints/gaddy_ctc.pt --freeze-encoder
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_transducer
from src.training.rnnt_train import train_rnnt


def main():
    parser = argparse.ArgumentParser(description='Train RNN-T with Gaddy pretrained encoder')
    parser.add_argument('--gaddy-checkpoint', type=str, required=True,
                        help='Path to Gaddy pretrained checkpoint (gaddy_ctc.pt)')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze pretrained encoder weights')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Update config for frozen encoder if requested
    if args.freeze_encoder:
        config['model']['encoder']['freeze_pretrained'] = True
        print("Encoder weights will be FROZEN during training")
    else:
        print("Encoder weights will be FINE-TUNED during training")

    # Check checkpoint exists
    if not Path(args.gaddy_checkpoint).exists():
        print(f"Error: Gaddy checkpoint not found: {args.gaddy_checkpoint}")
        print("Download it with:")
        print("  wget https://zenodo.org/records/7183877/files/model.pt -O checkpoints/gaddy_ctc.pt")
        return 1

    print("\n" + "=" * 60)
    print("RNN-T Training with Gaddy Pretrained Encoder")
    print("=" * 60)
    print(f"  Checkpoint: {args.gaddy_checkpoint}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Config: {args.config}")
    print("=" * 60 + "\n")

    # Verify checkpoint loads
    print("Verifying pretrained checkpoint...")
    try:
        from src.model import GaddyEncoder
        encoder = GaddyEncoder.from_pretrained(args.gaddy_checkpoint)
        print(f"  ✓ Loaded encoder: {encoder.num_layers} transformer layers, {encoder.d_model}-dim")
        del encoder
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗ Failed to load checkpoint: {e}")
        return 1

    # Initialize wandb if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(project='emg-rnnt', name='rnnt-gaddy', config=config)
        except ImportError:
            print("Warning: wandb not available")

    # Train!
    print("\nStarting training...")
    checkpoint_path = train_rnnt(
        config,
        gaddy_checkpoint=args.gaddy_checkpoint,
        epochs=args.epochs,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best checkpoint: {checkpoint_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
