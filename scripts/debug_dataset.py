#!/usr/bin/env python3
"""Debug script to check dataset and alignments."""

import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import EMGDataset, collate_fn
from torch.utils.data import DataLoader


def main():
    with open('config/default.yaml') as f:
        config = yaml.safe_load(f)

    data_dir = Path(config['data']['local_data_dir'])
    alignments_dir = data_dir / 'text_alignments'
    vocab_path = Path('data/vocab.json')

    print("=== Checking Validation Dataset ===")
    ds = EMGDataset(
        data_dir=str(data_dir),
        alignments_dir=str(alignments_dir),
        vocab_path=str(vocab_path),
        sessions=config['data']['val_sessions'],
        split='voiced',
        max_samples=10
    )

    print(f"Total samples: {len(ds)}")
    print()

    empty_seqs = 0
    for i in range(min(5, len(ds))):
        sample = ds[i]
        seq = sample["sequence"].tolist()
        seq_len = sample["sequence_length"]
        emg_frames = sample["emg"].shape[0]
        encoder_out_len = emg_frames // 4  # After 4x subsampling

        print(f'Sample {i}:')
        print(f'  EMG frames: {emg_frames}')
        print(f'  Encoder output length (4x subsample): {encoder_out_len}')
        print(f'  Target sequence: {seq}')
        print(f'  Target length: {seq_len}')

        # Check CTC constraint: input_len >= target_len
        if encoder_out_len < seq_len:
            print(f'  WARNING: CTC constraint violated! encoder_out ({encoder_out_len}) < target ({seq_len})')

        # Check if sequence is just EOS
        if seq == [3]:
            print(f'  WARNING: Sequence is just [EOS]!')
            empty_seqs += 1
        print()

    print(f"Empty sequences (just EOS): {empty_seqs}/{min(5, len(ds))}")
    print()

    # Test collate_fn
    print("=== Testing Collate Function ===")
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"Batch keys: {list(batch.keys())}")
    print(f"EMG shape: {batch['emg'].shape}")
    print(f"EMG lengths: {batch['emg_lengths'].tolist()}")
    print(f"Sequences shape: {batch['sequences'].shape}")
    print(f"Sequence lengths: {batch['sequence_lengths'].tolist()}")
    print(f"First sequence: {batch['sequences'][0].tolist()}")
    print(f"Second sequence: {batch['sequences'][1].tolist()}")


if __name__ == '__main__':
    main()
