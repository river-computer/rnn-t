#!/usr/bin/env python3
"""Debug script to check alignment loading."""

import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.alignment import AlignmentLoader, Vocabulary


def main():
    with open('config/default.yaml') as f:
        config = yaml.safe_load(f)

    data_dir = Path(config['data']['local_data_dir'])
    alignments_dir = data_dir / 'text_alignments'
    vocab_path = Path('data/vocab.json')

    print(f"=== Alignment Debug ===")
    print(f"Data dir: {data_dir}")
    print(f"Alignments dir: {alignments_dir}")
    print(f"Alignments dir exists: {alignments_dir.exists()}")
    print()

    # List what's in alignments dir
    if alignments_dir.exists():
        print("Contents of alignments dir:")
        for item in sorted(alignments_dir.iterdir())[:10]:
            print(f"  {item.name}")
            if item.is_dir():
                subitems = list(item.iterdir())[:3]
                for subitem in subitems:
                    print(f"    {subitem.name}")
        print()

    # Check for TextGrid files
    textgrids = list(alignments_dir.glob('**/*.TextGrid'))
    print(f"Total TextGrid files found: {len(textgrids)}")
    if textgrids:
        print("First few TextGrid paths:")
        for tg in textgrids[:5]:
            print(f"  {tg.relative_to(alignments_dir)}")
    print()

    # Try loading an alignment
    vocab = Vocabulary(str(vocab_path))
    loader = AlignmentLoader(str(alignments_dir), vocab)

    # Get a sample utterance from the EMG data
    emg_dir = data_dir / 'voiced_parallel_data'
    print(f"EMG data dir: {emg_dir}")
    print(f"EMG data dir exists: {emg_dir.exists()}")

    if emg_dir.exists():
        sessions = list(emg_dir.iterdir())[:2]
        for session_dir in sessions:
            session_id = session_dir.name
            print(f"\nSession: {session_id}")
            emg_files = list(session_dir.glob('*_emg.npy'))[:3]
            for emg_file in emg_files:
                utterance_id = emg_file.stem.replace('_emg', '')
                print(f"  Utterance: {utterance_id}")

                # Try to find alignment
                alignment_path = loader.find_alignment(utterance_id, session_id)
                print(f"    Alignment path: {alignment_path}")

                if alignment_path:
                    alignment = loader.load_textgrid(alignment_path)
                    if alignment:
                        seq = loader.alignment_to_sequence(alignment)
                        print(f"    Sequence: {seq.tolist()}")
                        print(f"    Intervals: {len(alignment.intervals)}")
                        if alignment.intervals:
                            print(f"    First interval: {alignment.intervals[0]}")


if __name__ == '__main__':
    main()
