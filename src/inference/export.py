"""Model export utilities for MacBook deployment.

Exports trained models to a portable format that can be loaded
without the full training infrastructure.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from ..model import Transducer, build_transducer
from ..data import EMGPreprocessor


def export_model(
    checkpoint_path: str,
    output_path: str,
    include_preprocessing_stats: bool = True,
    vocab_path: Optional[str] = None,
) -> str:
    """Export trained model for inference deployment.

    Creates a standalone checkpoint with:
    - Model state dict
    - Model configuration
    - Preprocessing statistics
    - Vocabulary

    Args:
        checkpoint_path: Path to training checkpoint
        output_path: Path for exported model
        include_preprocessing_stats: Include normalization stats
        vocab_path: Path to vocabulary file

    Returns:
        Path to exported model
    """
    # Load training checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model config
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # Build model to get architecture info
    model = build_transducer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare export dict
    export_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'data_config': config.get('data', {}),
        'special_tokens': {
            'blank_id': config.get('blank_id', 0),
            'sos_id': config.get('sos_id', 2),
            'eos_id': config.get('eos_id', 3),
        },
        'training_metrics': checkpoint.get('metrics', {}),
    }

    # Include vocabulary
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path) as f:
            export_dict['vocab'] = json.load(f)
    else:
        # Default vocabulary
        export_dict['vocab'] = _get_default_vocab()

    # Include preprocessing stats if available
    if include_preprocessing_stats:
        export_dict['preprocessing_stats'] = config.get('preprocessing_stats', {})

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(export_dict, output_path)
    print(f"Exported model to {output_path}")

    # Also save a metadata JSON for easy inspection
    metadata_path = output_path.replace('.pt', '_metadata.json')
    metadata = {
        'model_config': model_config,
        'data_config': export_dict['data_config'],
        'special_tokens': export_dict['special_tokens'],
        'training_metrics': export_dict['training_metrics'],
        'vocab_size': len(export_dict['vocab'].get('phonemes', {})) + len(export_dict['vocab'].get('special_tokens', {})),
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    return output_path


def load_exported_model(
    model_path: str,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Load exported model for inference.

    Args:
        model_path: Path to exported model
        device: Target device (CPU/MPS/CUDA)

    Returns:
        Dictionary with model, preprocessor, vocab, config
    """
    device = device or torch.device('cpu')

    # Load export dict
    export_dict = torch.load(model_path, map_location=device)

    # Build config from export
    config = {
        'model': export_dict['model_config'],
        'data': export_dict['data_config'],
        'blank_id': export_dict['special_tokens']['blank_id'],
        'sos_id': export_dict['special_tokens']['sos_id'],
        'eos_id': export_dict['special_tokens']['eos_id'],
    }

    # Build model
    model = build_transducer(config)
    model.load_state_dict(export_dict['model_state_dict'])
    model.to(device)
    model.eval()

    # Build preprocessor
    data_config = export_dict['data_config']
    preprocessor = EMGPreprocessor(
        sample_rate=data_config.get('emg_sample_rate', 1000),
        target_rate=data_config.get('target_sample_rate', 689),
        notch_freq=data_config.get('notch_freq', 60),
        bandpass_low=data_config.get('bandpass_low', 20),
        bandpass_high=data_config.get('bandpass_high', 450),
        frame_length_ms=data_config.get('frame_length_ms', 8),
        frame_shift_ms=data_config.get('frame_shift_ms', 4),
    )

    # Load preprocessing stats if available
    if 'preprocessing_stats' in export_dict:
        preprocessor.load_stats(export_dict['preprocessing_stats'])

    # Build vocab dict
    vocab = {}
    if 'vocab' in export_dict:
        vocab_data = export_dict['vocab']
        vocab.update(vocab_data.get('special_tokens', {}))
        vocab.update(vocab_data.get('phonemes', {}))

    return {
        'model': model,
        'preprocessor': preprocessor,
        'vocab': vocab,
        'config': config,
        'device': device,
    }


def _get_default_vocab() -> Dict[str, Any]:
    """Get default ARPAbet vocabulary."""
    return {
        'special_tokens': {
            '<blank>': 0,
            '<sil>': 1,
            '<sos>': 2,
            '<eos>': 3,
        },
        'phonemes': {
            'AA': 4, 'AE': 5, 'AH': 6, 'AO': 7, 'AW': 8, 'AY': 9,
            'B': 10, 'CH': 11, 'D': 12, 'DH': 13, 'EH': 14, 'ER': 15,
            'EY': 16, 'F': 17, 'G': 18, 'HH': 19, 'IH': 20, 'IY': 21,
            'JH': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26, 'NG': 27,
            'OW': 28, 'OY': 29, 'P': 30, 'R': 31, 'S': 32, 'SH': 33,
            'T': 34, 'TH': 35, 'UH': 36, 'UW': 37, 'V': 38, 'W': 39,
            'Y': 40, 'Z': 41, 'ZH': 42,
        },
    }


def verify_export(
    original_checkpoint: str,
    exported_model: str,
    test_input_shape: tuple = (1, 100, 8),
) -> bool:
    """Verify exported model produces same outputs as original.

    Args:
        original_checkpoint: Path to original training checkpoint
        exported_model: Path to exported model
        test_input_shape: Shape of test EMG input

    Returns:
        True if outputs match
    """
    device = torch.device('cpu')

    # Load original
    orig_ckpt = torch.load(original_checkpoint, map_location=device)
    orig_config = orig_ckpt.get('config', {})
    orig_model = build_transducer(orig_config)
    orig_model.load_state_dict(orig_ckpt['model_state_dict'])
    orig_model.eval()

    # Load exported
    loaded = load_exported_model(exported_model, device)
    export_model = loaded['model']

    # Test forward
    test_emg = torch.randn(test_input_shape)
    session_id = torch.zeros(test_input_shape[0], dtype=torch.long)

    with torch.no_grad():
        orig_out, _ = orig_model.encoder(test_emg, session_id)
        export_out, _ = export_model.encoder(test_emg, session_id)

    # Check outputs match
    max_diff = (orig_out - export_out).abs().max().item()
    matches = max_diff < 1e-5

    if matches:
        print(f"✓ Export verified: max difference = {max_diff:.2e}")
    else:
        print(f"✗ Export mismatch: max difference = {max_diff:.2e}")

    return matches


# CLI for export
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Export model for inference')
    parser.add_argument('checkpoint', type=str, help='Path to training checkpoint')
    parser.add_argument('--output', type=str, default='exported_model.pt',
                        help='Output path for exported model')
    parser.add_argument('--vocab', type=str, default=None,
                        help='Path to vocabulary file')
    parser.add_argument('--verify', action='store_true',
                        help='Verify export correctness')
    args = parser.parse_args()

    # Export
    output_path = export_model(
        args.checkpoint,
        args.output,
        vocab_path=args.vocab,
    )

    # Optionally verify
    if args.verify:
        verify_export(args.checkpoint, output_path)
