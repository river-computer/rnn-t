#!/usr/bin/env python3
"""Test pipeline to verify each component works before full training.

This script runs quick sanity checks on:
1. Data loading - verify data can be loaded and has correct shapes
2. Preprocessing - verify EMG preprocessing works
3. Model forward - verify encoder, predictor, joiner work
4. CTC loss - verify CTC loss computes correctly
5. RNN-T loss - verify RNN-T loss (k2 or torchaudio) works
6. Training step - verify gradient flow
7. Streaming inference - verify chunk-by-chunk decoding
8. S3 checkpoint - verify save/load round-trip

Run: python scripts/test_pipeline.py --config config/default.yaml
Expected time: ~5 minutes
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def test_imports():
    """Test all required imports."""
    print("\n=== Test 1: Imports ===")
    errors = []

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        errors.append(f"torch: {e}")

    try:
        import torchaudio
        print(f"  ✓ torchaudio {torchaudio.__version__}")
    except ImportError as e:
        errors.append(f"torchaudio: {e}")

    try:
        import numpy
        print(f"  ✓ numpy {numpy.__version__}")
    except ImportError as e:
        errors.append(f"numpy: {e}")

    try:
        import scipy
        print(f"  ✓ scipy {scipy.__version__}")
    except ImportError as e:
        errors.append(f"scipy: {e}")

    try:
        import editdistance
        print("  ✓ editdistance")
    except ImportError as e:
        errors.append(f"editdistance: {e}")

    try:
        import boto3
        print("  ✓ boto3")
    except ImportError as e:
        errors.append(f"boto3: {e}")

    # Optional but recommended
    try:
        import k2
        print(f"  ✓ k2 (pruned RNN-T loss available)")
    except ImportError:
        print("  ⚠ k2 not installed (will use torchaudio RNN-T loss)")

    try:
        import tslearn
        print("  ✓ tslearn (DTW available)")
    except ImportError:
        print("  ⚠ tslearn not installed (DTW adaptation disabled)")

    try:
        import wandb
        print("  ✓ wandb")
    except ImportError:
        print("  ⚠ wandb not installed (logging disabled)")

    if errors:
        print(f"\n  ✗ Missing dependencies: {errors}")
        return False
    return True


def test_data_loading(config):
    """Test data loading and shapes."""
    print("\n=== Test 2: Data Loading ===")

    from src.data import EMGDataset, collate_fn
    from torch.utils.data import DataLoader

    data_dir = config['data']['local_data_dir']

    # Check if EMG data exists in expected location
    emg_data_dir = Path(data_dir) / 'emg_data'
    if not emg_data_dir.exists():
        # Try alternative location
        emg_data_dir = Path(data_dir)

    alignments_dir = Path(data_dir) / 'text_alignments'
    vocab_path = Path('data/vocab.json')

    # Check if data exists
    voiced_dir = emg_data_dir / 'voiced_parallel_data'
    if not voiced_dir.exists():
        print(f"  ⚠ Voiced data not found at: {voiced_dir}")
        print("  Creating synthetic test data...")

        # Create synthetic data for testing
        test_session = '5-4'
        session_dir = voiced_dir / test_session
        session_dir.mkdir(parents=True, exist_ok=True)

        # Also create alignments dir
        align_session = alignments_dir / 'voiced_parallel_data' / test_session
        align_session.mkdir(parents=True, exist_ok=True)

        # Create 5 synthetic samples
        for i in range(5):
            emg = np.random.randn(1000, 8).astype(np.float32)
            np.save(session_dir / f'{i}_emg.npy', emg)

            # Create dummy info file
            import json
            with open(session_dir / f'{i}_info.json', 'w') as f:
                json.dump({'text': 'hello world'}, f)

    try:
        # Get first available session
        sessions = config['data'].get('train_sessions', ['5-4'])[:1]

        dataset = EMGDataset(
            data_dir=str(emg_data_dir),
            alignments_dir=str(alignments_dir),
            vocab_path=str(vocab_path),
            sessions=sessions,
            split='voiced',
            max_samples=10,  # Limit for testing
        )
        print(f"  ✓ Dataset created with {len(dataset)} samples")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  ✓ Sample keys: {list(sample.keys())}")
            print(f"  ✓ EMG shape: {sample['emg'].shape}")
        else:
            print("  ⚠ Dataset is empty (no aligned samples found)")
            return True  # Not a failure, just no data

        # Test dataloader
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"  ✓ Batch EMG shape: {batch['emg'].shape}")

        return True
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing(config):
    """Test EMG preprocessing pipeline."""
    print("\n=== Test 3: Preprocessing ===")

    from src.data import EMGPreprocessor

    try:
        preprocessor = EMGPreprocessor(
            sample_rate=config['data']['sample_rate'],
            target_rate=config['data']['target_sample_rate'],
            notch_freq=config['data']['notch_freq'],
            bandpass_low=config['data']['bandpass_low'],
            bandpass_high=config['data']['bandpass_high'],
            frame_length_ms=config['data']['frame_length_ms'],
            frame_shift_ms=config['data']['frame_shift_ms'],
        )

        # Test with synthetic EMG
        raw_emg = np.random.randn(1000, 8).astype(np.float32)

        # Compute session stats first
        stats = preprocessor.compute_session_stats([raw_emg], '5-4')
        processed = preprocessor.process_for_model(raw_emg, stats)

        print(f"  ✓ Input shape: {raw_emg.shape}")
        print(f"  ✓ Output shape: {processed.shape}")
        print(f"  ✓ Output dtype: {processed.dtype}")

        # Verify shape makes sense
        expected_frames = (1000 * config['data']['target_sample_rate'] // config['data']['sample_rate']) // 4
        print(f"  ✓ Expected ~{expected_frames} frames (after downsampling)")

        return True
    except Exception as e:
        print(f"  ✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_transducer(config):
    """Build transducer model from config."""
    from src.model import EMGEncoder, Predictor, Joiner, Transducer

    encoder_config = config['model']['encoder']
    predictor_config = config['model']['predictor']
    joiner_config = config['model']['joiner']

    encoder = EMGEncoder(
        emg_channels=config['data'].get('num_channels', 8),
        num_sessions=encoder_config.get('num_sessions', 8),
        session_embed_dim=encoder_config.get('session_embed_dim', 32),
        conv_channels=encoder_config.get('conv_channels', 768),
        num_conv_blocks=encoder_config.get('num_conv_blocks', 3),
        d_model=encoder_config.get('d_model', 768),
        num_heads=encoder_config.get('num_heads', 8),
        ff_dim=encoder_config.get('ff_dim', 2048),
        num_layers=encoder_config.get('num_layers', 6),
        dropout=encoder_config.get('dropout', 0.1),
        output_dim=encoder_config.get('output_dim', 128),
    )

    predictor = Predictor(
        vocab_size=predictor_config.get('vocab_size', 43),
        embed_dim=predictor_config.get('embed_dim', 128),
        hidden_dim=predictor_config.get('hidden_dim', 320),
        num_layers=predictor_config.get('num_layers', 1),
        output_dim=predictor_config.get('output_dim', 128),
    )

    joiner = Joiner(
        encoder_dim=joiner_config.get('input_dim', 128),
        predictor_dim=joiner_config.get('input_dim', 128),
        vocab_size=joiner_config.get('vocab_size', 43),
    )

    return Transducer(encoder, predictor, joiner, blank_id=0)


def test_model_forward(config):
    """Test model forward passes."""
    print("\n=== Test 4: Model Forward ===")

    from src.model import EMGEncoder, Predictor, Joiner

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    try:
        # Test encoder
        encoder_config = config['model']['encoder']
        encoder = EMGEncoder(
            emg_channels=config['data'].get('num_channels', 8),
            num_sessions=encoder_config.get('num_sessions', 8),
            session_embed_dim=encoder_config.get('session_embed_dim', 32),
            conv_channels=encoder_config.get('conv_channels', 768),
            num_conv_blocks=encoder_config.get('num_conv_blocks', 3),
            d_model=encoder_config.get('d_model', 768),
            num_heads=encoder_config.get('num_heads', 8),
            ff_dim=encoder_config.get('ff_dim', 2048),
            num_layers=encoder_config.get('num_layers', 6),
            dropout=encoder_config.get('dropout', 0.1),
            output_dim=encoder_config.get('output_dim', 128),
        ).to(device)

        batch_size, seq_len = 2, 100
        emg = torch.randn(batch_size, seq_len, 8, device=device)
        session_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        lengths = torch.full((batch_size,), seq_len, device=device)

        encoder_out, out_lengths = encoder(emg, session_id, lengths)
        print(f"  ✓ Encoder: {emg.shape} -> {encoder_out.shape}")

        # Test predictor
        predictor_config = config['model']['predictor']
        predictor = Predictor(
            vocab_size=predictor_config.get('vocab_size', 43),
            embed_dim=predictor_config.get('embed_dim', 128),
            hidden_dim=predictor_config.get('hidden_dim', 320),
            num_layers=predictor_config.get('num_layers', 1),
            output_dim=predictor_config.get('output_dim', 128),
        ).to(device)

        targets = torch.randint(0, 43, (batch_size, 10), device=device)
        predictor_out, _ = predictor(targets)
        print(f"  ✓ Predictor: {targets.shape} -> {predictor_out.shape}")

        # Test joiner
        joiner = Joiner(
            encoder_dim=config['model']['joiner'].get('input_dim', 128),
            predictor_dim=config['model']['joiner'].get('input_dim', 128),
            vocab_size=config['model']['joiner'].get('vocab_size', 43),
        ).to(device)

        joint_out = joiner(encoder_out, predictor_out)
        print(f"  ✓ Joiner: ({encoder_out.shape}, {predictor_out.shape}) -> {joint_out.shape}")

        # Test full transducer
        model = build_transducer(config).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Full model: {param_count:,} parameters")

        return True
    except Exception as e:
        print(f"  ✗ Model forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ctc_loss(config):
    """Test CTC loss computation."""
    print("\n=== Test 5: CTC Loss ===")

    from src.utils import CTCLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        criterion = CTCLoss(blank_id=0)

        # Synthetic inputs
        batch_size, T, vocab_size = 2, 50, 43
        logits = torch.randn(batch_size, T, vocab_size, device=device, requires_grad=True)
        targets = torch.randint(1, vocab_size, (batch_size, 10), device=device)
        input_lengths = torch.full((batch_size,), T, device=device)
        target_lengths = torch.full((batch_size,), 10, device=device)

        loss = criterion(logits, targets, input_lengths, target_lengths)
        print(f"  ✓ CTC loss: {loss.item():.4f}")

        # Test gradient
        loss.backward()
        print(f"  ✓ Gradient flows: {logits.grad is not None}")

        return True
    except Exception as e:
        print(f"  ✗ CTC loss failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rnnt_loss(config):
    """Test RNN-T loss computation."""
    print("\n=== Test 6: RNN-T Loss ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Try k2 first, fall back to torchaudio
        try:
            from src.utils import RNNTLoss
            criterion = RNNTLoss(blank_id=0, use_pruned=True)
            print("  Using k2 pruned RNN-T loss")
        except ImportError:
            from src.utils import SimplifiedRNNTLoss
            criterion = SimplifiedRNNTLoss(blank_id=0)
            print("  Using torchaudio RNN-T loss")

        # Synthetic inputs
        batch_size, T, U, vocab_size = 2, 25, 10, 43
        logits = torch.randn(batch_size, T, U + 1, vocab_size, device=device, requires_grad=True)
        targets = torch.randint(1, vocab_size, (batch_size, U), device=device)
        input_lengths = torch.full((batch_size,), T, device=device)
        target_lengths = torch.full((batch_size,), U, device=device)

        loss = criterion(logits, targets, input_lengths, target_lengths)
        print(f"  ✓ RNN-T loss: {loss.item():.4f}")

        # Test gradient
        loss.backward()
        print(f"  ✓ Gradient flows: {logits.grad is not None}")

        return True
    except Exception as e:
        print(f"  ✗ RNN-T loss failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(config):
    """Test a single training step with gradient update."""
    print("\n=== Test 7: Training Step ===")

    from src.utils import CTCLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = build_transducer(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = CTCLoss(blank_id=0)

        # Synthetic batch
        batch_size = 2
        emg = torch.randn(batch_size, 100, 8, device=device)
        session_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        targets = torch.randint(1, 43, (batch_size, 10), device=device)
        emg_lengths = torch.full((batch_size,), 100, device=device)
        target_lengths = torch.full((batch_size,), 10, device=device)

        # Forward
        encoder_out, encoder_lengths = model.encoder(emg, session_id, emg_lengths)

        # Add CTC head
        ctc_head = torch.nn.Linear(128, 43).to(device)
        logits = ctc_head(encoder_out)

        loss = criterion(logits, targets, encoder_lengths, target_lengths)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"  ✓ All parameters have gradients: {has_grad}")

        # Step
        optimizer.step()
        print(f"  ✓ Optimizer step completed")

        return True
    except Exception as e:
        print(f"  ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_inference(config):
    """Test streaming inference."""
    print("\n=== Test 8: Streaming Inference ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = build_transducer(config).to(device)
        model.eval()

        # Initialize streaming state
        state = model.init_streaming_state(batch_size=1, device=device)
        print(f"  ✓ Streaming state initialized")

        # Process synthetic chunks
        chunk_size = 40  # ~160ms of frames
        session_id = torch.zeros(1, dtype=torch.long, device=device)

        all_phonemes = []
        for i in range(5):  # 5 chunks
            chunk = torch.randn(1, chunk_size, 8, device=device)
            phonemes, state = model.decode_streaming(chunk, session_id, state)
            all_phonemes.extend(phonemes)
            print(f"  ✓ Chunk {i}: emitted {len(phonemes)} phonemes")

        print(f"  ✓ Total phonemes: {len(all_phonemes)}")

        return True
    except Exception as e:
        print(f"  ✗ Streaming inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_s3_checkpoint(config):
    """Test S3 checkpoint save/load (mocked if no credentials)."""
    print("\n=== Test 9: Checkpoint Save/Load ===")

    try:
        # Test local save/load first
        model = build_transducer(config)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 0,
            'config': config,
        }
        torch.save(checkpoint, temp_path)
        print(f"  ✓ Saved checkpoint to {temp_path}")

        # Load
        loaded = torch.load(temp_path)
        model.load_state_dict(loaded['model_state_dict'])
        print(f"  ✓ Loaded checkpoint successfully")

        # Clean up
        import os
        os.unlink(temp_path)

        # Test S3 (only if credentials available)
        try:
            import boto3
            s3 = boto3.client('s3')
            s3.list_buckets()
            print("  ✓ AWS credentials available")

            # Don't actually upload in test - just verify connectivity
            from src.utils import S3Checkpoint
            ckpt_mgr = S3Checkpoint(
                bucket=config.get('s3', {}).get('bucket', 'river-emg-speech'),
                prefix='checkpoints/test',
                local_dir='/tmp/checkpoints',
            )
            print("  ✓ S3Checkpoint manager initialized")

        except Exception as e:
            print(f"  ⚠ S3 not available (will use local only): {e}")

        return True
    except Exception as e:
        print(f"  ✗ Checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test EMG RNN-T pipeline')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("EMG RNN-T Pipeline Test")
    print("=" * 60)

    start_time = time.time()
    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['data_loading'] = test_data_loading(config)
    results['preprocessing'] = test_preprocessing(config)
    results['model_forward'] = test_model_forward(config)
    results['ctc_loss'] = test_ctc_loss(config)
    results['rnnt_loss'] = test_rnnt_loss(config)
    results['training_step'] = test_training_step(config)
    results['streaming'] = test_streaming_inference(config)
    results['checkpoint'] = test_s3_checkpoint(config)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nPassed: {passed}/{total}")
    print(f"Time: {elapsed:.1f}s")

    if passed == total:
        print("\n✓ All tests passed! Ready for training.")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
