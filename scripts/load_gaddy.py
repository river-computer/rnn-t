"""Test loading Gaddy pretrained weights."""
import torch
import sys
sys.path.insert(0, '.')

from src.model.gaddy_encoder import GaddyEncoder

# Load pretrained
print("Loading Gaddy pretrained weights...")
model = GaddyEncoder.from_pretrained('checkpoints/gaddy_ctc.pt')
print("Success!")

# Print model info
print(f"\nModel architecture:")
print(f"  - Conv blocks: 3 ResBlocks (8x downsampling)")
print(f"  - Transformer: {model.num_layers} layers, {model.d_model}-dim")
print(f"  - Vocab size: {model.w_out.out_features}")

# Test forward pass
print("\nTesting forward pass...")
batch_size = 2
seq_len = 800  # Must be divisible by 8
emg_channels = 8

x = torch.randn(batch_size, seq_len, emg_channels)
lengths = torch.tensor([800, 600])

logits, out_lengths = model(x, lengths)
print(f"  Input:  ({batch_size}, {seq_len}, {emg_channels})")
print(f"  Output: {tuple(logits.shape)}")
print(f"  Lengths: {out_lengths.tolist()}")

# Test encoder output (for RNN-T)
encoder_out, enc_lengths = model.get_encoder_output(x, lengths)
print(f"\nEncoder output (for RNN-T):")
print(f"  Shape: {tuple(encoder_out.shape)}")
print(f"  Dim: {encoder_out.shape[-1]} (768-dim, usable for joiner)")

print("\nPretrained Gaddy encoder loaded successfully!")
print("You can now use this for RNN-T training (skip CTC pretraining).")
