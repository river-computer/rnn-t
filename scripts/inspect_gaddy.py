"""Inspect Gaddy checkpoint structure."""
import torch

ckpt = torch.load("checkpoints/gaddy_ctc.pt", map_location="cpu")

print("=== Checkpoint Keys ===")
if isinstance(ckpt, dict):
    for k in ckpt.keys():
        print(f"  {k}")

    # Check for state_dict
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
else:
    state = ckpt

print("\n=== Model Layers ===")
for name, param in state.items():
    print(f"{name:50} {tuple(param.shape)}")
