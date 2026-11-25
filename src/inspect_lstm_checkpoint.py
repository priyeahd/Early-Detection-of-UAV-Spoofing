
import torch

checkpoint = torch.load(
    "models/msv_stlstm_px4_relative_best.pth",
    map_location="cpu",
    weights_only=False
)

print("CHECKPOINT KEYS:")
print(list(checkpoint.keys()))

print("\nMODEL WEIGHTS:")

for name, weight in checkpoint["model_state_dict"].items():
    print(name, tuple(weight.shape))

print("\nINSPECTION COMPLETE")
