import torch

pt_path = "/home/rohan/ESM/Models/ESM2_35M.pt"

with torch.serialization.safe_globals([Namespace]):
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    print(type(ckpt))
    if isinstance(ckpt, dict):
        print("keys:", sorted(ckpt.keys()))
