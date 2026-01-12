#!/usr/bin/env python3
import argparse
import pickle
import inspect
import torch
from argparse import Namespace


def load_checkpoint_trusted(path: str):
    # PyTorch 2.6+ defaults weights_only=True, so we must explicitly allow pickle objects here.
    with torch.serialization.safe_globals([Namespace]):
        return torch.load(path, map_location="cpu", weights_only=False)


def build_model_from_checkpoint(ckpt: dict):
    """
    For esm versions where:
      esm.pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data=None)
    """
    import re
    import esm.pretrained as ep

    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected dict checkpoint, got: {type(ckpt)}")
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model'. Keys: {sorted(ckpt.keys())}")

    # 1) Try to derive model_name from checkpoint metadata
    candidates = []

    # ckpt["cfg"] is often an OmegaConf / dict-like with model info
    cfg = ckpt.get("cfg", None)
    if cfg is not None:
        # Try common places
        for keypath in [
            ("model", "name"),
            ("model", "arch"),
            ("arch",),
            ("model_name",),
            ("name",),
        ]:
            try:
                cur = cfg
                for k in keypath:
                    cur = cur[k]
                if isinstance(cur, str):
                    candidates.append(cur)
            except Exception:
                pass

    args = ckpt.get("args", None)
    if args is not None:
        for attr in ["model_name", "arch", "name"]:
            val = getattr(args, attr, None)
            if isinstance(val, str):
                candidates.append(val)

    # 2) If filename-like names got baked in, normalize a bit
    normed = []
    for c in candidates:
        c2 = c.strip()
        c2 = re.sub(r"\.pt$", "", c2)
        normed.append(c2)
    candidates = normed

    # 3) Add known likely ESM2-35M names
    # The most common 35M checkpoint is esm2_t12_35M_UR50D
    candidates.extend([
        "esm2_t12_35M_UR50D",
        "esm2_t12_35M",
        "esm2_35M",
    ])

    # de-duplicate while preserving order
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    fn = ep.load_model_and_alphabet_core  # confirmed signature in your error

    last_err = None
    for model_name in candidates:
        try:
            model, alphabet = fn(model_name, ckpt, None)
            return model, alphabet
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Could not reconstruct model with any candidate model_name.\n"
        f"Tried: {candidates}\n"
        f"Last error: {repr(last_err)}"
    )



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pt", required=True, help="Input ESM checkpoint (.pt) that loads as a dict")
    ap.add_argument("--out_pt", required=True, help="Output path to save the full pickled nn.Module")
    ap.add_argument("--out_alphabet", default=None, help="Optional: also save alphabet to a pickle file")
    args = ap.parse_args()

    ckpt = load_checkpoint_trusted(args.in_pt)
    print("Loaded checkpoint type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Checkpoint keys:", sorted(ckpt.keys()))

    model, alphabet = build_model_from_checkpoint(ckpt)
    model.eval()

    # Save the WHOLE MODULE (pickled).
    torch.save(model, args.out_pt)
    print(f"Saved full module to: {args.out_pt}")

    if args.out_alphabet is not None:
        with open(args.out_alphabet, "wb") as f:
            pickle.dump(alphabet, f)
        print(f"Saved alphabet to: {args.out_alphabet}")


if __name__ == "__main__":
    main()

"""
python3 Convert_ESM_checkpoint_to_module.py \
  --in_pt /home/rohan/ESM/Models/ESM2_35M.pt \
  --out_pt /home/rohan/ESM/Models/ESM2_35M_FULLMODULE.pt \
  --out_alphabet /home/rohan/ESM/Models/ESM2_alphabet.pkl
"""