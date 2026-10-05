"""Pick the compute device for the models. CUDA when available, else CPU.

``GRAIN_DEVICE`` overrides it ("cpu", "cuda", "mps"). Apple's MPS is NOT chosen automatically:
it is untested with SAM here and can lack some operators.
"""
from __future__ import annotations

import os


def pick_device() -> str:
    forced = os.environ.get("GRAIN_DEVICE", "").strip().lower()
    if forced:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing or broken
        return "cpu"


def describe_device() -> str:
    dev = pick_device()
    if dev.startswith("cuda"):
        try:
            import torch

            return f"GPU: {torch.cuda.get_device_name(0)}"
        except Exception:
            return "GPU (CUDA)"
    return {"cpu": "procesor (CPU) — wolniej", "mps": "Apple GPU (MPS, eksperymentalnie)"}.get(dev, dev)
