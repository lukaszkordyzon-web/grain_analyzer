"""Relative depth with Depth Anything V2 (pretrained, no training)."""
from __future__ import annotations

import numpy as np

DEPTH_MODELS = {
    "Depth Anything V2 Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth Anything V2 Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth Anything V2 Large": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthEstimator:
    """Returns a relative (non-metric) map in [0, 1]; larger = closer to the camera."""

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._pipe = pipeline("depth-estimation", model=self.model_id, device=device)
        return self._pipe

    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        depth = self._load()(Image.fromarray(image_rgb))["predicted_depth"]
        depth = np.asarray(depth.squeeze().cpu() if hasattr(depth, "cpu") else depth, np.float32)
        if depth.shape != image_rgb.shape[:2]:
            import cv2
            depth = cv2.resize(depth, (image_rgb.shape[1], image_rgb.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        lo, hi = float(depth.min()), float(depth.max())
        return (depth - lo) / (hi - lo + 1e-9)
