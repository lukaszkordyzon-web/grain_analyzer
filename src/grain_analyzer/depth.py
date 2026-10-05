"""Relative depth with Depth Anything V2 (pretrained, no training)."""
from __future__ import annotations

import numpy as np

DEPTH_MODELS = {
    "Depth Anything V2 Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth Anything V2 Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth Anything V2 Large": "depth-anything/Depth-Anything-V2-Large-hf",
}


METRIC_DEPTH_MODELS = {
    "Depth Anything V2 Metric Outdoor Small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "Depth Anything V2 Metric Outdoor Base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
}


class DepthEstimator:
    """Returns a relative (non-metric) map in [0, 1]; larger = closer to the camera."""

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str | None = None, metric: bool = False):
        self.model_id = model_id
        self.metric = metric          # True: output in metres (Metric-* checkpoints)
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline

            from .device import pick_device

            device = self.device or pick_device()
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
        if self.metric:
            return depth
        lo, hi = float(depth.min()), float(depth.max())
        return (depth - lo) / (hi - lo + 1e-9)
