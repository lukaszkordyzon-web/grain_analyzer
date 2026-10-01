"""Grain instance segmentation with SAM (automatic mask generation, no training)."""
from __future__ import annotations

from typing import Protocol

import numpy as np

SAM_MODELS = {
    "SAM ViT-B (szybki)": "facebook/sam-vit-base",
    "SAM ViT-L": "facebook/sam-vit-large",
    "SAM ViT-H (najdokładniejszy)": "facebook/sam-vit-huge",
}


class Segmenter(Protocol):
    def segment(self, image_rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Return [(bool mask HxW, quality score)] for every candidate region."""


class SamSegmenter:
    """Wraps the HF ``mask-generation`` pipeline (SAM). Model is loaded lazily."""

    def __init__(self, model_id: str = "facebook/sam-vit-base", device: str | None = None,
                 points_per_batch: int = 32, pred_iou_thresh: float = 0.88,
                 stability_score_thresh: float = 0.92):
        self.model_id = model_id
        self.device = device
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._pipe = pipeline("mask-generation", model=self.model_id, device=device)
        return self._pipe

    def segment(self, image_rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        from PIL import Image

        out = self._load()(
            Image.fromarray(image_rgb),
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
        )
        scores = out.get("scores", [1.0] * len(out["masks"]))
        return [(np.asarray(m, bool), float(s)) for m, s in zip(out["masks"], scores)]


def select_grain_masks(candidates: list[tuple[np.ndarray, float]], *, min_area_px: int,
                       max_area_frac: float, exclude: np.ndarray | None = None,
                       max_overlap: float = 0.5) -> list[np.ndarray]:
    """Turn raw SAM candidates into a clean, non-duplicated set of grain masks.

    SAM returns nested/overlapping masks (grain, part of grain, grain+neighbour).
    Keep the best-scoring ones greedily and drop any mask that overlaps an already
    accepted one by more than ``max_overlap`` of its own area.
    """
    h, w = candidates[0][0].shape if candidates else (0, 0)
    max_area = max_area_frac * h * w
    taken = np.zeros((h, w), bool)
    kept: list[np.ndarray] = []
    for mask, _ in sorted(candidates, key=lambda c: -c[1]):
        area = int(mask.sum())
        if area < min_area_px or area > max_area:
            continue
        if exclude is not None and (mask & exclude).sum() > 0.2 * area:
            continue
        if (mask & taken).sum() > max_overlap * area:
            continue
        kept.append(mask)
        taken |= mask
    return kept
