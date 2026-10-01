"""End-to-end analysis: scale -> SAM -> (depth) -> measurements -> statistics."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from . import scale as scale_mod
from .depth import DepthEstimator
from .measure import SIZE_METRICS, depth_outliers, measure_grains
from .segmentation import Segmenter, select_grain_masks
from .stats import size_distribution


@dataclass
class Params:
    max_side: int = 1600                 # working resolution (long side, px)
    # scale
    marker_size_mm: float = 20.0
    marker_dict: str = "4x4_50"
    marker_id: int | None = None
    manual_mm_per_px: float | None = None  # used when the marker is not found / forced
    # mask filtering
    min_area_px: int = 150
    max_area_frac: float = 0.05
    max_overlap: float = 0.5
    # depth
    use_depth: bool = True
    depth_reject_sigma: float | None = 3.0  # None = do not drop depth outliers
    # statistics
    size_metric: str = "ecd"             # key of SIZE_METRICS
    weighting: str = "number"            # "number" | "volume"


@dataclass
class AnalysisResult:
    image: np.ndarray                    # working-resolution RGB
    scale: scale_mod.ScaleResult
    masks: list[np.ndarray]
    grains: pd.DataFrame
    depth: np.ndarray | None
    size_column: str
    percentiles: dict[str, float] = field(default_factory=dict)
    n_rejected_depth: int = 0

    @property
    def sizes(self) -> np.ndarray:
        return self.grains[self.size_column].to_numpy()


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    f = max_side / max(h, w)
    if f >= 1:
        return image
    return cv2.resize(image, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)


def resolve_scale(image: np.ndarray, p: Params, resize_factor: float = 1.0) -> scale_mod.ScaleResult:
    """Marker-based scale, or the manual one. ``manual_mm_per_px`` is given for the original
    photo, so it is divided by ``resize_factor`` (working px / original px)."""
    if p.manual_mm_per_px is not None:
        return scale_mod.manual_scale(p.manual_mm_per_px / resize_factor)
    res = scale_mod.detect_marker(image, p.marker_size_mm, p.marker_dict, p.marker_id)
    if res is None:
        raise ValueError("Nie znaleziono znacznika ArUco w kadrze — popraw zdjęcie/słownik "
                         "albo podaj skalę ręcznie (mm/px).")
    return res


def analyze(image_rgb: np.ndarray, p: Params, segmenter: Segmenter,
            depth_estimator: DepthEstimator | None = None) -> AnalysisResult:
    if p.size_metric not in SIZE_METRICS:
        raise ValueError(f"size_metric must be one of {list(SIZE_METRICS)}")
    image = resize_max_side(image_rgb, p.max_side)
    factor = image.shape[1] / image_rgb.shape[1]
    scale = resolve_scale(image, p, factor)

    candidates = segmenter.segment(image)
    masks = select_grain_masks(
        candidates, min_area_px=p.min_area_px, max_area_frac=p.max_area_frac,
        exclude=scale.marker_mask(image.shape, pad_px=5), max_overlap=p.max_overlap)

    depth = depth_estimator.estimate(image) if (p.use_depth and depth_estimator) else None
    grains = measure_grains(masks, scale.mm_per_px, depth)

    n_rej = 0
    if depth is not None and p.depth_reject_sigma is not None and len(grains) >= 5:
        bad = depth_outliers(grains, p.depth_reject_sigma)
        n_rej = int(bad.sum())
        grains = grains[~bad]

    masks = [masks[i] for i in grains["mask_idx"]]
    grains = grains.drop(columns="mask_idx").reset_index(drop=True)
    grains.insert(0, "id", np.arange(1, len(grains) + 1))   # id n <-> masks[n-1]

    col = {"ecd": "ecd_mm", "feret_min": "feret_min_mm", "feret_max": "feret_max_mm"}[p.size_metric]
    perc = size_distribution(grains[col].to_numpy(), p.weighting) if len(grains) else {}
    return AnalysisResult(image, scale, masks, grains, depth, col, perc, n_rej)
