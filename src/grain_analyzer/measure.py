"""Per-grain geometry (px -> mm) and depth-based features."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

SIZE_METRICS = {
    "ecd": "Średnica równoważna koła (ECD)",
    "feret_min": "Feret min (wymiar „sitowy”)",
    "feret_max": "Feret max (długość)",
}


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def feret_diameters(contour: np.ndarray, n_angles: int = 180) -> tuple[float, float]:
    """(max, min) Feret diameter in px via projection of the convex hull."""
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    theta = np.linspace(0, np.pi, n_angles, endpoint=False)
    dirs = np.stack([np.cos(theta), np.sin(theta)])          # (2, A)
    proj = hull @ dirs                                        # (N, A)
    widths = proj.max(0) - proj.min(0)
    # +1: contour pixels are pixel centres, the grain extends half a pixel beyond
    return float(widths.max() + 1), float(widths.min() + 1)


def measure_grains(masks: list[np.ndarray], mm_per_px: float,
                   depth: np.ndarray | None = None, ring_px: int = 15) -> pd.DataFrame:
    """One row per grain; ``mask_idx`` points back into ``masks``. Lengths in mm, area in mm², ``rel_height`` unit-less."""
    union = np.any(masks, axis=0) if masks else None
    rows = []
    for idx, mask in enumerate(masks):
        cnt = _largest_contour(mask)
        if cnt is None or len(cnt) < 5:
            continue
        area_px = float(mask.sum())
        perim = cv2.arcLength(cnt, True)
        fmax, fmin = feret_diameters(cnt)
        (_, _), (ea, eb), _ = cv2.fitEllipse(cnt)
        m = float(mm_per_px)
        row = {
            "mask_idx": idx,
            "area_mm2": area_px * m * m,
            "ecd_mm": 2 * np.sqrt(area_px / np.pi) * m,
            "feret_max_mm": fmax * m,
            "feret_min_mm": fmin * m,
            "major_mm": max(ea, eb) * m,
            "minor_mm": min(ea, eb) * m,
            "aspect_ratio": fmax / max(fmin, 1e-9),
            "circularity": float(min(1.0, 4 * np.pi * area_px / max(perim, 1e-9) ** 2)),
        }
        ys, xs = np.nonzero(mask)
        row["cx_px"], row["cy_px"] = float(xs.mean()), float(ys.mean())
        if depth is not None:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1,) * 2)
            ring = cv2.dilate(mask.astype(np.uint8), k).astype(bool) & ~union
            row["depth_mean"] = float(depth[mask].mean())
            row["rel_height"] = (float(depth[mask].mean() - np.median(depth[ring]))
                                 if ring.any() else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def depth_outliers(grains: pd.DataFrame, sigma: float = 3.0) -> np.ndarray:
    """Boolean flags for grains much farther from the camera than the typical grain
    (likely background / pit between grains). Robust z-score on mean depth."""
    d = grains["depth_mean"].to_numpy()
    med = np.median(d)
    mad = 1.4826 * np.median(np.abs(d - med)) + 1e-9
    return (d - med) / mad < -sigma
