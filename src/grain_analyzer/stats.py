"""Grain size distribution: cumulative passing curve, D10/D50/D90, bounds for unmeasured fines."""
from __future__ import annotations

import numpy as np

WEIGHTINGS = {
    "number": "liczbowo",
    "area": "powierzchniowo (d²) — standard dla zdjęć z góry",
    "volume": "objętościowo (d³) — jak analiza sitowa",
}


def _weights(sizes: np.ndarray, weighting: str) -> np.ndarray | None:
    if weighting not in WEIGHTINGS:
        raise ValueError(weighting)
    return {"number": None, "area": sizes ** 2, "volume": sizes ** 3}[weighting]


def weighted_percentiles(sizes: np.ndarray, weights: np.ndarray | None,
                         percentiles=(10, 50, 90)) -> dict[str, float]:
    """Percentiles of the cumulative distribution, linearly interpolated.

    Dxx = size below which xx % of the (weighted) population lies.
    """
    sizes = np.asarray(sizes, float)
    if sizes.size == 0:
        return {f"D{p}": float("nan") for p in percentiles}
    w = np.ones_like(sizes) if weights is None else np.asarray(weights, float)
    order = np.argsort(sizes)
    s, w = sizes[order], w[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()   # mid-point cumulative fraction
    return {f"D{p}": float(np.interp(p / 100, cum, s)) for p in percentiles}


def size_distribution(sizes: np.ndarray, weighting: str = "number") -> dict[str, float]:
    """D10/D50/D90 of the measured grains under the given weighting."""
    sizes = np.asarray(sizes, float)
    return weighted_percentiles(sizes, _weights(sizes, weighting))


def cumulative_passing(sizes: np.ndarray, weighting: str = "number"):
    """(sorted sizes, cumulative % passing) - the particle size distribution curve."""
    sizes = np.sort(np.asarray(sizes, float))
    w = _weights(sizes, weighting)
    w = np.ones_like(sizes) if w is None else w
    return sizes, 100 * np.cumsum(w) / w.sum()


def passing_curve_percentiles(sizes, passing, percentiles=(10, 50, 90)) -> dict[str, float]:
    """Dxx read off a passing curve (%). NaN when the curve already starts above xx %
    (the size is below the smallest measured one)."""
    out = {}
    for p in percentiles:
        out[f"D{p}"] = (float("nan") if p < passing[0] else float(np.interp(p, passing, sizes)))
    return out


def passing_bounds(sizes_mm, areas_mm2, roi_area_mm2: float, min_size_mm: float) -> dict:
    """Area-based passing curves with the unmeasured part of the surface bracketed.

    Only rocks >= ``min_size_mm`` are measured; they cover ``c`` of the area. What is left
    (fines, voids, shadows, missed rocks) is unknown:
      * lower curve - it is ignored (distribution of the measured rocks alone, 0..100 %),
      * upper curve - ALL of it is fines finer than ``min_size_mm``.
    The true curve lies between them. D-values on the upper curve are NaN where the size
    is below the measurement limit.
    """
    order = np.argsort(sizes_mm)
    s = np.asarray(sizes_mm, float)[order]
    a = np.asarray(areas_mm2, float)[order]
    cum = np.cumsum(a)
    u = max(0.0, 1.0 - cum[-1] / roi_area_mm2)         # unmeasured area fraction
    xs = np.concatenate([[min(min_size_mm, s[0])], s])
    lower = 100 * np.concatenate([[0.0], cum / cum[-1]])
    upper = 100 * np.concatenate([[u], u + cum / roi_area_mm2])
    return {"sizes": xs, "lower": lower, "upper": upper, "unmeasured_fraction": u,
            "D_lower": passing_curve_percentiles(xs, lower),
            "D_upper": passing_curve_percentiles(xs, upper)}


def histogram(sizes: np.ndarray, bins: int = 20, weighting: str = "number"):
    """(counts-or-weight share %, bin edges)."""
    sizes = np.asarray(sizes, float)
    w = _weights(sizes, weighting)
    h, edges = np.histogram(sizes, bins=bins, weights=w)
    return (100 * h / max(h.sum(), 1e-12), edges) if w is not None else (h, edges)
