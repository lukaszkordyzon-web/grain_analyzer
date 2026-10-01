"""Grain size distribution: D10/D50/D90 and histogram."""
from __future__ import annotations

import numpy as np


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
    """weighting: "number" (count-based) or "volume" (weight ~ d³, comparable to sieve/mass
    percentiles; assumes similar grain shape and density)."""
    if weighting not in ("number", "volume"):
        raise ValueError(weighting)
    sizes = np.asarray(sizes, float)
    weights = sizes ** 3 if weighting == "volume" else None
    return weighted_percentiles(sizes, weights)


def histogram(sizes: np.ndarray, bins: int = 20, weighting: str = "number"):
    """(counts-or-volume-fraction %, bin edges)."""
    sizes = np.asarray(sizes, float)
    w = sizes ** 3 if weighting == "volume" else None
    h, edges = np.histogram(sizes, bins=bins, weights=w)
    return (100 * h / max(h.sum(), 1e-12), edges) if weighting == "volume" else (h, edges)
