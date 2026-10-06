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


# --------------------------------------------------------------------------- fines
_NICE_EDGES_MM = (50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000)


def fraction_table(sizes_mm, areas_mm2, roi_area_mm2: float, min_size_mm: float,
                   max_edges: int = 4) -> list[dict]:
    """Share of the analysed surface by size class (area basis). Everything below the
    measurement limit - fines, voids, shadow, missed rocks - is one explicit row."""
    s = np.asarray(sizes_mm, float)
    a = np.asarray(areas_mm2, float)
    edges = [e for e in _NICE_EDGES_MM if min_size_mm * 1.15 < e < s.max()][:max_edges]
    bounds = [min_size_mm] + edges + [np.inf]
    rows = [{"label": f"< {min_size_mm:.0f} mm — niezmierzone (drobnica, szczeliny, cień)",
             "fraction": max(0.0, 1.0 - a.sum() / roi_area_mm2)}]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = (s >= lo) & (s < hi)
        rows.append({"label": (f"{lo:.0f}–{hi:.0f} mm" if np.isfinite(hi) else f"≥ {lo:.0f} mm"),
                     "fraction": float(a[sel].sum() / roi_area_mm2)})
    return rows


def rosin_rammler_fit(sizes, passing_pct, p_lo: float = 2.0, p_hi: float = 98.0) -> dict | None:
    """Fit P(d) = 1 - exp(-(d/xc)^n) by linear regression of ln(-ln(1-P)) on ln d.
    Returns None when the data do not support a sensible fit."""
    d = np.asarray(sizes, float)
    P = np.asarray(passing_pct, float) / 100
    ok = (d > 0) & (P > p_lo / 100) & (P < p_hi / 100)
    if ok.sum() < 8 or np.ptp(np.log(d[ok])) < 0.2:
        return None
    x, y = np.log(d[ok]), np.log(-np.log(1 - P[ok]))
    n, b = np.polyfit(x, y, 1)
    if n < 0.2:
        return None
    r2 = 1 - np.sum((y - (n * x + b)) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return {"xc": float(np.exp(-b / n)), "n": float(n), "r2": float(r2), "n_points": int(ok.sum())}


def rr_passing(fit: dict, d) -> np.ndarray:
    return 100 * (1 - np.exp(-(np.asarray(d, float) / fit["xc"]) ** fit["n"]))


def rr_percentile(fit: dict, p: float) -> float:
    return float(fit["xc"] * (-np.log(1 - p / 100)) ** (1 / fit["n"]))


def rr_anchored_passing(fit: dict, d, d_min: float, u: float) -> np.ndarray:
    """Passing curve BELOW the measurement limit, anchored in what was measured: ``u`` (share
    of the surface that is smaller than ``d_min``) is a fact; the Rosin-Rammler shape only
    decides how that share is spread over sizes. Continuous with the measured curve at d_min."""
    f = lambda x: 1 - np.exp(-(np.asarray(x, float) / fit["xc"]) ** fit["n"])  # noqa: E731
    return 100 * u * f(d) / f(d_min)


def rr_anchored_percentile(fit: dict, p: float, d_min: float, u: float) -> float:
    """Size below the limit at which ``p`` % of the whole surface is passed (NaN if p >= u)."""
    if p / 100 >= u:
        return float("nan")
    f_min = 1 - np.exp(-(d_min / fit["xc"]) ** fit["n"])
    target = (p / 100) / u * f_min
    return float(fit["xc"] * (-np.log(1 - target)) ** (1 / fit["n"]))


MAX_EXTRAPOLATION = 5.0       # estimates further than this factor below the limit are not given


def estimate_fines(bounds: dict, min_r2: float = 0.9, reliable_mm: float | None = None) -> dict | None:
    """Estimate of the whole-surface curve BELOW the measurement limit, as two scenarios.

    Rosin-Rammler is fitted only on the RELIABLE part of the curve (sizes >= ``reliable_mm``:
    detection there is complete), where "passing" = 1 - share of the surface covered by larger
    stones. Below it the data cannot tell two things apart, so both are given:
      * D     - everything unmeasured is fines smaller than the limit: the RR shape is rescaled to
                the measured unmeasured share u (never claims more than was measured),
      * D_alt - stones are MISSED just above the limit: the plain RR curve continues downwards
                (so the plateau of the measured curve is missed stones).
    D <= D_alt; the truth lies between them. Estimates further than 5x below the limit are not
    quoted (NaN). Returns None when no sensible fit exists."""
    sizes, upper = np.asarray(bounds["sizes"], float), np.asarray(bounds["upper"], float)
    fit = None
    if reliable_mm:
        keep = sizes >= reliable_mm
        fit = rosin_rammler_fit(sizes[keep], upper[keep])
    used = "reliable" if fit is not None else "all"
    if fit is None:
        fit = rosin_rammler_fit(sizes, upper)
    if fit is None or fit["r2"] < min_r2:
        return None
    d_min, u = float(sizes[0]), bounds["unmeasured_fraction"]
    floor = d_min / MAX_EXTRAPOLATION                  # lowest size we are willing to quote
    D, D_alt = {}, {}
    for p in (10, 50, 90):
        measured = bounds["D_upper"][f"D{p}"]          # reachable on the measured part: use it
        if np.isfinite(measured):
            v = alt = measured
        else:
            v = rr_anchored_percentile(fit, p, d_min, u)
            alt = rr_percentile(fit, p)
        D[f"D{p}"] = v if v >= floor else float("nan")     # NaN = out of reach
        D_alt[f"D{p}"] = alt if alt >= floor else float("nan")
    return {"fit": fit, "D": D, "D_alt": D_alt, "floor_mm": floor, "fit_on": used,
            "reliable_mm": reliable_mm}


def completeness_limit(sizes_mm, min_size_mm: float, fallback_factor: float = 1.7,
                       min_stones: int = 40) -> tuple[float, str]:
    """Size above which stones are detected (nearly) completely.

    In a pile of broken rock the NUMBER of stones keeps growing towards smaller sizes, so where the
    count per log-size class peaks and then falls, detection is losing stones. That turnover is the
    completeness limit. With too few stones (or no clear peak) fall back to a fixed factor of the
    measurement limit. Returns (limit_mm, "turnover" | "fallback")."""
    s = np.asarray(sizes_mm, float)
    s = s[s > 0]
    fallback = (float(min_size_mm * fallback_factor), "fallback")
    if len(s) < min_stones or s.max() / s.min() < 2:
        return fallback
    edges = np.geomspace(s.min(), s.max(), 15)
    cnt, _ = np.histogram(s, edges)
    sm = np.convolve(np.pad(cnt.astype(float), 1, mode="edge"), [1, 2, 1], mode="valid") / 4
    centers = np.sqrt(edges[:-1] * edges[1:])
    ok = edges[:-1] < np.median(s)                    # the turnover lies on the small side
    if not ok.any():
        return fallback
    k = int(np.argmax(np.where(ok, sm, -1)))
    if k == 0 or sm[k] < 3:                           # monotonic growth: no turnover visible
        return fallback
    return float(max(centers[k], min_size_mm * 1.15)), "turnover"


def histogram(sizes: np.ndarray, bins: int = 20, weighting: str = "number"):
    """(counts-or-weight share %, bin edges)."""
    sizes = np.asarray(sizes, float)
    w = _weights(sizes, weighting)
    h, edges = np.histogram(sizes, bins=bins, weights=w)
    return (100 * h / max(h.sum(), 1e-12), edges) if w is not None else (h, edges)
