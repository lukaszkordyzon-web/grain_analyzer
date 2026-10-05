"""Overlay with contours and the size histogram."""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .stats import (cumulative_passing, histogram, passing_curve_percentiles,  # noqa: E402
                    rr_passing, rr_percentile)


def draw_overlay(result, show_ids: bool = False, thickness: int = 1) -> np.ndarray:
    out = result.image.copy()
    if result.contours is not None:
        cv2.drawContours(out, result.contours, -1, (0, 255, 0), thickness)
    for m in result.masks:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 0), thickness)
    ann = result.annotations
    if "roi" in ann:
        x0, y0, x1, y1 = ann["roi"]
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 160, 0), 2)
    if "head" in ann:
        cv2.line(out, tuple(map(int, ann["head"])), tuple(map(int, ann["feet"])), (255, 0, 255), 2)
    if result.scale.corners is not None:
        cv2.polylines(out, [result.scale.corners.astype(np.int32)], True, (255, 0, 0), 2)
    if show_ids:
        for _, r in result.grains.iterrows():
            cv2.putText(out, str(int(r["id"])), (int(r["cx_px"]) - 6, int(r["cy_px"]) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def depth_preview(depth: np.ndarray) -> np.ndarray:
    col = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(col, cv2.COLOR_BGR2RGB)


def plot_histogram(sizes: np.ndarray, percentiles: dict[str, float], label: str,
                   weighting: str = "number", bins: int = 20):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    h, edges = histogram(sizes, bins, weighting)
    ax.bar(edges[:-1], h, width=np.diff(edges), align="edge", color="#8fa8c8", edgecolor="white")
    for (name, v), c in zip(percentiles.items(), ("#2a9d8f", "#e76f51", "#6a4c93")):
        ax.axvline(v, color=c, lw=2, label=f"{name} = {v:.2f} mm")
    ax.set_xlabel(f"{label} [mm]")
    ax.set_ylabel("Liczba ziaren" if weighting == "number" else "Udział objętościowy [%]")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# Mid-tone ink / series colours that stay legible on both light and dark app themes.
_INK, _SERIES, _MUTED, _EST = "#7f8591", "#3987e5", "#a9aeb8", "#e8743b"
_WEIGHT_LABEL = {"number": "liczby ziaren", "area": "powierzchni", "volume": "objętości"}


def plot_psd(result, label: str, log_x: bool = True):
    """Cumulative passing curve (particle size distribution) with D10/D50/D90.

    Drone mode (area weighting) adds the unmeasured part of the surface:
      blue solid  - measured rocks only,
      blue dashed - unmeasured area counted as fines (upper bound), band = uncertainty,
      orange      - Rosin-Rammler estimate of the whole surface, extrapolated below the
                    measurement limit (an estimate, not a measurement).
    """
    b, est = result.bounds, result.estimate
    if b is not None:
        xs, low = b["sizes"], b["lower"]
        d_low = b["D_lower"]
    else:
        xs, low = cumulative_passing(result.sizes, result.weighting)
        d_low = passing_curve_percentiles(xs, low)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.plot(xs, low, color=_SERIES, lw=2,
            label="Zmierzone kamienie" + (" (bez niezmierzonej drobnicy)" if b is not None else ""))
    if b is not None:
        ax.plot(xs, b["upper"], color=_SERIES, lw=1.6, ls=(0, (4, 3)),
                label="Górna granica: niezmierzony obszar = drobnica")
        ax.fill_between(xs, low, b["upper"], color=_SERIES, alpha=0.10, lw=0)
    d_marks = d_low
    x_left = xs.min()
    if est is not None:
        fit = est["fit"]
        x_left = est["floor_mm"]
        grid = np.geomspace(min(x_left, xs.min()), xs.max(), 200)
        ax.plot(grid, rr_passing(fit, grid), color=_EST, lw=2,
                label="Szacunek całej hałdy (ekstrapolacja Rosina–Rammlera)")
        ax.axvspan(grid.min(), result.min_size_mm, color=_MUTED, alpha=0.08, lw=0)
        d_marks = est["D"]
    if result.min_size_mm:
        ax.axvline(result.min_size_mm, color=_MUTED, lw=1, ls=":")
        ax.text(result.min_size_mm, 101.5, " próg pomiaru", color=_INK, fontsize=8, va="bottom")

    mark_color = _EST if est is not None else _SERIES
    for (name, val), p in zip(d_marks.items(), (10, 50, 90)):
        ax.axhline(p, color=_MUTED, lw=0.6, alpha=0.6)
        if np.isfinite(val):
            ax.plot([val], [p], "o", ms=6, color=mark_color, mec="none")
            ax.annotate(f"{name} = {val:.0f} mm", (val, p), xytext=(6, -12),
                        textcoords="offset points", color=_INK, fontsize=9)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))   # label 2x and 5x ticks
    ax.set_xlim(left=min(x_left, xs.min()) * 0.95)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.set_xlabel(f"{label} [mm]", color=_INK)
    ax.set_ylabel(f"Skumulowany udział {_WEIGHT_LABEL[result.weighting]} poniżej rozmiaru [%]",
                  color=_INK, fontsize=9)
    ax.tick_params(colors=_INK, labelsize=9)
    ax.grid(True, which="major", color=_MUTED, alpha=0.25, lw=0.6)
    if log_x:
        ax.grid(True, which="minor", axis="x", color=_MUTED, alpha=0.12, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ax.spines.values():
        sp.set_color(_MUTED)
    if b is not None:
        leg = ax.legend(loc="lower right", frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_color(_INK)
    fig.tight_layout()
    return fig
