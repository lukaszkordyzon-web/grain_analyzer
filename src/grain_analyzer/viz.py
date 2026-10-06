"""Overlay with contours and the size histogram."""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .stats import (cumulative_passing, histogram, passing_curve_percentiles,  # noqa: E402
                    rr_anchored_passing)


def size_colors(sizes: np.ndarray, lo: float | None = None, hi: float | None = None):
    """RGB colour per size on a green -> yellow -> red scale (log size). Small stones are green,
    large red. The range ends at the 97th percentile so a few giants do not wash out the rest.
    Returns (colours (N, 3) uint8, lo, hi)."""
    s = np.asarray(sizes, float)
    lo = float(np.min(s)) if lo is None else lo
    hi = float(np.percentile(s, 97)) if hi is None else hi
    hi = max(hi, lo * 1.01)
    t = np.clip((np.log(np.maximum(s, 1e-9)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
    hsv = np.zeros((len(s), 1, 3), np.uint8)
    hsv[:, 0, 0] = (60 * (1 - t)).astype(np.uint8)           # OpenCV hue: 60 = green ... 0 = red
    hsv[:, 0, 1:] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[:, 0, :], lo, hi


def _draw_legend(out: np.ndarray, lo: float, hi: float) -> None:
    """Colour bar with the size range (cm) in the bottom-left corner."""
    h, w = out.shape[:2]
    bar_w, bar_h = int(0.26 * w), max(10, int(0.02 * w))
    x0, y0 = int(0.02 * w), h - int(0.02 * w) - bar_h
    scale = max(0.5, w / 1800)
    pad = int(8 * scale) + 4
    cv2.rectangle(out, (x0 - pad, y0 - int(30 * scale) - pad), (x0 + bar_w + pad, y0 + bar_h + int(26 * scale) + pad),
                  (0, 0, 0), -1)
    grad = np.linspace(0, 1, bar_w)
    cols, _, _ = size_colors(np.exp(np.log(lo) + grad * (np.log(hi) - np.log(lo))), lo, hi)
    out[y0:y0 + bar_h, x0:x0 + bar_w] = cols[None, :, :]
    font, th = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(scale))
    cv2.putText(out, "rozmiar kamienia (ECD)", (x0, y0 - int(10 * scale)), font, 0.5 * scale,
                (255, 255, 255), th, cv2.LINE_AA)
    for frac in (0.0, 0.5, 1.0):
        v = np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo))) / 10          # mm -> cm
        label = f"{v:.0f} cm" if v >= 10 else f"{v:.1f} cm"
        (tw, _), _ = cv2.getTextSize(label, font, 0.45 * scale, th)
        tx = int(x0 + frac * bar_w - frac * tw)
        cv2.putText(out, label, (tx, y0 + bar_h + int(20 * scale)), font, 0.45 * scale,
                    (255, 255, 255), th, cv2.LINE_AA)


def draw_overlay(result, show_ids: bool = False, thickness: int | None = None,
                 color_by_size: bool = True, legend: bool = True) -> np.ndarray:
    """Stone outlines on the photo. With ``color_by_size`` every outline gets a colour from green
    (small) to red (large) on a log scale, with a legend; each outline has a dark casing so it
    stays visible on both light and dark ground. ``thickness`` defaults to ~1/700 of the image."""
    out = result.image.copy()
    h, w = out.shape[:2]
    thickness = thickness or max(2, round(max(h, w) / 700))
    if result.contours is not None:
        contours = list(result.contours)
    else:
        contours = []
        for m in result.masks:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.append(max(cnts, key=cv2.contourArea) if cnts else None)
    sizes = result.grains[result.size_column].to_numpy() if len(result.grains) else np.array([])
    paint = color_by_size and len(sizes) == len(contours) and len(sizes) > 0
    if paint:
        colors, lo, hi = size_colors(sizes)
    for i, c in enumerate(contours):
        if c is None:
            continue
        cv2.drawContours(out, [c], -1, (0, 0, 0), thickness + 2, cv2.LINE_AA)       # casing
        col = tuple(int(v) for v in colors[i]) if paint else (0, 255, 0)
        cv2.drawContours(out, [c], -1, col, thickness, cv2.LINE_AA)
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
    if paint and legend:
        _draw_legend(out, lo, hi)
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

    Drone mode (area weighting) draws ONE curve for the whole analysed surface:
      blue   - measured stones (>= measurement limit); the unmeasured area is counted as
               fines, so the curve starts at that share (e.g. 77 %) at the limit,
      orange - the same curve continued BELOW the limit: an estimate (Rosin-Rammler),
      thin   - lower bound: unmeasured area is voids/shadow (stones measured alone, 0-100 %).
    The true curve lies in the shaded band.
    """
    b, est = result.bounds, result.estimate
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    x_left = None
    if b is not None:
        xs, low, up = b["sizes"], b["lower"], b["upper"]
        ax.plot(xs, up, color=_SERIES, lw=2,
                label="Cała hałda: kamienie zmierzone")
        ax.plot(xs, low, color=_SERIES, lw=1, alpha=0.45,
                label="Dolna granica (niezmierzone = szczeliny/cień)")
        ax.fill_between(xs, low, up, color=_SERIES, alpha=0.10, lw=0)
        if est is not None:                       # continue the same curve below the limit
            fit = est["fit"]
            x_left = est["floor_mm"]
            grid = np.geomspace(x_left, xs[0], 80)
            ax.plot(grid, rr_anchored_passing(fit, grid, xs[0], b["unmeasured_fraction"]),
                    color=_EST, lw=2, label="Szacunek poniżej progu (Rosin–Rammler)")
            ax.axvspan(x_left, xs[0], color=_MUTED, alpha=0.08, lw=0)
        marks = est["D"] if est is not None else b["D_upper"]
        mark_x0 = xs[0]
    else:
        xs, low = cumulative_passing(result.sizes, result.weighting)
        ax.plot(xs, low, color=_SERIES, lw=2, label="Zmierzone kamienie")
        marks, mark_x0 = passing_curve_percentiles(xs, low), 0
    if result.min_size_mm:
        ax.axvline(result.min_size_mm, color=_MUTED, lw=1, ls=":")
        ax.text(result.min_size_mm, 101.5, " próg pomiaru", color=_INK, fontsize=8, va="bottom")

    for (name, val), p in zip(marks.items(), (10, 50, 90)):
        ax.axhline(p, color=_MUTED, lw=0.6, alpha=0.6)
        if np.isfinite(val):
            ax.plot([val], [p], "o", ms=6, color=_EST if val < mark_x0 else _SERIES, mec="none")
            ax.annotate(f"{name} = {val:.0f} mm" + (" (szac.)" if val < mark_x0 else ""), (val, p),
                        xytext=(6, -12), textcoords="offset points", color=_INK, fontsize=9)
    x_min = min(v for v in (x_left, xs.min()) if v is not None)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))   # label 2x and 5x ticks
    ax.set_xlim(left=x_min * 0.95)
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
