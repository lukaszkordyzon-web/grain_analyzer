"""Overlay with contours and the size histogram."""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .stats import histogram  # noqa: E402


def draw_overlay(result, show_ids: bool = False, thickness: int = 1) -> np.ndarray:
    out = result.image.copy()
    for m in result.masks:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 0), thickness)
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
