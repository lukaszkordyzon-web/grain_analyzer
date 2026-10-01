"""Oblique drone photos: no marker, a person of known height is the scale reference.

Scale model (pinhole camera, zero roll):
  1. The camera depression angle (gimbal pitch) and focal length (px) are known.
  2. The person's feet and head are clicked. A person of height ``h`` standing on the
     ground spans a number of image pixels that depends on the distance Z of the person
     and on the viewing angle -> solve for Z (``solve_person_distance``).
  3. mm/px at the person = Z / f.
  4. Everywhere else the scale is proportional to the metric depth map:
     mm/px(u, v) = mm/px(person) * Z(u, v) / Z(person)   (only depth *ratios* are used,
     so a global bias of the depth model cancels out).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from .measure import grain_geometry


# --------------------------------------------------------------------------- calibration
def _down_vector(pitch_deg: float) -> np.ndarray:
    """World 'down' in camera coords (x right, y down, z forward); pitch 90 = nadir."""
    t = math.radians(pitch_deg)
    return np.array([0.0, math.cos(t), math.sin(t)])


def person_span_px(z: float, feet: tuple[float, float], f: float, cx: float, cy: float,
                   pitch_deg: float, height_m: float) -> float:
    """Pixel distance feet->head predicted for a person standing ``z`` m (along the optical
    axis) from the camera at pixel ``feet``."""
    d = np.array([(feet[0] - cx) / f, (feet[1] - cy) / f, 1.0])
    p_feet = z * d
    p_head = p_feet - height_m * _down_vector(pitch_deg)
    if p_head[2] <= 1e-6:
        return float("inf")
    uh = f * p_head[0] / p_head[2] + cx
    vh = f * p_head[1] / p_head[2] + cy
    return math.hypot(uh - feet[0], vh - feet[1])


def solve_person_distance(head: tuple[float, float], feet: tuple[float, float], f: float,
                          cx: float, cy: float, pitch_deg: float, height_m: float) -> float:
    """Distance Z [m] at which a person of ``height_m`` appears as the clicked head/feet span."""
    observed = math.hypot(head[0] - feet[0], head[1] - feet[1])
    if observed < 3:
        raise ValueError("Głowa i stopy są zbyt blisko siebie — kliknij dokładniej.")
    # smallest distance for which the head is still in front of the camera
    d_z, g_z = 1.0, _down_vector(pitch_deg)[2]
    lo, hi = max(0.3, 1.001 * height_m * g_z / d_z), 5000.0
    if person_span_px(lo, feet, f, cx, cy, pitch_deg, height_m) < observed:
        raise ValueError("Człowiek wygląda na większego niż pozwala geometria — sprawdź kąt "
                         "gimbala, ogniskową i punkty głowy/stóp.")
    for _ in range(80):                              # span(z) decreases with z -> bisection
        mid = math.sqrt(lo * hi)
        if person_span_px(mid, feet, f, cx, cy, pitch_deg, height_m) > observed:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


@dataclass
class PersonCalibration:
    z_person_m: float          # distance of the person from the camera (along the axis)
    mm_per_px_person: float
    span_px: float             # clicked head->feet length
    z_model_m: float | None = None   # what the depth model says at the person (sanity check)

    @property
    def warnings(self) -> list[str]:
        w = []
        if self.span_px < 15:
            w.append(f"Człowiek ma tylko {self.span_px:.0f} px — kalibracja jest mało dokładna "
                     "(użyj zdjęcia w wyższej rozdzielczości albo takiego, gdzie jest bliżej).")
        if self.z_model_m and not 0.5 < self.z_model_m / self.z_person_m < 2.0:
            w.append(f"Model głębi widzi człowieka w {self.z_model_m:.0f} m, a geometria wychodzi "
                     f"{self.z_person_m:.0f} m — sprawdź kąt gimbala i ogniskową.")
        return w


def calibrate_person(head, feet, depth_m: np.ndarray | None, f_px: float, pitch_deg: float,
                     height_m: float = 1.75) -> PersonCalibration:
    h, w = (depth_m.shape if depth_m is not None else (0, 0))
    cx, cy = (w / 2, h / 2) if depth_m is not None else (0.0, 0.0)
    return _calibrate(head, feet, depth_m, f_px, pitch_deg, height_m, cx, cy)


def _calibrate(head, feet, depth_m, f_px, pitch_deg, height_m, cx, cy) -> PersonCalibration:
    z = solve_person_distance(head, feet, f_px, cx, cy, pitch_deg, height_m)
    z_model = person_depth(depth_m, head, feet) if depth_m is not None else None
    return PersonCalibration(z, 1000 * z / f_px,
                             math.hypot(head[0] - feet[0], head[1] - feet[1]), z_model)


def person_depth(depth_m: np.ndarray, head, feet) -> float:
    """Median model depth in a small window around the person."""
    r = max(3, int(0.3 * math.hypot(head[0] - feet[0], head[1] - feet[1])))
    mu, mv = (head[0] + feet[0]) / 2, (head[1] + feet[1]) / 2
    h, w = depth_m.shape
    win = depth_m[max(0, int(mv) - r):min(h, int(mv) + r + 1),
                  max(0, int(mu) - r):min(w, int(mu) + r + 1)]
    return float(np.median(win))


def scale_map(depth_m: np.ndarray, cal: PersonCalibration, head, feet) -> np.ndarray:
    """mm per pixel for every pixel, anchored at the person."""
    z_ref = person_depth(depth_m, head, feet)
    return (cal.mm_per_px_person / max(z_ref, 1e-6) * depth_m).astype(np.float32)


# --------------------------------------------------------------------------- segmentation
def tile_boxes(x0: int, y0: int, x1: int, y1: int, tile: int, overlap: float):
    """Overlapping (ya, yb, xa, xb) tiles covering the box."""
    step = max(1, int(tile * (1 - overlap)))

    def starts(a, b):
        if b - a <= tile:
            return [a]
        s = list(range(a, b - tile, step))
        return s + [b - tile]

    return [(ya, min(ya + tile, y1), xa, min(xa + tile, x1))
            for ya in starts(y0, y1) for xa in starts(x0, x1)]


def build_labels(image: np.ndarray, segmenter, roi, *, tile: int = 800, overlap: float = 0.2,
                 min_area_px: int = 100, max_area_px: int = 10**9, max_overlap: float = 0.4,
                 progress=None):
    """Run the segmenter tile by tile and paint accepted masks into one label image.

    Memory stays flat: only one tile's masks exist at a time (a full-frame mask per rock
    would not fit). Masks cut by an *inner* tile edge are dropped - the neighbouring tile
    sees the whole rock thanks to the overlap.
    Returns (labels int32 HxW, {label: (ya, yb, xa, xb)}).
    """
    h, w = image.shape[:2]
    x0, y0, x1, y1 = roi
    labels = np.zeros((h, w), np.int32)
    boxes: dict[int, tuple[int, int, int, int]] = {}
    tiles = tile_boxes(x0, y0, x1, y1, tile, overlap)
    nxt = 1
    for n, (ya, yb, xa, xb) in enumerate(tiles):
        if progress:
            progress(n / len(tiles), f"Kafelek {n + 1}/{len(tiles)}")
        sub = labels[ya:yb, xa:xb]                    # view
        for mask, _score in sorted(segmenter.segment(image[ya:yb, xa:xb]), key=lambda c: -c[1]):
            area = int(mask.sum())
            if area < min_area_px or area > max_area_px:
                continue
            ys, xs = np.nonzero(mask)
            top, bot, left, right = ys.min(), ys.max(), xs.min(), xs.max()
            th, tw = mask.shape
            if ((top == 0 and ya > y0) or (bot == th - 1 and yb < y1) or
                    (left == 0 and xa > x0) or (right == tw - 1 and xb < x1)):
                continue
            if (sub[mask] > 0).mean() > max_overlap:
                continue
            free = mask & (sub == 0)
            if free.sum() < min_area_px:
                continue
            sub[free] = nxt
            boxes[nxt] = (ya + top, ya + bot + 1, xa + left, xa + right + 1)
            nxt += 1
    if progress:
        progress(1.0, "Gotowe")
    return labels, boxes


def measure_labels(labels: np.ndarray, boxes: dict, scale_mm_px: np.ndarray,
                   depth_m: np.ndarray | None = None):
    """Per-rock geometry with the local scale. Returns (DataFrame, contours)."""
    rows, contours = [], []
    for lab, (ya, yb, xa, xb) in boxes.items():
        mask = labels[ya:yb, xa:xb] == lab
        if mask.sum() < 5:
            continue
        s = float(np.median(scale_mm_px[ya:yb, xa:xb][mask]))
        row = grain_geometry(mask, s)
        if row is None:
            continue
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        contours.append(max(cnts, key=cv2.contourArea) + np.array([xa, ya]))
        row["cx_px"] += xa
        row["cy_px"] += ya
        row["mm_per_px"] = s
        if depth_m is not None:
            row["depth_m"] = float(np.median(depth_m[ya:yb, xa:xb][mask]))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "id", np.arange(1, len(df) + 1))
    return df, contours
