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
from .stats import passing_bounds, size_distribution


@dataclass
class Params:
    max_side: int = 1200                 # working resolution (long side, px)
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
    contours: list | None = None          # full-frame contours (drone mode; masks not kept)
    annotations: dict = field(default_factory=dict)   # person/roi markers for the overlay
    notes: list = field(default_factory=list)         # warnings shown next to the results
    weighting: str = "number"
    roi_area_mm2: float | None = None     # drone mode: ground area of the analysed region
    min_size_mm: float | None = None      # smallest measurable rock (resolution limit)
    bounds: dict | None = None            # area passing curves incl. unmeasured fines

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
    return AnalysisResult(image, scale, masks, grains, depth, col, perc, n_rej, weighting=p.weighting)


@dataclass
class DroneParams:
    max_side: int = 2000
    person_height_m: float = 1.75
    pitch_deg: float = 45.0               # camera depression, 90 = straight down
    focal_35mm: float = 24.0
    tile: int = 800
    min_diameter_px: float = 12.0         # below this a rock is not resolved
    max_area_frac: float = 0.02           # of the selected area
    max_overlap: float = 0.4
    size_metric: str = "ecd"
    weighting: str = "area"               # surface fraction by size (top-view standard)
    max_tiles: int = 8                    # guard: each tile = one SAM pass (slow, RAM-hungry)


def plan_drone(image_hw: tuple[int, int], roi_xyxy, p: DroneParams):
    """(number of SAM tiles, working-resolution ROI) for the given settings, without
    running anything - used for the cost estimate and the guard."""
    from . import drone

    h0, w0 = image_hw
    k = min(1.0, p.max_side / max(h0, w0))
    w, h = round(w0 * k), round(h0 * k)
    roi = (0, 0, w, h) if roi_xyxy is None else tuple(int(round(v * k)) for v in roi_xyxy)
    roi = (max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3]))
    return len(drone.tile_boxes(*roi, p.tile, 0.2)), roi


def analyze_drone(image_rgb: np.ndarray, p: DroneParams, segmenter: Segmenter,
                  head_xy, feet_xy, roi_xyxy=None, progress=None,
                  depth_estimator=None) -> AnalysisResult:
    """Oblique drone photo, person of known height as the scale reference.
    Points/ROI are in *original* image pixels.

    Default scale model: horizontal ground plane through the person's feet (geometry only,
    from camera pitch + focal length). ``depth_estimator`` (a *metric* depth model) switches to
    experimental depth-ratio scaling; such models are trained on ground-level scenes and
    often misjudge aerial views."""
    from . import drone
    from .camera import focal_px

    if p.size_metric not in SIZE_METRICS:
        raise ValueError(f"size_metric must be one of {list(SIZE_METRICS)}")
    image = resize_max_side(image_rgb, p.max_side)
    k = image.shape[1] / image_rgb.shape[1]
    h, w = image.shape[:2]
    head = (head_xy[0] * k, head_xy[1] * k)
    feet = (feet_xy[0] * k, feet_xy[1] * k)
    roi = (0, 0, w, h) if roi_xyxy is None else tuple(int(round(v * k)) for v in roi_xyxy)
    roi = (max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3]))
    if roi[2] - roi[0] < 50 or roi[3] - roi[1] < 50:
        raise ValueError("Zaznaczony obszar hałdy jest zbyt mały.")

    n_tiles, _ = plan_drone(image_rgb.shape[:2], roi_xyxy, p)
    if n_tiles > p.max_tiles:
        raise ValueError(
            f"Analiza wymagałaby {n_tiles} kafelków (limit {p.max_tiles}) i przekroczyłaby "
            "pamięć serwera. Zmniejsz rozdzielczość roboczą, zwiększ rozmiar kafelka albo "
            "zaznacz mniejszy obszar hałdy.")

    f = focal_px(p.focal_35mm, w, h)
    depth = None
    if depth_estimator is not None:
        if progress:
            progress(0.0, "Głębia metryczna…")
        depth = depth_estimator.estimate(image)
    cal = drone._calibrate(head, feet, depth, f, p.pitch_deg, p.person_height_m, w / 2, h / 2)
    smap = (drone.scale_map(depth, cal, head, feet) if depth is not None
            else drone.plane_scale_map((h, w), f, p.pitch_deg, feet, cal.z_person_m))

    roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])
    labels, boxes = drone.build_labels(
        image, segmenter, roi, tile=p.tile, min_area_px=int(np.pi / 4 * p.min_diameter_px ** 2),
        max_area_px=int(p.max_area_frac * roi_area), max_overlap=p.max_overlap, progress=progress)
    grains, contours = drone.measure_labels(labels, boxes, smap, depth)

    col = {"ecd": "ecd_mm", "feret_min": "feret_min_mm", "feret_max": "feret_max_mm"}[p.size_metric]
    perc = size_distribution(grains[col].to_numpy(), p.weighting) if len(grains) else {}
    coverage = float((labels[roi[1]:roi[3], roi[0]:roi[2]] > 0).mean())
    notes = list(cal.warnings)
    if depth is None:
        notes.append("Skala zakłada, że hałda leży w płaszczyźnie terenu, na którym stoi człowiek. "
                     "Wyższe partie hałdy są bliżej kamery, więc ich rozmiary są lekko zawyżone.")
    notes.append(f"Pokrycie obszaru zmierzonymi kamieniami: {coverage:.0%}. Drobniejsza frakcja "
                 f"(średnica < {p.min_diameter_px:.0f} px ≈ "
                 f"{p.min_diameter_px * float(np.median(smap[roi[1]:roi[3], roi[0]:roi[2]])) / 10:.0f} cm "
                 "w środku obszaru) nie jest mierzona.")
    scale = scale_mod.ScaleResult(float(grains["mm_per_px"].median()) if len(grains) else 0.0,
                                  "człowiek" + (" + głębia" if depth is not None else " + płaszczyzna"))
    roi_smap = smap[roi[1]:roi[3], roi[0]:roi[2]].astype(np.float64)
    roi_area_mm2 = float(np.sum(roi_smap ** 2))
    min_size_mm = float(p.min_diameter_px * np.median(roi_smap))
    # bracketing the unmeasured fines is only meaningful on an area basis
    bounds = (passing_bounds(grains[col].to_numpy(), grains["area_mm2"].to_numpy(),
                             roi_area_mm2, min_size_mm)
              if len(grains) and p.weighting == "area" else None)
    return AnalysisResult(image, scale, [], grains, depth, col, perc, 0, contours,
                          {"head": head, "feet": feet, "roi": roi}, notes, p.weighting,
                          roi_area_mm2, min_size_mm, bounds)
