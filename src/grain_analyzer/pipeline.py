"""End-to-end analysis: scale -> SAM -> (depth) -> measurements -> statistics."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from . import scale as scale_mod
from .depth import DepthEstimator
from .measure import SIZE_METRICS, depth_outliers, measure_grains
from .segmentation import Segmenter, select_grain_masks
from .stats import (completeness_limit, estimate_fines, fraction_table, passing_bounds,
                    size_distribution)


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
    estimate: dict | None = None          # Rosin-Rammler estimate of the whole surface
    fractions: list | None = None         # surface share by size class
    reliable_mm: float | None = None      # above this size detection is (nearly) complete

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


REFERENCES = {
    "person": "Człowiek w kadrze (stopy i głowa)",
    "wall": "Ściana o znanej wysokości (dolna i górna krawędź)",
    "marker": "Znacznik ArUco na ziemi (automatycznie)",
}


@dataclass
class DroneParams:
    max_side: int = 2000
    reference: str = "person"             # key of REFERENCES
    person_height_m: float = 1.75
    wall_height_m: float = 10.0
    wall_slope_deg: float = 90.0          # 90 = vertical; smaller = leaning back, away from the camera
    marker_size_mm: float = 200.0
    marker_dict: str = "4x4_50"
    pitch_deg: float = 45.0               # camera depression, 90 = straight down
    focal_35mm: float = 24.0
    tile: int = 800
    min_diameter_px: float = 12.0         # below this a rock is not resolved
    max_area_frac: float = 0.02           # of the selected area
    max_overlap: float = 0.4
    size_metric: str = "ecd"
    weighting: str = "area"               # surface fraction by size (top-view standard)
    max_stone_mm: float = 3000.0          # larger "stones" are masks of shadow/wall; 0 = no cap
    shadow_ratio: float = 0.6             # drop uniform patches darker than this x surroundings; 0 = off
    # guard: each tile = one SAM pass (slow, RAM-hungry); raise it on a strong machine
    max_tiles: int = field(default_factory=lambda: int(os.environ.get("GRAIN_MAX_TILES", "8")))


@dataclass
class DroneSegmentation:
    """Result of the slow step (SAM on tiles). Independent of the scale reference, so the
    reference, heights and statistics can be changed without running SAM again."""
    image: np.ndarray                    # working-resolution RGB
    labels: np.ndarray                   # int32 label image
    boxes: dict
    roi: tuple                           # working-resolution (x0, y0, x1, y1)
    k: float                             # working px / original px
    n_tiles: int
    seconds: float
    key: tuple = ()


def segmentation_key(p: DroneParams, roi_xyxy, model: str = "") -> tuple:
    """Everything that changes the segmentation (scale reference and statistics do not)."""
    roi = None if roi_xyxy is None else tuple(round(float(v)) for v in roi_xyxy)
    return (model, p.max_side, p.tile, round(p.min_diameter_px, 3), round(p.max_area_frac, 5),
            p.max_overlap, roi)


def _working_roi(image_hw, roi_xyxy, max_side):
    h0, w0 = image_hw
    k = min(1.0, max_side / max(h0, w0))
    w, h = round(w0 * k), round(h0 * k)
    roi = (0, 0, w, h) if roi_xyxy is None else tuple(int(round(v * k)) for v in roi_xyxy)
    return k, (w, h), (max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3]))


def plan_drone(image_hw: tuple[int, int], roi_xyxy, p: DroneParams):
    """(number of SAM tiles, working-resolution ROI) for the given settings, without
    running anything - used for the cost estimate and the guard."""
    from . import drone

    _, _, roi = _working_roi(image_hw, roi_xyxy, p.max_side)
    return len(drone.tile_boxes(*roi, p.tile, 0.2)), roi


def segment_drone(image_rgb: np.ndarray, p: DroneParams, segmenter: Segmenter, roi_xyxy=None,
                  progress=None, model: str = "") -> DroneSegmentation:
    """Slow step: SAM on overlapping tiles of the selected area (original-pixel ROI)."""
    import time

    from . import drone

    image = resize_max_side(image_rgb, p.max_side)
    k, _, roi = _working_roi(image_rgb.shape[:2], roi_xyxy, p.max_side)
    if roi[2] - roi[0] < 50 or roi[3] - roi[1] < 50:
        raise ValueError("Zaznaczony obszar hałdy jest zbyt mały.")
    n_tiles, _ = plan_drone(image_rgb.shape[:2], roi_xyxy, p)
    if n_tiles > p.max_tiles:
        raise ValueError(
            f"Analiza wymagałaby {n_tiles} kafelków (limit {p.max_tiles}) i przekroczyłaby "
            "pamięć serwera. Zwiększ najmniejszy mierzony kamień, zwiększ rozmiar kafelka "
            "albo zaznacz mniejszy obszar hałdy.")
    t0 = time.perf_counter()
    roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])
    labels, boxes = drone.build_labels(
        image, segmenter, roi, tile=p.tile, min_area_px=int(np.pi / 4 * p.min_diameter_px ** 2),
        max_area_px=int(p.max_area_frac * roi_area), max_overlap=p.max_overlap, progress=progress)
    return DroneSegmentation(image, labels, boxes, roi, k, n_tiles, time.perf_counter() - t0,
                             segmentation_key(p, roi_xyxy, model))


def _reference(image, p: DroneParams, head, feet, f, cx, cy, depth_estimator=None, progress=None):
    """Camera height above the reference plane from the chosen reference.
    Returns (d_cam, cal, depth, notes, exclude_mask)."""
    from . import drone

    notes: list[str] = []
    if p.reference not in REFERENCES:
        raise ValueError(f"reference must be one of {list(REFERENCES)}")
    if p.reference == "marker":
        found = scale_mod.detect_marker(image, p.marker_size_mm, p.marker_dict)
        if found is None:
            raise ValueError("Nie znaleziono znacznika ArUco w kadrze — wybierz inne źródło skali.")
        d_cam = drone.camera_height_from_marker(found.corners, p.marker_size_mm / 1000, f, cx, cy,
                                                p.pitch_deg)
        side_px = float(np.mean([np.linalg.norm(found.corners[i] - found.corners[(i + 1) % 4])
                                 for i in range(4)]))
        if side_px < 15:
            notes.append(f"Znacznik ma tylko {side_px:.0f} px — skala jest mało dokładna.")
        return d_cam, None, None, notes, found.marker_mask(image.shape, pad_px=5)
    if head is None or feet is None:
        raise ValueError("Zaznacz dwa punkty odniesienia (dół i górę).")
    ref_h = p.person_height_m if p.reference == "person" else p.wall_height_m
    lean = (0.0 if p.reference == "person" or p.wall_slope_deg >= 90
            else ref_h / math.tan(math.radians(p.wall_slope_deg)))
    depth = None
    if depth_estimator is not None:
        if progress:
            progress(0.0, "Głębia metryczna…")
        depth = depth_estimator.estimate(image)
    cal = drone._calibrate(head, feet, depth, f, p.pitch_deg, ref_h, cx, cy, lean)
    notes += cal.warnings
    if p.reference == "wall":
        notes.append("Skala ze ściany zakłada, że klikasz górną i dolną krawędź w jednym "
                     "pionie, a ściana ma podany kąt nachylenia.")
    return drone.camera_height(cal.z_person_m, feet, f, cy, p.pitch_deg), cal, depth, notes, None


def measure_drone(seg: DroneSegmentation, p: DroneParams, head_xy=None, feet_xy=None,
                  depth_estimator=None, progress=None) -> AnalysisResult:
    """Fast step: scale reference + per-rock measurements + statistics on a finished
    segmentation. ``head_xy``/``feet_xy`` are in *original* image pixels."""
    from . import drone
    from .camera import focal_px

    if p.size_metric not in SIZE_METRICS:
        raise ValueError(f"size_metric must be one of {list(SIZE_METRICS)}")
    image, labels, roi, k = seg.image, seg.labels, seg.roi, seg.k
    h, w = image.shape[:2]
    head = (head_xy[0] * k, head_xy[1] * k) if head_xy is not None else None
    feet = (feet_xy[0] * k, feet_xy[1] * k) if feet_xy is not None else None
    f = focal_px(p.focal_35mm, w, h)
    d_cam, cal, depth, notes, exclude = _reference(image, p, head, feet, f, w / 2, h / 2,
                                                   depth_estimator, progress)
    boxes = seg.boxes
    if exclude is not None:                        # drop the reference marker itself
        boxes = {lab: b for lab, b in boxes.items()
                 if (exclude[b[0]:b[1], b[2]:b[3]][labels[b[0]:b[1], b[2]:b[3]] == lab]).mean() <= 0.2}
    smap = (drone.scale_map(depth, cal, head, feet) if depth is not None
            else drone.scale_map_from_camera_height((h, w), f, p.pitch_deg, d_cam))
    shadows = (drone.shadow_labels(image, labels, boxes, p.shadow_ratio)
               if p.shadow_ratio and p.shadow_ratio > 0 else set())
    boxes = {lab: b for lab, b in boxes.items() if lab not in shadows}
    grains, contours = drone.measure_labels(labels, boxes, smap, depth,
                                            p.max_stone_mm if p.max_stone_mm else None)
    n_oversize = len(boxes) - len(grains)

    col = {"ecd": "ecd_mm", "feret_min": "feret_min_mm", "feret_max": "feret_max_mm"}[p.size_metric]
    perc = size_distribution(grains[col].to_numpy(), p.weighting) if len(grains) else {}
    kept = (np.isin(labels[roi[1]:roi[3], roi[0]:roi[2]], grains["mask_label"].to_numpy())
            if len(grains) else np.zeros(1, bool))
    coverage = float(kept.mean())
    if shadows or n_oversize > 0:
        notes.append(f"Odrzucono: {len(shadows)} masek wyglądających na cienie i {max(n_oversize, 0)} "
                     f"zbyt dużych (> {p.max_stone_mm / 1000:.1f} m).")
    roi_smap = smap[roi[1]:roi[3], roi[0]:roi[2]].astype(np.float64)
    roi_area_mm2 = float(np.sum(roi_smap ** 2))
    min_size_mm = float(p.min_diameter_px * np.median(roi_smap))
    if depth is None:
        notes.append("Skala zakłada, że hałda leży w płaszczyźnie terenu punktu odniesienia. "
                     "Wyższe partie hałdy są bliżej kamery, więc ich rozmiary są lekko zawyżone.")
    notes.append(f"Pokrycie obszaru zmierzonymi kamieniami: {coverage:.0%}. Kamienie mniejsze niż "
                 f"{min_size_mm / 10:.0f} cm ({p.min_diameter_px:.0f} px) nie są mierzone. "
                 f"Segmentacja: {seg.n_tiles} kafelków w {seg.seconds:.0f} s.")
    scale = scale_mod.ScaleResult(
        float(grains["mm_per_px"].median()) if len(grains) else 0.0,
        {"person": "człowiek", "wall": "ściana", "marker": "znacznik"}[p.reference]
        + (" + głębia" if depth is not None else " + płaszczyzna"))
    ann = {"roi": roi}
    if head is not None:
        ann.update(head=head, feet=feet)
    # bracketing the unmeasured fines is only meaningful on an area basis
    bounds = (passing_bounds(grains[col].to_numpy(), grains["area_mm2"].to_numpy(),
                             roi_area_mm2, min_size_mm)
              if len(grains) and p.weighting == "area" else None)
    reliable_mm = None
    if len(grains):
        reliable_mm, how = completeness_limit(grains[col].to_numpy(), min_size_mm)
    estimate = estimate_fines(bounds, reliable_mm=reliable_mm) if bounds else None
    fractions = (fraction_table(grains[col].to_numpy(), grains["area_mm2"].to_numpy(),
                                roi_area_mm2, min_size_mm) if bounds else None)
    if len(grains):
        notes.append(
            f"Kamienie mniejsze niż ok. {reliable_mm / 10:.0f} cm są wykrywane niepełnie"
            + (" (liczba kamieni spada poniżej tej wielkości, zamiast rosnąć)" if how == "turnover"
               else " (oszacowanie przybliżone)")
            + f"; frakcja {min_size_mm / 10:.0f}–{reliable_mm / 10:.0f} cm jest niedoszacowana.")
    return AnalysisResult(image, scale, [], grains, depth, col, perc, 0, contours, ann, notes,
                          p.weighting, roi_area_mm2, min_size_mm, bounds, estimate, fractions,
                          reliable_mm)


def analyze_drone(image_rgb: np.ndarray, p: DroneParams, segmenter: Segmenter,
                  head_xy=None, feet_xy=None, roi_xyxy=None, progress=None,
                  depth_estimator=None) -> AnalysisResult:
    """Oblique drone photo; scale from a person, a wall of known height, or an ArUco marker
    (``p.reference``). Points/ROI are in *original* image pixels. Convenience wrapper:
    ``segment_drone`` (slow) + ``measure_drone`` (fast)."""
    seg = segment_drone(image_rgb, p, segmenter, roi_xyxy, progress)
    return measure_drone(seg, p, head_xy, feet_xy, depth_estimator, progress)


def preview_drone(image_hw: tuple[int, int], p: DroneParams, head_xy, feet_xy, roi_xyxy,
                  image_rgb: np.ndarray | None = None) -> dict | None:
    """Cheap geometry-only look at the scale BEFORE any model runs (person/wall: pure math;
    marker: needs ``image_rgb``). Returns the median mm per ORIGINAL pixel in the area, or None
    while the reference is not complete / not usable."""
    from . import drone
    from .camera import focal_px

    if roi_xyxy is None:
        return None
    h0, w0 = image_hw
    f = focal_px(p.focal_35mm, w0, h0)
    try:
        img = image_rgb if image_rgb is not None else np.zeros((1, 1, 3), np.uint8)
        d_cam, *_ = _reference(img, p, head_xy, feet_xy, f, w0 / 2, h0 / 2)
    except (ValueError, TypeError):
        return None
    rows = drone.row_scale(h0, f, p.pitch_deg, d_cam)
    y0, y1 = max(0, int(roi_xyxy[1])), min(h0, int(roi_xyxy[3]))
    if y1 <= y0:
        return None
    return {"mm_per_px_orig": float(np.median(rows[y0:y1])), "camera_height_m": float(d_cam)}


def working_side_for(target_mm: float, mm_per_px_orig: float, long_side_orig: int,
                     min_diameter_px: float) -> int:
    """Working resolution (long side, px) at which a stone of ``target_mm`` is
    ``min_diameter_px`` wide. May exceed the original size - then the photo cannot do it."""
    return int(np.ceil(long_side_orig * min_diameter_px * mm_per_px_orig / target_mm))
