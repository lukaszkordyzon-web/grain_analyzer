"""Pixel -> millimetre scale from an ArUco marker of known size in the frame."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "original": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


@dataclass
class ScaleResult:
    mm_per_px: float
    method: str                      # "aruco" | "manual"
    corners: np.ndarray | None = None  # (4, 2) marker corners in image px
    marker_id: int | None = None

    def marker_mask(self, shape: tuple[int, int], pad_px: int = 0) -> np.ndarray:
        """Boolean mask of the marker area (grains must not be found on it)."""
        mask = np.zeros(shape[:2], np.uint8)
        if self.corners is not None:
            cv2.fillConvexPoly(mask, self.corners.astype(np.int32), 1)
            if pad_px:
                mask = cv2.dilate(mask, np.ones((2 * pad_px + 1,) * 2, np.uint8))
        return mask.astype(bool)


def detect_marker(image_rgb: np.ndarray, marker_size_mm: float,
                  dictionary: str = "4x4_50", marker_id: int | None = None) -> ScaleResult | None:
    """Find an ArUco marker and derive mm/px from its mean edge length.

    ``marker_size_mm`` is the side of the black square (outer border), not the paper.
    Returns None when no (matching) marker is found.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary]),
        cv2.aruco.DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    best = None
    for c, i in zip(corners, ids.ravel()):
        if marker_id is not None and int(i) != marker_id:
            continue
        pts = c.reshape(4, 2)
        side_px = float(np.mean([np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]))
        if best is None or side_px > best[0]:
            best = (side_px, pts, int(i))
    if best is None:
        return None
    side_px, pts, mid = best
    return ScaleResult(marker_size_mm / side_px, "aruco", pts, mid)


def manual_scale(mm_per_px: float) -> ScaleResult:
    if mm_per_px <= 0:
        raise ValueError("mm_per_px must be positive")
    return ScaleResult(mm_per_px, "manual")
