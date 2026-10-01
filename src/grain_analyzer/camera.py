"""Camera metadata (focal length, DJI gimbal pitch) read from a JPEG's EXIF/XMP."""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass

from PIL import Image

DEFAULT_FOCAL_35MM = 24.0   # DJI Matrice 3D/4D wide camera (35 mm equivalent)


@dataclass
class CameraMeta:
    focal_35mm: float | None = None      # 35 mm-equivalent focal length [mm]
    gimbal_pitch_deg: float | None = None  # DJI: 0 = horizon, -90 = straight down
    rel_altitude_m: float | None = None


def read_camera_meta(raw: bytes) -> CameraMeta:
    """Best effort: any field that is missing (e.g. metadata stripped) stays None."""
    meta = CameraMeta()
    try:
        exif = Image.open(io.BytesIO(raw)).getexif()
        f35 = exif.get_ifd(0x8769).get(0xA405)       # FocalLengthIn35mmFilm
        if f35:
            meta.focal_35mm = float(f35)
    except Exception:
        pass
    # DJI writes XMP as plain text in the first segments of the file
    head = raw[:200_000].decode("latin-1", errors="ignore")

    def num(tag):
        m = re.search(rf'{tag}(?:="|>)\s*([+-]?\d+(?:\.\d+)?)', head)
        return float(m.group(1)) if m else None

    meta.gimbal_pitch_deg = num("GimbalPitchDegree")
    meta.rel_altitude_m = num("RelativeAltitude")
    return meta


def focal_px(focal_35mm: float, width_px: int, height_px: int) -> float:
    """Focal length in pixels. 35 mm-equivalent is defined on the sensor diagonal (43.27 mm)."""
    return focal_35mm * math.hypot(width_px, height_px) / 43.267
