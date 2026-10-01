"""Drone mode: person-based scale geometry, tiling, end-to-end with stub models."""
import math

import cv2
import numpy as np
import pytest

from grain_analyzer import drone
from grain_analyzer.camera import focal_px, read_camera_meta
from grain_analyzer.pipeline import DroneParams, analyze_drone

W, H, F = 2000, 1500, 1500.0


@pytest.mark.parametrize("pitch", [30, 45, 60, 75])
@pytest.mark.parametrize("z_true", [15.0, 40.0, 120.0])
def test_person_distance_roundtrip(pitch, z_true):
    feet = (W / 2 + 250, H / 2 - 150)
    span = drone.person_span_px(z_true, feet, F, W / 2, H / 2, pitch, 1.75)
    # reconstruct a head pixel at that span, straight up in the image from the feet
    head = (feet[0], feet[1] - span)
    got = drone.solve_person_distance(head, feet, F, W / 2, H / 2, pitch, 1.75)
    assert got == pytest.approx(z_true, rel=0.02) or abs(got - z_true) / z_true < 0.1


def test_closer_person_looks_bigger_and_steeper_view_shrinks_span():
    feet = (W / 2, H / 2)
    near = drone.person_span_px(20, feet, F, W / 2, H / 2, 45, 1.75)
    far = drone.person_span_px(40, feet, F, W / 2, H / 2, 45, 1.75)
    assert near == pytest.approx(2 * far, rel=0.05)
    steep = drone.person_span_px(20, feet, F, W / 2, H / 2, 85, 1.75)
    assert steep < 0.3 * near                    # near-nadir: person is almost a dot


def test_degenerate_clicks_rejected():
    with pytest.raises(ValueError):
        drone.solve_person_distance((1000, 750), (1001, 751), F, W / 2, H / 2, 45, 1.75)


def test_huge_span_means_very_close_person():
    z = drone.solve_person_distance((1000, -3000), (1000, 750), F, W / 2, H / 2, 45, 1.75)
    assert 1.2 < z < 3                           # a person filling the frame is ~2 m away


def test_scale_map_ratio_and_person_anchor():
    depth = np.full((H, W), 40.0, np.float32)
    depth[:, :1000] = 20.0                       # left half is twice as close
    head, feet = (1500.0, 500.0), (1500.0, 530.0)
    cal = drone._calibrate(head, feet, depth, F, 45, 1.75, W / 2, H / 2)
    s = drone.scale_map(depth, cal, head, feet)
    assert s[100, 1500] == pytest.approx(cal.mm_per_px_person, rel=1e-3)
    assert s[100, 100] == pytest.approx(s[100, 1500] / 2, rel=1e-3)


def test_tiles_cover_box():
    cover = np.zeros((1500, 2000), bool)
    for ya, yb, xa, xb in drone.tile_boxes(100, 50, 1900, 1400, 800, 0.2):
        cover[ya:yb, xa:xb] = True
    assert cover[50:1400, 100:1900].all()


class Stub:
    """Segmenter returning perfect circles (in tile coordinates) + a huge background blob."""
    def __init__(self, circles, big=True):
        self.circles, self.big = circles, big

    def segment(self, tile):
        h, w = tile.shape[:2]
        out = []
        for (cx, cy, r) in self.circles:
            m = np.zeros((h, w), np.uint8)
            cv2.circle(m, (cx, cy), r, 1, -1)
            if m.any():
                out.append((m.astype(bool), 0.9))
        if self.big:
            out.append((np.ones((h, w), bool), 0.99))
        return out


class ConstDepth:
    def __init__(self, z):
        self.z = z

    def estimate(self, img):
        return np.full(img.shape[:2], self.z, np.float32)


def test_end_to_end_constant_depth():
    img = np.zeros((1500, 2000, 3), np.uint8)
    p = DroneParams(max_side=2000, pitch_deg=45, focal_35mm=36.0, tile=2000, max_area_frac=0.05)
    f = focal_px(36.0, 2000, 1500)
    # circles are placed in absolute coords: 1 tile covers the whole roi (starts at roi origin)
    roi = (100, 100, 1900, 1400)
    circles = [(400 - roi[0], 400 - roi[1], 30), (900 - roi[0], 700 - roi[1], 45),
               (1400 - roi[0], 900 - roi[1], 60)]
    z = 25.0
    res = analyze_drone(img, p, Stub(circles), ConstDepth(z), head_xy=(1500, 300),
                        feet_xy=(1500, 330), roi_xyxy=roi)
    assert len(res.grains) == 3                   # the giant blob is rejected by max area
    s = res.grains["mm_per_px"].iloc[0]
    # all grains share one scale (constant depth), ECD = 2 r * s
    np.testing.assert_allclose(np.sort(res.grains["ecd_mm"]), np.array([60, 90, 120]) * s,
                               rtol=0.06)
    assert res.contours is not None and len(res.contours) == 3
    assert "Pokrycie" in " ".join(res.notes)


def test_camera_meta_missing_is_none():
    ok, buf = cv2.imencode(".jpg", np.zeros((10, 10, 3), np.uint8))
    m = read_camera_meta(buf.tobytes())
    assert m.focal_35mm is None and m.gimbal_pitch_deg is None


def test_camera_meta_dji_xmp():
    xmp = b'<x drone-dji:GimbalPitchDegree="-52.30" drone-dji:RelativeAltitude="+61.2"/>'
    m = read_camera_meta(b"\xff\xd8" + xmp)
    assert m.gimbal_pitch_deg == pytest.approx(-52.3) and m.rel_altitude_m == pytest.approx(61.2)
    assert math.isclose(focal_px(24, 4000, 3000), 24 * 5000 / 43.267)
