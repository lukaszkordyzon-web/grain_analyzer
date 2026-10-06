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


def _scene(max_area=0.05):
    img = np.zeros((1500, 2000, 3), np.uint8)
    p = DroneParams(max_stone_mm=0.0, max_side=2000, pitch_deg=45, focal_35mm=36.0, tile=2000, max_area_frac=max_area)
    roi = (100, 100, 1900, 1400)
    abs_circles = [(400, 400, 30), (900, 700, 45), (1400, 1100, 60)]
    circles = [(x - roi[0], y - roi[1], r) for x, y, r in abs_circles]
    return img, p, roi, abs_circles, circles


def test_plane_scale_map_geometry():
    f = focal_px(36.0, 2000, 1500)
    feet = (1500.0, 330.0)
    z = drone.solve_person_distance((1500, 300), feet, f, W / 2, H / 2, 45, 1.75)
    s = drone.plane_scale_map((H, W), f, 45, feet, z)
    assert s[330, 0] == pytest.approx(1000 * z / f, rel=1e-3)     # anchored at the person
    assert s[1400, 0] < s[700, 0] < s[330, 0] < s[100, 0]         # lower in frame = closer
    assert np.isfinite(s).all()


def test_end_to_end_ground_plane():
    img, p, roi, abs_circles, circles = _scene()
    head, feet = (1500, 300), (1500, 330)
    res = analyze_drone(img, p, Stub(circles), head, feet, roi)
    assert len(res.grains) == 3                   # the giant blob is rejected by max area
    f = focal_px(36.0, 2000, 1500)
    z = drone.solve_person_distance(head, feet, f, W / 2, H / 2, 45, 1.75)
    smap = drone.plane_scale_map((H, W), f, 45, feet, z)
    expected = sorted(2 * r * smap[y, x] for x, y, r in abs_circles)
    np.testing.assert_allclose(np.sort(res.grains["ecd_mm"]), expected, rtol=0.06)
    assert res.contours is not None and len(res.contours) == 3
    assert "Pokrycie" in " ".join(res.notes)


def test_end_to_end_depth_ratio_mode():
    img, p, roi, abs_circles, circles = _scene()
    res = analyze_drone(img, p, Stub(circles), (1500, 300), (1500, 330), roi,
                        depth_estimator=ConstDepth(25.0))
    s = res.grains["mm_per_px"]
    assert s.nunique() == 1                       # constant depth -> one scale everywhere
    np.testing.assert_allclose(np.sort(res.grains["ecd_mm"]), np.array([60, 90, 120]) * s.iloc[0],
                               rtol=0.06)


def test_camera_meta_missing_is_none():
    ok, buf = cv2.imencode(".jpg", np.zeros((10, 10, 3), np.uint8))
    m = read_camera_meta(buf.tobytes())
    assert m.focal_35mm is None and m.gimbal_pitch_deg is None


def test_camera_meta_dji_xmp():
    xmp = b'<x drone-dji:GimbalPitchDegree="-52.30" drone-dji:RelativeAltitude="+61.2"/>'
    m = read_camera_meta(b"\xff\xd8" + xmp)
    assert m.gimbal_pitch_deg == pytest.approx(-52.3) and m.rel_altitude_m == pytest.approx(61.2)
    assert math.isclose(focal_px(24, 4000, 3000), 24 * 5000 / 43.267)


def test_tile_guard_refuses_heavy_settings():
    from grain_analyzer.pipeline import plan_drone
    img = np.zeros((3000, 4000, 3), np.uint8)
    heavy = DroneParams(max_stone_mm=0.0, max_side=4000, tile=600)
    n, _ = plan_drone(img.shape[:2], (0, 0, 4000, 3000), heavy)
    assert n > heavy.max_tiles
    with pytest.raises(ValueError, match="kafelków"):
        analyze_drone(img, heavy, Stub([]), (100, 100), (100, 130), (0, 0, 4000, 3000))
    light = DroneParams(max_stone_mm=0.0, max_side=2000, tile=1024)
    assert plan_drone(img.shape[:2], (0, 0, 4000, 3000), light)[0] <= light.max_tiles


# ---------------------------------------------------------------- other scale references
def _project(p, f, cx, cy):
    return (f * p[0] / p[2] + cx, f * p[1] / p[2] + cy)


def _ground_point(pixel, d_m, f, cx, cy, pitch):
    """3D point where the ray through ``pixel`` meets the ground plane at camera height d_m."""
    ray = np.array([(pixel[0] - cx) / f, (pixel[1] - cy) / f, 1.0])
    return ray * d_m / float(drone._down_vector(pitch) @ ray)


@pytest.mark.parametrize("pitch", [35, 50, 70])
@pytest.mark.parametrize("slope", [90, 75])
def test_wall_reference_recovers_camera_height(pitch, slope):
    d_true, wall_h = 60.0, 12.0
    lean = 0.0 if slope == 90 else wall_h / math.tan(math.radians(slope))
    foot = _ground_point((W / 2 + 150, H / 2 + 100), d_true, F, W / 2, H / 2, pitch)
    top = foot - wall_h * drone._down_vector(pitch) + lean * drone._forward_horizontal(pitch)
    feet_px, head_px = _project(foot, F, W / 2, H / 2), _project(top, F, W / 2, H / 2)
    z = drone.solve_person_distance(head_px, feet_px, F, W / 2, H / 2, pitch, wall_h, lean)
    assert z == pytest.approx(foot[2], rel=0.01)
    assert drone.camera_height(z, feet_px, F, H / 2, pitch) == pytest.approx(d_true, rel=0.01)


def test_ignoring_wall_lean_overestimates_height_of_view_error():
    """A leaning wall treated as vertical gives a visibly different D - why the slope input exists."""
    pitch, d_true, wall_h = 50, 60.0, 12.0
    lean = wall_h / math.tan(math.radians(70))
    foot = _ground_point((W / 2, H / 2 + 100), d_true, F, W / 2, H / 2, pitch)
    top = foot - wall_h * drone._down_vector(pitch) + lean * drone._forward_horizontal(pitch)
    fp, hp = _project(foot, F, W / 2, H / 2), _project(top, F, W / 2, H / 2)
    z_naive = drone.solve_person_distance(hp, fp, F, W / 2, H / 2, pitch, wall_h, 0.0)
    d_naive = drone.camera_height(z_naive, fp, F, H / 2, pitch)
    assert abs(d_naive - d_true) / d_true > 0.05


@pytest.mark.parametrize("pitch", [30, 55, 80])
def test_marker_reference_recovers_camera_height(pitch):
    d_true, side = 45.0, 2.0
    p0 = _ground_point((W / 2 + 300, H / 2 + 200), d_true, F, W / 2, H / 2, pitch)
    e1 = drone._forward_horizontal(pitch)
    e2 = np.cross(drone._down_vector(pitch), e1)
    e2 /= np.linalg.norm(e2)
    corners = [_project(p0 + sx * side / 2 * e1 + sy * side / 2 * e2, F, W / 2, H / 2)
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    got = drone.camera_height_from_marker(corners, side, F, W / 2, H / 2, pitch)
    assert got == pytest.approx(d_true, rel=0.01)


def test_scale_map_from_camera_height_matches_plane_scale_map():
    f = 1500.0
    z = 70.0
    feet = (1000.0, 500.0)
    a = drone.plane_scale_map((H, W), f, 50, feet, z)
    d = drone.camera_height(z, feet, f, H / 2, 50)
    b = drone.scale_map_from_camera_height((H, W), f, 50, d)
    np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------- fines + reference modes
def _rr_scene(xc=150.0, n=1.1, d_min=200.0, roi_area=1.0):
    """Perfect world: area-passing is Rosin-Rammler(xc, n); only rocks >= d_min are 'measured'."""
    from grain_analyzer.stats import rr_passing
    edges = np.geomspace(d_min, 2500, 60)
    mid = np.sqrt(edges[:-1] * edges[1:])
    share = np.diff(rr_passing({"xc": xc, "n": n}, edges)) / 100       # area share per class
    return mid, share * roi_area


def test_fines_estimate_recovers_rosin_rammler():
    from grain_analyzer.stats import estimate_fines, passing_bounds, rr_percentile
    d, a = _rr_scene()
    b = passing_bounds(d, a, roi_area_mm2=1.0, min_size_mm=200.0)
    est = estimate_fines(b)
    assert est is not None and est["fit"]["n"] == pytest.approx(1.1, rel=0.05)
    true = {p: rr_percentile({"xc": 150.0, "n": 1.1}, p) for p in (10, 50, 90)}
    for p in (50, 90):                           # D50 comes from extrapolation below the limit
        assert est["D"][f"D{p}"] == pytest.approx(true[p], rel=0.10)
    assert est["D"]["D50"] < 200                  # below the measurement limit
    # D10 (19 mm) lies > 5x below the 200 mm limit: not quoted
    assert true[10] < 200 / 5 and np.isnan(est["D"]["D10"])


def test_fines_estimate_refused_for_unfittable_data():
    from grain_analyzer.stats import estimate_fines, passing_bounds
    b = passing_bounds(np.array([300.0, 310.0, 320.0]), np.array([1.0, 1.0, 1.0]), 1000.0, 200.0)
    assert estimate_fines(b) is None


def test_fraction_table_adds_up():
    from grain_analyzer.stats import fraction_table
    d, a = _rr_scene()
    rows = fraction_table(d, a * 0.8, roi_area_mm2=1.0, min_size_mm=200.0)   # 80 % of the area measured
    assert rows[0]["label"].startswith("< 200")
    assert sum(r["fraction"] for r in rows) == pytest.approx(1.0, abs=1e-6)
    assert rows[0]["fraction"] > 0.2             # unmeasured part is its own, explicit row


def test_marker_reference_nadir_scale():
    """Nadir view: a 200 mm marker spanning 200 px must give 1 mm/px everywhere."""
    img = np.full((1500, 2000, 3), 150, np.uint8)
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img[100:300, 100:300] = cv2.aruco.generateImageMarker(dic, 3, 200)[..., None]
    marker_blob = np.zeros((1300, 1800), bool)     # tile = roi (1800 x 1300)
    marker_blob[0:200, 0:200] = True              # SAM would also segment the marker itself
    p = DroneParams(max_stone_mm=0.0, max_side=2000, reference="marker", marker_size_mm=200.0, pitch_deg=90,
                    focal_35mm=36.0, tile=2000, max_area_frac=0.05)
    circles = [(800, 600, 40), (1200, 900, 55)]
    stub = Stub(circles, big=False)
    stub_masks = stub.segment

    def seg(tile):
        return stub_masks(tile) + [(marker_blob, 0.99)]
    stub.segment = seg
    res = analyze_drone(img, p, stub, roi_xyxy=(100, 100, 1900, 1400))
    assert res.scale.method.startswith("znacznik")
    assert len(res.grains) == 2                   # the marker is excluded
    np.testing.assert_allclose(res.grains["mm_per_px"], 1.0, rtol=0.03)
    np.testing.assert_allclose(np.sort(res.grains["ecd_mm"]), [80, 110], rtol=0.06)


def test_wall_reference_end_to_end_matches_person_math():
    img = np.zeros((1500, 2000, 3), np.uint8)
    f = focal_px(24.0, 2000, 1500)
    pitch, wall_h, d_true = 50.0, 12.0, 60.0
    foot = _ground_point((1000, 900), d_true, f, 1000, 750, pitch)
    top = foot - wall_h * drone._down_vector(pitch)
    feet_px, head_px = _project(foot, f, 1000, 750), _project(top, f, 1000, 750)
    p = DroneParams(max_stone_mm=0.0, max_side=2000, reference="wall", wall_height_m=wall_h, wall_slope_deg=90,
                    pitch_deg=pitch, focal_35mm=24.0, tile=2000, max_area_frac=0.05)
    res = analyze_drone(img, p, Stub([(900, 700, 40)], big=False), head_px, feet_px,
                        (100, 100, 1900, 1400))
    expected = drone.scale_map_from_camera_height((1500, 2000), f, pitch, d_true)[800, 1000]            # tile coords (900, 700) + roi origin (100, 100)
    assert res.scale.method.startswith("ściana")
    assert res.grains["mm_per_px"].iloc[0] == pytest.approx(expected, rel=0.02)


# ---------------------------------------------------------------- segment once, measure many times
class CountingStub(Stub):
    calls = 0

    def segment(self, tile):
        CountingStub.calls += 1
        return super().segment(tile)


def test_segmentation_is_reusable_across_references():
    from grain_analyzer.pipeline import measure_drone, segment_drone, segmentation_key
    img, p, roi, abs_circles, circles = _scene()
    CountingStub.calls = 0
    seg = segment_drone(img, p, CountingStub(circles, big=False), roi)
    n_calls = CountingStub.calls
    head, feet = (1500, 300), (1500, 330)
    a = measure_drone(seg, p, head, feet)                                   # person 1.75 m
    p2 = DroneParams(**{**p.__dict__, "person_height_m": 1.95})
    b = measure_drone(seg, p2, head, feet)                                  # taller person
    assert CountingStub.calls == n_calls                                    # SAM not run again
    assert (b.grains["ecd_mm"].to_numpy() != a.grains["ecd_mm"].to_numpy()).all()
    assert len(a.grains) == len(b.grains) == 3
    # scale-only settings do not change the cache key, segmentation settings do
    assert segmentation_key(p, roi) == segmentation_key(p2, roi)
    p3 = DroneParams(**{**p.__dict__, "tile": 800})
    assert segmentation_key(p, roi) != segmentation_key(p3, roi)


def test_preview_and_required_resolution():
    from grain_analyzer.pipeline import preview_drone, working_side_for
    p = DroneParams(max_stone_mm=0.0, pitch_deg=45, focal_35mm=24.0, reference="person")
    f = focal_px(24.0, 4000, 3000)
    z = drone.solve_person_distance((2000, 800), (2000, 830), f, 2000, 1500, 45, 1.75)
    pv = preview_drone((3000, 4000), p, (2000, 800), (2000, 830), (500, 900, 3500, 2800))
    assert pv is not None and pv["mm_per_px_orig"] > 0
    assert preview_drone((3000, 4000), p, None, None, (500, 900, 3500, 2800)) is None   # incomplete
    s_orig = pv["mm_per_px_orig"]
    side = working_side_for(300.0, s_orig, 4000, 12)       # 30 cm stone = 12 px
    assert side == pytest.approx(4000 * 12 * s_orig / 300.0, abs=1)
    assert working_side_for(150.0, s_orig, 4000, 12) > side  # smaller stone -> more pixels


def test_headline_d_uses_measured_part_and_curve_is_one_basis():
    """D values reachable on the measured part come from the curve itself (not the RR fit);
    the main plotted line is the whole-surface curve, starting at the unmeasured share."""
    import pandas as pd
    from grain_analyzer import scale as sm
    from grain_analyzer.pipeline import AnalysisResult
    from grain_analyzer.stats import estimate_fines, passing_bounds
    from grain_analyzer.viz import plot_psd
    d, a = _rr_scene(xc=90.0, n=0.75, d_min=170.0)
    b = passing_bounds(d, a, 1.0, 170.0)
    est = estimate_fines(b)
    assert est["D"]["D90"] == pytest.approx(b["D_upper"]["D90"])     # measured, not extrapolated
    res = AnalysisResult(np.zeros((5, 5, 3), np.uint8), sm.ScaleResult(1.0, "x"), [],
                         pd.DataFrame({"ecd_mm": d, "area_mm2": a}), None, "ecd_mm", {}, 0, [], {},
                         [], "area", 1.0, 170.0, b, est, [])
    fig = plot_psd(res, "ECD")
    main = fig.axes[0].lines[0]                    # first line = whole-surface (upper) curve
    assert main.get_ydata()[0] == pytest.approx(b["unmeasured_fraction"] * 100)   # starts at ~80 %, not 0
    assert fig.axes[0].lines[1].get_ydata()[0] == pytest.approx(0)               # lower bound starts at 0


def test_estimate_below_limit_is_continuous_with_measured_curve():
    """Regression: the extrapolated part used to end ~7 points below the measured curve (a jump)."""
    from grain_analyzer.stats import (estimate_fines, passing_bounds, rr_anchored_passing,
                                      rr_anchored_percentile)
    # a measured curve that is NOT Rosin-Rammler shaped near the limit (few stones just above it)
    d = np.concatenate([np.linspace(200, 300, 5), np.linspace(300, 1500, 60)])
    a = np.concatenate([np.full(5, 0.002), np.full(60, 0.16 / 60)])
    b = passing_bounds(d, a, 1.0, 200.0)
    est = estimate_fines(b, min_r2=0.0)
    u, d_min = b["unmeasured_fraction"], b["sizes"][0]
    grid = np.geomspace(est["floor_mm"], d_min, 50)
    y = rr_anchored_passing(est["fit"], grid, d_min, u)
    assert y[-1] == pytest.approx(100 * u)                  # joins the measured curve exactly
    assert (np.diff(y) > 0).all() and y[0] < y[-1]          # monotonic, below the limit
    p50 = rr_anchored_percentile(est["fit"], 50, d_min, u)
    assert p50 < d_min and not np.isnan(p50)                 # lies below the limit...
    assert np.isnan(est["D"]["D50"]) == (p50 < est["floor_mm"])   # ...and is only quoted above the floor
    assert np.isnan(rr_anchored_percentile(est["fit"], 100 * u + 1, d_min, u))   # above u: not below the limit


# ---------------------------------------------------------------- shadows + size cap
def test_shadow_labels_catches_cast_shadow_not_dark_rock_in_shadow():
    rng = np.random.default_rng(0)
    img = np.full((300, 300, 3), 130, np.uint8)
    img[:, 150:] = 40                                          # right half: a big shadowed zone
    labels = np.zeros((300, 300), np.int32)
    cv2.circle(labels, (60, 150), 25, 1, -1)                   # rock in the sunny part
    cv2.circle(labels, (110, 150), 14, 2, -1)                  # cast shadow right next to it
    cv2.circle(labels, (230, 150), 20, 3, -1)                  # dark rock inside the shadowed zone
    img[labels == 1] = np.clip(200 + rng.normal(0, 25, ((labels == 1).sum(), 1)), 0, 255)   # textured
    img[labels == 2] = 25                                      # uniform and much darker than its surroundings
    img[labels == 3] = np.clip(55 + rng.normal(0, 8, ((labels == 3).sum(), 1)), 0, 255)   # same level as the zone
    boxes = {}
    for lab in (1, 2, 3):
        ys, xs = np.nonzero(labels == lab)
        boxes[lab] = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    assert drone.shadow_labels(img, labels, boxes) == {2}


def test_max_stone_cap_and_shadow_filter_in_pipeline():
    img = np.full((1500, 2000, 3), 130, np.uint8)
    p = DroneParams(max_side=2000, pitch_deg=45, focal_35mm=36.0, tile=2000, max_area_frac=0.2,
                    max_stone_mm=3000.0)
    roi = (100, 100, 1900, 1400)
    circles = [(400, 400, 10), (900, 700, 20), (1400, 1100, 100)]    # the last one is ~10 m wide
    res = analyze_drone(img, p, Stub([(x - 100, y - 100, r) for x, y, r in circles], big=False),
                        (1500, 300), (1500, 330), roi)
    assert len(res.grains) == 2 and res.grains["ecd_mm"].max() < 3000
    assert any("zbyt dużych" in n for n in res.notes)
    res2 = analyze_drone(img, DroneParams(**{**p.__dict__, "max_stone_mm": 0.0}),
                         Stub([(x - 100, y - 100, r) for x, y, r in circles], big=False),
                         (1500, 300), (1500, 330), roi)
    assert len(res2.grains) == 3                                      # cap off -> kept


def test_fit_uses_reliable_part_and_gives_two_scenarios():
    """With a detection plateau just above the limit, fitting everything distorts the shape;
    fitting only the reliable part recovers it, and the two scenarios bracket the truth."""
    from grain_analyzer.stats import estimate_fines, passing_bounds, rr_percentile
    d, a = _rr_scene(xc=240.0, n=0.95, d_min=100.0)
    a = a * np.clip((d - 100) / 300, 0, 1).clip(0.0, 1.0) ** 1 * (d < 400) + a * (d >= 400)   # leaky below 400
    b = passing_bounds(d, a, 1.0, 100.0)
    truth = {k: rr_percentile({"xc": 240.0, "n": 0.95}, k) for k in (10, 50, 90)}
    est = estimate_fines(b, reliable_mm=400.0)
    assert est["fit_on"] == "reliable"
    assert est["fit"]["n"] == pytest.approx(0.95, rel=0.12)
    assert est["D"]["D50"] == pytest.approx(truth[50], rel=0.15)          # main = plain RR from the reliable part
    assert est["D_low"]["D50"] <= est["D"]["D50"]                          # "all unmeasured is fines" is lower
    est_all = estimate_fines(b)                                              # old behaviour: fit on everything
    assert abs(est_all["fit"]["n"] - 0.95) > abs(est["fit"]["n"] - 0.95)     # the plateau distorted it


def test_main_curve_never_exceeds_what_was_measured():
    """The fitted curve may not lie above the measured ceiling: below the limit that is the unmeasured
    share u, above it the "unmeasured = fines" curve."""
    from grain_analyzer.stats import (estimate_fines, passing_bounds, rr_main_passing, rr_passing)
    d, a = _rr_scene(xc=150.0, n=1.1, d_min=200.0)
    a = np.where(d < 300, a * 2.2, a)                  # extra stones just above the limit -> u is small
    b = passing_bounds(d, a, 1.0, 200.0)
    u = b["unmeasured_fraction"]
    est = estimate_fines(b, reliable_mm=300.0)
    assert rr_passing(est["fit"], 200.0) > 100 * u + 1                     # the plain fit would break the cap
    grid = np.geomspace(est["floor_mm"], 300.0, 60)
    main = rr_main_passing(est, grid, b)
    ceiling = np.where(grid < 200.0, 0, np.interp(grid, b["sizes"], b["upper"]))
    assert (main[grid >= 200.0] <= ceiling[grid >= 200.0] + 1e-9).all()    # above the limit: under the measured curve
    below = grid < 200.0
    assert main[below].max() <= 100 * u + 1e-9                             # below it: never more than u
    assert (np.diff(main) >= -1e-9).all()                                  # still a passing curve (monotonic)
    assert est["D_low"]["D50"] <= est["D"]["D50"] or np.isnan(est["D"]["D50"])


def test_rr_x50_matches_percentile_and_is_exported():
    from grain_analyzer.stats import rr_percentile, rr_x50
    fit = {"xc": 240.0, "n": 0.95}
    assert rr_x50(fit) == pytest.approx(rr_percentile(fit, 50))


def test_manual_fit_threshold_overrides_automatic_limit():
    img = np.full((1500, 2000, 3), 150, np.uint8)
    circles = [(300 + 90 * i, 300 + 60 * (i % 7), 20 + 3 * (i % 9)) for i in range(40)]
    kw = dict(max_stone_mm=0.0, max_side=2000, reference="person", pitch_deg=90, focal_35mm=36.0,
              tile=2000, max_area_frac=0.05)
    head, feet = (1000, 700), (1000, 800)
    auto = analyze_drone(img, DroneParams(**kw), Stub(circles, big=False), head, feet, (100, 100, 1900, 1400))
    man = analyze_drone(img, DroneParams(fit_from_mm=300.0, **kw), Stub(circles, big=False),
                        head, feet, (100, 100, 1900, 1400))
    assert man.reliable_mm == pytest.approx(max(300.0, man.min_size_mm))
    assert any("ręcznie" in n for n in man.notes)
    assert not any("ręcznie" in n for n in auto.notes)
