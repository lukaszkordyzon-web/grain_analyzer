"""Model-free tests: synthetic image with an ArUco marker and elliptical 'grains'."""
import cv2
import numpy as np
import pytest

from grain_analyzer.pipeline import Params, analyze
from grain_analyzer.scale import detect_marker
from grain_analyzer.stats import size_distribution, weighted_percentiles

MARKER_PX, MARKER_MM = 100, 20.0           # -> 0.2 mm/px
RADII = [20, 25, 30, 35, 40]               # px, circles -> ECD = 2r*0.2 mm


def make_scene():
    img = np.full((500, 700, 3), 200, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = cv2.aruco.generateImageMarker(d, 7, MARKER_PX)
    img[30:30 + MARKER_PX, 30:30 + MARKER_PX] = m[..., None]
    masks = []
    for k, r in enumerate(RADII):
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(mask, (250 + 90 * k, 300), r, 1, -1)
        img[mask > 0] = (90, 60, 40)
        masks.append(mask.astype(bool))
    return img, masks


class FakeSegmenter:
    def __init__(self, masks):
        self.masks = masks

    def segment(self, image):
        return [(m, 0.95) for m in self.masks]


def test_marker_scale():
    img, _ = make_scene()
    s = detect_marker(img, MARKER_MM)
    assert s is not None and s.mm_per_px == pytest.approx(MARKER_MM / MARKER_PX, rel=0.03)


def test_no_marker_raises():
    img = np.full((200, 200, 3), 128, np.uint8)
    with pytest.raises(ValueError):
        analyze(img, Params(use_depth=False), FakeSegmenter([]))


def test_end_to_end_sizes():
    img, masks = make_scene()
    marker_blob = np.zeros(img.shape[:2], bool)
    marker_blob[30:130, 30:130] = True            # SAM would also return the marker itself
    res = analyze(img, Params(marker_size_mm=MARKER_MM, use_depth=False, min_area_px=50),
                  FakeSegmenter(masks + [marker_blob]))
    assert len(res.grains) == len(RADII)
    expected = np.array(RADII) * 2 * 0.2
    np.testing.assert_allclose(np.sort(res.sizes), expected, rtol=0.05)
    assert res.percentiles["D10"] < res.percentiles["D50"] < res.percentiles["D90"]
    assert len(res.masks) == len(res.grains)


def test_manual_scale_follows_resize():
    img, masks = make_scene()
    res = analyze(img, Params(manual_mm_per_px=0.2, use_depth=False, min_area_px=50, max_side=350),
                  FakeSegmenter([cv2.resize(m.astype(np.uint8), (350, 250)).astype(bool)
                                 for m in masks]))
    np.testing.assert_allclose(np.sort(res.sizes), np.array(RADII) * 0.4, rtol=0.08)


def test_percentiles():
    x = np.arange(1, 101, dtype=float)
    p = weighted_percentiles(x, None)
    assert p["D50"] == pytest.approx(50.5, abs=0.6)
    vol = size_distribution(x, "volume")
    assert vol["D50"] > p["D50"]                  # big grains dominate the volume


def test_cumulative_passing_curve():
    from grain_analyzer.stats import cumulative_passing
    s, pas = cumulative_passing(np.array([30.0, 10.0, 20.0]), "number")
    np.testing.assert_allclose(s, [10, 20, 30])
    np.testing.assert_allclose(pas, [100 / 3, 200 / 3, 100])
    _, pv = cumulative_passing(np.array([10.0, 20.0]), "volume")
    assert pv[0] == pytest.approx(100 * 1000 / 9000)           # 1000 / (1000 + 8000)


def test_passing_bounds_bracket_unmeasured_fines():
    from grain_analyzer.stats import passing_bounds
    sizes = np.array([200.0, 300.0, 400.0, 500.0])
    areas = np.array([10.0, 20.0, 30.0, 40.0])                 # measured = 100 of roi 400
    b = passing_bounds(sizes, areas, roi_area_mm2=400.0, min_size_mm=150.0)
    assert b["unmeasured_fraction"] == pytest.approx(0.75)
    assert (b["upper"] >= b["lower"]).all() and b["lower"][-1] == pytest.approx(100)
    assert b["upper"][-1] == pytest.approx(100) and b["upper"][0] == pytest.approx(75)
    assert np.isnan(b["D_upper"]["D10"]) and np.isnan(b["D_upper"]["D50"])   # below the limit
    assert 400 <= b["D_upper"]["D90"] <= 500                    # 90 % is reached among the rocks
    assert b["D_lower"]["D10"] < b["D_lower"]["D50"] < b["D_lower"]["D90"]


def test_psd_plot_renders():
    from grain_analyzer.pipeline import Params, analyze
    from grain_analyzer.viz import plot_psd
    img, masks = make_scene()
    res = analyze(img, Params(marker_size_mm=MARKER_MM, use_depth=False, min_area_px=50),
                  FakeSegmenter(masks))
    fig = plot_psd(res, "ECD")
    assert fig.axes[0].get_ylim() == (0, 100)


def test_device_override_and_tile_limit_env(monkeypatch):
    from grain_analyzer.device import pick_device
    from grain_analyzer.pipeline import DroneParams
    monkeypatch.setenv("GRAIN_DEVICE", "CPU")
    assert pick_device() == "cpu"
    monkeypatch.delenv("GRAIN_MAX_TILES", raising=False)
    assert DroneParams().max_tiles == 8                  # hosted default
    monkeypatch.setenv("GRAIN_MAX_TILES", "40")
    assert DroneParams().max_tiles == 40                 # strong local machine


def test_overlay_colours_by_size_green_to_red_with_legend():
    import pandas as pd
    from grain_analyzer import scale as sm
    from grain_analyzer.pipeline import AnalysisResult
    from grain_analyzer.viz import draw_overlay, size_colors
    img = np.full((400, 600, 3), 90, np.uint8)
    small = np.array([[[50, 50]], [[90, 50]], [[90, 90]], [[50, 90]]], np.int32)         # 40x40
    big = np.array([[[200, 200]], [[330, 200]], [[330, 330]], [[200, 330]]], np.int32)   # 130x130
    g = pd.DataFrame({"id": [1, 2], "ecd_mm": [50.0, 400.0], "cx_px": [70.0, 265.0], "cy_px": [70.0, 265.0]})
    res = AnalysisResult(img, sm.ScaleResult(1.0, "x"), [], g, None, "ecd_mm", {}, 0, [small, big])
    cols, lo, hi = size_colors(g["ecd_mm"].to_numpy())
    assert cols[0][1] > cols[0][0] and cols[1][0] > cols[1][1]        # small: green > red; large: red > green
    out = draw_overlay(res, thickness=3)
    px_small = out[50, 70]                                              # on the small stone's top edge
    px_big = out[200, 265]
    assert px_small[1] > px_small[0] and px_big[0] > px_big[1]
    assert (out[-12:, :40] != img[-12:, :40]).any()                     # legend drawn bottom-left
    plain = draw_overlay(res, thickness=3, color_by_size=False)
    assert tuple(plain[50, 70]) == (0, 255, 0)                          # old single-colour mode still there


def test_completeness_limit_finds_turnover_of_a_leaky_detection():
    from grain_analyzer.stats import completeness_limit
    rng = np.random.default_rng(1)
    d = 100 * (1 - rng.random(30000)) ** (-1 / 2.0)           # power-law sizes >= 100 (number grows to small)
    keep = rng.random(len(d)) < np.clip((d - 100) / 150, 0, 1)    # detection loses stones below 250
    lim, how = completeness_limit(d[keep][:1500], min_size_mm=110.0)
    assert how == "turnover" and 150 < lim < 330
    # too few stones -> fixed factor of the measurement limit
    lim2, how2 = completeness_limit(d[keep][:20], min_size_mm=110.0)
    assert how2 == "fallback" and lim2 == pytest.approx(110 * 1.7)
    # counts that only grow towards small sizes: no turnover, fallback
    lim3, how3 = completeness_limit(d[:1500], min_size_mm=110.0)
    assert how3 == "fallback"


def test_psd_plot_keeps_the_axis_label_and_marks_estimates_orange():
    import pickle  # noqa: F401  (only to keep the import block tidy for the test below)
    import pandas as pd
    from grain_analyzer import scale as sm
    from grain_analyzer.pipeline import AnalysisResult
    from grain_analyzer.stats import estimate_fines, passing_bounds
    from grain_analyzer.viz import plot_psd
    from test_drone import _rr_scene
    d, a = _rr_scene(xc=240.0, n=0.95, d_min=100.0)
    a = a * np.where(d < 400, np.clip((d - 100) / 300, 0, 1), 1.0)
    b = passing_bounds(d, a, 1.0, 100.0)
    est = estimate_fines(b, reliable_mm=400.0)
    res = AnalysisResult(np.zeros((5, 5, 3), np.uint8), sm.ScaleResult(1.0, "x"), [],
                         pd.DataFrame({"ecd_mm": d, "area_mm2": a}), None, "ecd_mm", {}, 0, [], {}, [],
                         "area", 1.0, 100.0, b, est, [], 400.0)
    fig = plot_psd(res, "Średnica ECD")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Średnica ECD [mm]"
    assert any("strefa niepełnego wykrywania do 40 cm" in t.get_text() for t in ax.texts)
    d50 = est["D"]["D50"]
    assert d50 < 400.0                                        # an estimate inside the unreliable zone
    dot = [ln for ln in ax.lines if ln.get_marker() == "o" and abs(ln.get_xdata()[0] - d50) < 1e-6][0]
    assert dot.get_color() == "#e8743b"                       # estimates are orange, not blue


def test_fastsam_adapter_lifts_the_300_detection_cap():
    """ultralytics keeps only max_det=300 detections per image by default: the adapter must raise it,
    otherwise small stones in a dense pile are silently dropped (observed on a real tile)."""
    import torch
    from grain_analyzer.segmentation import SEGMENTERS, UltralyticsSegmenter

    seen = {}

    class R:
        masks = type("M", (), {"data": torch.zeros((2, 16, 16), dtype=torch.bool)})()
        boxes = type("B", (), {"conf": torch.tensor([0.9, 0.8])})()

    class FakeModel:
        def __call__(self, img, **kw):
            seen.update(kw)
            return [R()]

    seg = UltralyticsSegmenter("FastSAM-s.pt", "fastsam")
    seg._model = FakeModel()
    out = seg.segment(np.zeros((16, 16, 3), np.uint8))
    assert len(out) == 2 and seen["max_det"] >= 1000 and seen["retina_masks"] is True
    spec = SEGMENTERS["fastsam"]
    assert spec.tile < 1024 and spec.max_tiles > SEGMENTERS["sam-b"].max_tiles
