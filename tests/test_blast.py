import math

import pytest

from grain_analyzer.blast import (BlastInputs, calibrate, uniformity_n, x50_model_cm, xc_from_x50)


def test_x50_hand_calculation():
    # A=8, K=0.6 kg/m3, Q=100 kg, ANFO: 8 * 0.6^-0.8 * 100^(1/6) * (100/115)^(-19/30) = 28.3 cm
    assert x50_model_cm(8.0, 0.6, 100.0, 100.0) == pytest.approx(28.34, abs=0.05)


def test_uniformity_index_hand_calculation():
    b = BlastInputs(burden_m=3.0, spacing_m=3.5, hole_diameter_mm=100.0, bench_height_m=10.0,
                    drilling_error_m=0.1, bottom_charge_m=1.0, column_charge_m=5.0)
    # (2.2-0.42) * sqrt(1.0833) * (1-0.0333) * (0.6667+0.1)^0.1 * 0.6 = 1.046
    assert uniformity_n(b) == pytest.approx(1.046, abs=0.002)


def test_incomplete_inputs_give_none_not_garbage():
    assert uniformity_n(BlastInputs()) is None
    out = calibrate(BlastInputs(), 250.0, 0.9)
    assert out["ka"] is None and out["kn"] is None and out["n_model"] is None


def test_calibration_round_trip_gives_unit_factors():
    b = BlastInputs(burden_m=3.0, spacing_m=3.5, hole_diameter_mm=100.0, bench_height_m=10.0,
                    charge_per_hole_kg=60.0, rws=100.0, drilling_error_m=0.1, bottom_charge_m=1.0,
                    column_charge_m=5.0, rock_factor_a=8.0)
    k = b.powder_factor
    assert k == pytest.approx(60 / (3.0 * 3.5 * 10.0))
    x50 = x50_model_cm(8.0, k, 60.0, 100.0) * 10            # a "measurement" exactly equal to the model, in mm
    out = calibrate(b, x50, uniformity_n(b))
    assert out["ka"] == pytest.approx(1.0) and out["kn"] == pytest.approx(1.0)
    assert out["A_measured"] == pytest.approx(8.0)
    # a coarser result than the model predicts -> a larger rock factor
    assert calibrate(b, x50 * 1.5, uniformity_n(b))["ka"] > 1.0
    # lower measured n -> kn < 1
    assert calibrate(b, x50, uniformity_n(b) * 0.7)["kn"] == pytest.approx(0.7)
    assert xc_from_x50(10.0, 1.0) == pytest.approx(10.0 / math.log(2))
