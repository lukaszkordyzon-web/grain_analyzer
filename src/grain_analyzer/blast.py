"""Kuz-Ram blast fragmentation model (Cunningham 1983/87) and calibration of its factors.

Model (Rosin-Rammler with parameters from the blast design):
    X50 [cm] = A * K^-0.8 * Q^(1/6) * (E/115)^(-19/30)
    n        = (2.2 - 14 B/D) * sqrt((1 + S/B) / 2) * (1 - W/B) * (|BCL - CCL| / L + 0.1)^0.1 * L/H
    x_c      = X50 / (ln 2)^(1/n)

    A    rock factor                                  K    powder factor [kg/m3] = Q / (B S H)
    Q    explosive per hole [kg]                      E    relative weight strength (ANFO = 100)
    B    burden [m]       S spacing [m]               D    hole diameter [mm]
    H    bench height [m] W drilling error (std) [m] BCL/CCL bottom/column charge length [m]
    L    total charge length = BCL + CCL [m]

Calibration against a measured Rosin-Rammler fit (x_c, n from the photo):
    A_measured = X50_measured * K^0.8 / (Q^(1/6) * (E/115)^(-19/30))     ->  ka = A_measured / A_assumed
    kn = n_measured / n_model

The formulas are written from the literature as remembered: check them against your company's
standard before relying on the numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BlastInputs:
    burden_m: float = 0.0            # B
    spacing_m: float = 0.0           # S
    hole_diameter_mm: float = 0.0    # D
    bench_height_m: float = 0.0      # H
    charge_per_hole_kg: float = 0.0  # Q
    rws: float = 100.0               # E (ANFO = 100)
    drilling_error_m: float = 0.0    # W
    bottom_charge_m: float = 0.0     # BCL
    column_charge_m: float = 0.0     # CCL
    rock_factor_a: float = 0.0       # A assumed from the rock mass description (0 = not given)

    @property
    def charge_length_m(self) -> float:
        return self.bottom_charge_m + self.column_charge_m

    @property
    def powder_factor(self) -> float | None:
        v = self.burden_m * self.spacing_m * self.bench_height_m
        return self.charge_per_hole_kg / v if v > 0 and self.charge_per_hole_kg > 0 else None


def x50_model_cm(a: float, k: float, q: float, e: float = 100.0) -> float:
    return a * k ** -0.8 * q ** (1 / 6) * (e / 115) ** (-19 / 30)


def uniformity_n(b: BlastInputs) -> float | None:
    """Uniformity index n from the blast geometry (None when the inputs are incomplete)."""
    L = b.charge_length_m
    if min(b.burden_m, b.hole_diameter_mm, b.bench_height_m, L) <= 0 or b.spacing_m <= 0:
        return None
    return ((2.2 - 14 * b.burden_m / b.hole_diameter_mm)
            * math.sqrt((1 + b.spacing_m / b.burden_m) / 2)
            * (1 - b.drilling_error_m / b.burden_m)
            * (abs(b.bottom_charge_m - b.column_charge_m) / L + 0.1) ** 0.1
            * (L / b.bench_height_m))


def xc_from_x50(x50: float, n: float) -> float:
    return x50 / math.log(2) ** (1 / n)


def calibrate(b: BlastInputs, x50_measured_mm: float | None, n_measured: float | None) -> dict:
    """Model prediction and the correction factors ka, kn for one blast. ``x50_measured_mm`` /
    ``n_measured`` may be None (no usable fit): then only the model prediction is returned.
    Entries are None where the inputs do not allow the computation."""
    k = b.powder_factor
    out = {"K": k, "n_model": uniformity_n(b), "A_assumed": b.rock_factor_a or None,
           "A_measured": None, "ka": None, "kn": None, "X50_model_cm": None, "xc_model_cm": None}
    if k and b.charge_per_hole_kg > 0 and b.rws > 0:
        if x50_measured_mm:
            out["A_measured"] = (x50_measured_mm / 10) * k ** 0.8 / (
                b.charge_per_hole_kg ** (1 / 6) * (b.rws / 115) ** (-19 / 30))
            if b.rock_factor_a > 0:
                out["ka"] = out["A_measured"] / b.rock_factor_a
        if b.rock_factor_a > 0:
            out["X50_model_cm"] = x50_model_cm(b.rock_factor_a, k, b.charge_per_hole_kg, b.rws)
            if out["n_model"]:
                out["xc_model_cm"] = xc_from_x50(out["X50_model_cm"], out["n_model"])
    if out["n_model"] and n_measured:
        out["kn"] = n_measured / out["n_model"]
    return out
