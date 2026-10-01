"""CSV export."""
from __future__ import annotations

import pandas as pd


def grains_csv(result) -> bytes:
    """Per-grain table (UTF-8 with BOM so Excel opens it correctly)."""
    return result.grains.round(4).to_csv(index=False).encode("utf-8-sig")


def summary_csv(result, size_label: str, weighting: str) -> bytes:
    rows = {"n_grains": len(result.grains), "mm_per_px": result.scale.mm_per_px,
            "scale_method": result.scale.method, "size_metric": size_label,
            "weighting": weighting, **result.percentiles}
    return pd.Series(rows).rename("value").to_csv(index_label="parameter").encode("utf-8-sig")
