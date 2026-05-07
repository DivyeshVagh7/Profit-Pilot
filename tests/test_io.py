from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from profit_pilot.utils.io import slice_bundle_for_evaluation_start


def test_slice_bundle_for_evaluation_start_preserves_lookback_warmup() -> None:
    timestamps = pd.date_range("2026-04-18 00:00:00+00:00", periods=10, freq="5min")
    bundle = {
        "root": Path("unused"),
        "price_array": np.ones((10, 2), dtype=np.float32),
        "tech_array": np.ones((10, 4), dtype=np.float32),
        "timestamps": pd.DataFrame({"datetime": timestamps.astype(str)}),
        "metadata": {},
    }

    sliced = slice_bundle_for_evaluation_start(
        bundle=bundle,
        evaluation_start="2026-04-18T00:20:00Z",
        lookback=3,
    )

    assert sliced["price_array"].shape == (8, 2)
    assert sliced["timestamps"]["datetime"].iloc[2] == "2026-04-18 00:20:00+00:00"
    assert sliced["metadata"]["warmup_rows_in_slice"] == 2


if __name__ == "__main__":
    test_slice_bundle_for_evaluation_start_preserves_lookback_warmup()
    print("io slicing checks passed")

