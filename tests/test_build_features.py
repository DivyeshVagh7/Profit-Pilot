from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from profit_pilot.features.build_features import (
    BASE_FEATURE_COLUMNS,
    add_technical_indicators,
    compute_normalization_stats,
    filter_frame_by_date,
    normalize_tech_array,
)


def test_tech_normalization_centers_and_scales_columns() -> None:
    raw = np.asarray(
        [
            [1.0, 10.0, 5.0],
            [2.0, 10.0, 7.0],
            [3.0, 10.0, 9.0],
        ],
        dtype=np.float32,
    )

    stats = compute_normalization_stats(raw)
    normalized = normalize_tech_array(raw, stats)

    assert normalized.dtype == np.float32
    assert np.allclose(normalized[:, 0].mean(), 0.0, atol=1e-6)
    assert np.allclose(normalized[:, 0].std(), 1.0, atol=1e-6)
    assert np.allclose(normalized[:, 1], 0.0, atol=1e-6)
    assert np.allclose(normalized[:, 2].mean(), 0.0, atol=1e-6)


def test_filter_frame_by_date_respects_config_window() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-04-19 23:50:00+00:00", periods=4, freq="5min"),
            "close": [1.0, 2.0, 3.0, 4.0],
        }
    )

    filtered = filter_frame_by_date(
        frame,
        since="2026-04-19T23:55:00Z",
        until="2026-04-20T00:05:00Z",
    )

    assert filtered["close"].tolist() == [2.0, 3.0, 4.0]


def test_stationary_feature_columns_are_finite_after_warmup() -> None:
    periods = 160
    base_price = np.linspace(100.0, 120.0, periods)
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01 00:00:00+00:00", periods=periods, freq="5min"),
            "open": base_price,
            "high": base_price * 1.002,
            "low": base_price * 0.998,
            "close": base_price * (1.0 + 0.001 * np.sin(np.arange(periods))),
            "volume": 1_000.0 + 25.0 * np.cos(np.arange(periods)),
        }
    )

    enriched = add_technical_indicators(frame)

    assert set(BASE_FEATURE_COLUMNS).issubset(enriched.columns)
    assert "open" not in BASE_FEATURE_COLUMNS
    assert "close" not in BASE_FEATURE_COLUMNS
    assert np.isfinite(enriched[BASE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)).all()


if __name__ == "__main__":
    test_tech_normalization_centers_and_scales_columns()
    test_filter_frame_by_date_respects_config_window()
    test_stationary_feature_columns_are_finite_after_warmup()
    print("build_features normalization checks passed")
