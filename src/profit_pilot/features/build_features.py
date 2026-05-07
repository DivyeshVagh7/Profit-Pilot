from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, CCIIndicator, MACD

from profit_pilot.config import get_exchange_name, load_project_config
from profit_pilot.utils.io import bundle_directory, save_json, symbol_slug


BASE_FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_3",
    "log_return_12",
    "rolling_volatility_12",
    "rolling_volatility_48",
    "rolling_volatility_96",
    "high_low_range",
    "close_to_open",
    "close_to_sma_12",
    "close_to_sma_48",
    "close_to_sma_96",
    "volume_zscore_48",
    "volume_zscore_96",
    "log_volume_change_1",
    "macd_pct",
    "macd_signal_pct",
    "macd_hist_pct",
    "rsi",
    "cci",
    "adx",
    "trend_regime",
    "volatility_regime",
]


def load_symbol_csv(raw_root: Path, exchange_name: str, timeframe: str, symbol: str) -> pd.DataFrame:
    path = raw_root / exchange_name / timeframe / f"{symbol_slug(symbol)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")
    return pd.read_csv(path)


def filter_frame_by_date(frame: pd.DataFrame, since: str, until: str) -> pd.DataFrame:
    working = frame.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], utc=True)
    start = pd.Timestamp(since)
    end = pd.Timestamp(until)
    filtered = working[(working["datetime"] >= start) & (working["datetime"] <= end)]
    if filtered.empty:
        raise ValueError(f"No rows found between {since} and {until}")
    return filtered.sort_values("datetime").reset_index(drop=True)


def add_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["datetime"] = pd.to_datetime(enriched["datetime"], utc=True)

    close = enriched["close"].astype(float)
    open_price = enriched["open"].astype(float)
    high = enriched["high"].astype(float)
    low = enriched["low"].astype(float)
    volume = enriched["volume"].astype(float)
    log_close = np.log(close.clip(lower=1e-12))
    log_volume = np.log1p(volume.clip(lower=0.0))

    macd_indicator = MACD(close=enriched["close"])
    macd = macd_indicator.macd()
    macd_signal = macd_indicator.macd_signal()
    macd_hist = macd_indicator.macd_diff()

    enriched["log_return_1"] = log_close.diff(1)
    enriched["log_return_3"] = log_close.diff(3)
    enriched["log_return_12"] = log_close.diff(12)
    enriched["rolling_volatility_12"] = enriched["log_return_1"].rolling(12).std()
    enriched["rolling_volatility_48"] = enriched["log_return_1"].rolling(48).std()
    enriched["rolling_volatility_96"] = enriched["log_return_1"].rolling(96).std()
    enriched["high_low_range"] = (high - low) / close.clip(lower=1e-12)
    enriched["close_to_open"] = (close / open_price.clip(lower=1e-12)) - 1.0

    sma_12 = close.rolling(12).mean()
    sma_48 = close.rolling(48).mean()
    sma_96 = close.rolling(96).mean()
    enriched["close_to_sma_12"] = (close / sma_12.clip(lower=1e-12)) - 1.0
    enriched["close_to_sma_48"] = (close / sma_48.clip(lower=1e-12)) - 1.0
    enriched["close_to_sma_96"] = (close / sma_96.clip(lower=1e-12)) - 1.0

    volume_mean_48 = volume.rolling(48).mean()
    volume_std_48 = volume.rolling(48).std().replace(0.0, np.nan)
    volume_mean_96 = volume.rolling(96).mean()
    volume_std_96 = volume.rolling(96).std().replace(0.0, np.nan)
    enriched["volume_zscore_48"] = (volume - volume_mean_48) / volume_std_48
    enriched["volume_zscore_96"] = (volume - volume_mean_96) / volume_std_96
    enriched["log_volume_change_1"] = log_volume.diff(1)

    enriched["macd_pct"] = macd / close.clip(lower=1e-12)
    enriched["macd_signal_pct"] = macd_signal / close.clip(lower=1e-12)
    enriched["macd_hist_pct"] = macd_hist / close.clip(lower=1e-12)
    enriched["rsi"] = RSIIndicator(close=enriched["close"], window=14).rsi()
    enriched["cci"] = CCIIndicator(
        high=enriched["high"],
        low=enriched["low"],
        close=enriched["close"],
        window=14,
    ).cci()
    enriched["adx"] = ADXIndicator(
        high=enriched["high"],
        low=enriched["low"],
        close=enriched["close"],
        window=14,
    ).adx()

    enriched["trend_regime"] = np.where(sma_12 >= sma_48, 1.0, -1.0)
    volatility_ratio = enriched["rolling_volatility_12"] / enriched["rolling_volatility_96"].clip(lower=1e-12)
    enriched["volatility_regime"] = volatility_ratio.clip(lower=0.0, upper=5.0) - 1.0

    enriched = enriched.dropna().reset_index(drop=True)
    return enriched


def align_frames(symbol_frames: dict[str, pd.DataFrame]) -> tuple[pd.Index, list[pd.DataFrame]]:
    common_index = None
    aligned_frames: list[pd.DataFrame] = []

    for symbol, frame in symbol_frames.items():
        indexed = frame.set_index("datetime").sort_index()
        if common_index is None:
            common_index = indexed.index
        else:
            common_index = common_index.intersection(indexed.index)
        symbol_frames[symbol] = indexed

    if common_index is None or common_index.empty:
        raise ValueError("No common timestamps found across assets.")

    for symbol in symbol_frames:
        aligned_frames.append(symbol_frames[symbol].loc[common_index].copy())

    return common_index, aligned_frames


def build_arrays(symbols: list[str], aligned_frames: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    price_matrix = np.column_stack([frame["close"].to_numpy(dtype=np.float32) for frame in aligned_frames])

    feature_parts = []
    long_frames = []
    for symbol, frame in zip(symbols, aligned_frames, strict=True):
        feature_frame = frame[BASE_FEATURE_COLUMNS].copy()
        feature_parts.append(feature_frame.to_numpy(dtype=np.float32))

        export_frame = feature_frame.copy()
        export_frame["symbol"] = symbol
        export_frame["datetime"] = frame.index
        long_frames.append(export_frame)

    tech_array = np.concatenate(feature_parts, axis=1)
    feature_export = pd.concat(long_frames, ignore_index=True)
    return price_matrix, tech_array, feature_export


def compute_normalization_stats(tech_array: np.ndarray) -> dict[str, np.ndarray]:
    tech_mean = tech_array.mean(axis=0).astype(np.float32)
    tech_std = tech_array.std(axis=0).astype(np.float32)
    tech_std = np.where(tech_std < 1e-8, 1.0, tech_std).astype(np.float32)
    return {"mean": tech_mean, "std": tech_std}


def load_normalization_stats(stats_dir: str | Path) -> dict[str, np.ndarray]:
    root = Path(stats_dir)
    mean_path = root / "tech_mean.npy"
    std_path = root / "tech_std.npy"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(f"Missing tech_mean.npy or tech_std.npy in normalization stats directory: {root}")
    return {
        "mean": np.load(mean_path).astype(np.float32),
        "std": np.load(std_path).astype(np.float32),
    }


def normalize_tech_array(tech_array: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    if stats["mean"].shape[0] != tech_array.shape[1] or stats["std"].shape[0] != tech_array.shape[1]:
        raise ValueError(
            "Normalization stats feature count does not match tech_array: "
            f"stats={stats['mean'].shape[0]}, tech_array={tech_array.shape[1]}"
        )
    return ((tech_array - stats["mean"]) / stats["std"]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build technical features and array files from raw OHLCV CSVs.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to the project config YAML file.")
    parser.add_argument(
        "--normalization-stats-dir",
        default=None,
        help="Optional processed bundle directory containing tech_mean.npy and tech_std.npy to reuse.",
    )
    args = parser.parse_args()

    config = load_project_config(args.config)
    raw_root = Path(config.data.raw_dir)
    processed_root = Path(config.data.processed_dir)
    exchange_name = get_exchange_name(config.project.exchange)

    symbol_frames: dict[str, pd.DataFrame] = {}
    for symbol in config.data.symbols:
        raw_frame = load_symbol_csv(raw_root, exchange_name, config.data.timeframe, symbol)
        raw_frame = filter_frame_by_date(raw_frame, config.data.since, config.data.until)
        symbol_frames[symbol] = add_technical_indicators(raw_frame)

    timestamps, aligned_frames = align_frames(symbol_frames)
    price_array, raw_tech_array, feature_export = build_arrays(config.data.symbols, aligned_frames)

    stats_dir = args.normalization_stats_dir or config.data.normalization_stats_dir
    if stats_dir:
        normalization_stats = load_normalization_stats(stats_dir)
        normalization_source = str(stats_dir)
    else:
        normalization_stats = compute_normalization_stats(raw_tech_array)
        normalization_source = "computed_from_current_bundle"
    tech_array = normalize_tech_array(raw_tech_array, normalization_stats)

    output_dir = bundle_directory(processed_root, config.data.timeframe, config.data.symbols)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "price_array.npy", price_array)
    np.save(output_dir / "tech_array.npy", tech_array)
    np.save(output_dir / "tech_mean.npy", normalization_stats["mean"])
    np.save(output_dir / "tech_std.npy", normalization_stats["std"])
    pd.DataFrame({"datetime": timestamps.astype(str)}).to_csv(output_dir / "timestamps.csv", index=False)
    feature_export.to_parquet(output_dir / "feature_frame.parquet", index=False)

    metadata = {
        "label": output_dir.name,
        "symbols": config.data.symbols,
        "feature_columns_per_asset": BASE_FEATURE_COLUMNS,
        "n_assets": len(config.data.symbols),
        "n_features_per_asset": len(BASE_FEATURE_COLUMNS),
        "price_array_shape": list(price_array.shape),
        "tech_array_shape": list(tech_array.shape),
        "timeframe": config.data.timeframe,
        "since": config.data.since,
        "until": config.data.until,
        "source_since": config.data.since,
        "source_until": config.data.until,
        "first_timestamp_after_indicator_warmup": str(timestamps.min()),
        "last_timestamp": str(timestamps.max()),
        "normalization": "standard_score",
        "normalization_stats_source": normalization_source,
    }
    save_json(metadata, output_dir / "metadata.json")

    print(f"Saved processed bundle to {output_dir}")
    print(f"price_array shape: {price_array.shape}")
    print(f"tech_array shape: {tech_array.shape}")


if __name__ == "__main__":
    main()
