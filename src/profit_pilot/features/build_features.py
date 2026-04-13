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
    "open",
    "high",
    "low",
    "close",
    "volume",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi",
    "cci",
    "adx",
]


def load_symbol_csv(raw_root: Path, exchange_name: str, timeframe: str, symbol: str) -> pd.DataFrame:
    path = raw_root / exchange_name / timeframe / f"{symbol_slug(symbol)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")
    return pd.read_csv(path)


def add_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["datetime"] = pd.to_datetime(enriched["datetime"], utc=True)

    macd_indicator = MACD(close=enriched["close"])
    enriched["macd"] = macd_indicator.macd()
    enriched["macd_signal"] = macd_indicator.macd_signal()
    enriched["macd_hist"] = macd_indicator.macd_diff()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build technical features and array files from raw OHLCV CSVs.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to the project config YAML file.")
    args = parser.parse_args()

    config = load_project_config(args.config)
    raw_root = Path(config.data.raw_dir)
    processed_root = Path(config.data.processed_dir)
    exchange_name = get_exchange_name(config.project.exchange)

    symbol_frames: dict[str, pd.DataFrame] = {}
    for symbol in config.data.symbols:
        raw_frame = load_symbol_csv(raw_root, exchange_name, config.data.timeframe, symbol)
        symbol_frames[symbol] = add_technical_indicators(raw_frame)

    timestamps, aligned_frames = align_frames(symbol_frames)
    price_array, tech_array, feature_export = build_arrays(config.data.symbols, aligned_frames)

    output_dir = bundle_directory(processed_root, config.data.timeframe, config.data.symbols)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "price_array.npy", price_array)
    np.save(output_dir / "tech_array.npy", tech_array)
    pd.DataFrame({"datetime": timestamps.astype(str)}).to_csv(output_dir / "timestamps.csv", index=False)
    feature_export.to_parquet(output_dir / "feature_frame.parquet", index=False)

    metadata = {
        "symbols": config.data.symbols,
        "feature_columns_per_asset": BASE_FEATURE_COLUMNS,
        "n_assets": len(config.data.symbols),
        "n_features_per_asset": len(BASE_FEATURE_COLUMNS),
        "price_array_shape": list(price_array.shape),
        "tech_array_shape": list(tech_array.shape),
        "timeframe": config.data.timeframe,
        "since": config.data.since,
        "until": config.data.until,
    }
    save_json(metadata, output_dir / "metadata.json")

    print(f"Saved processed bundle to {output_dir}")
    print(f"price_array shape: {price_array.shape}")
    print(f"tech_array shape: {tech_array.shape}")


if __name__ == "__main__":
    main()
