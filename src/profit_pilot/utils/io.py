from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


def symbol_slug(symbol: str) -> str:
    return symbol.lower().replace("/", "_").replace(":", "_").replace("-", "_")


def bundle_name(timeframe: str, symbols: list[str]) -> str:
    symbol_part = "_".join(symbol_slug(symbol) for symbol in symbols)
    return f"{timeframe}_{symbol_part}"


def bundle_directory(processed_root: str | Path, timeframe: str, symbols: list[str]) -> Path:
    return Path(processed_root) / bundle_name(timeframe, symbols)


def save_json(payload: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_processed_bundle(processed_root: str | Path, timeframe: str, symbols: list[str]) -> dict:
    root = bundle_directory(processed_root, timeframe, symbols)
    price_array = np.load(root / "price_array.npy")
    tech_array = np.load(root / "tech_array.npy")
    timestamps = pd.read_csv(root / "timestamps.csv")
    metadata = load_json(root / "metadata.json")
    return {
        "root": root,
        "price_array": price_array,
        "tech_array": tech_array,
        "timestamps": timestamps,
        "metadata": metadata,
    }
