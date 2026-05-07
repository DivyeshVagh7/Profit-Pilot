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


def slice_bundle_for_evaluation_window(
    bundle: dict,
    evaluation_start: str,
    lookback: int,
    evaluation_end: str | None = None,
) -> dict:
    timestamps = pd.to_datetime(bundle["timestamps"]["datetime"], utc=True).reset_index(drop=True)
    start_timestamp = pd.Timestamp(evaluation_start)
    start_index = int(timestamps.searchsorted(start_timestamp, side="left"))
    slice_start = start_index - lookback + 1
    if slice_start < 0:
        raise ValueError(
            "Not enough warm-up rows before evaluation_start. "
            "Move data.since earlier or lower environment.lookback."
        )

    if evaluation_end is None:
        slice_end = len(timestamps)
    else:
        end_timestamp = pd.Timestamp(evaluation_end)
        slice_end = int(timestamps.searchsorted(end_timestamp, side="right"))
        if slice_end <= start_index:
            raise ValueError("evaluation_end must be after evaluation_start")

    sliced_timestamps = timestamps.iloc[slice_start:slice_end].reset_index(drop=True)
    first_eval_timestamp = sliced_timestamps.iloc[lookback - 1]
    metadata = dict(bundle["metadata"])
    metadata.update(
        {
            "evaluation_start": evaluation_start,
            "evaluation_end": evaluation_end,
            "first_evaluation_timestamp": str(first_eval_timestamp),
            "warmup_rows_in_slice": int(lookback - 1),
            "price_array_shape": list(bundle["price_array"][slice_start:slice_end].shape),
            "tech_array_shape": list(bundle["tech_array"][slice_start:slice_end].shape),
        }
    )
    return {
        **bundle,
        "price_array": bundle["price_array"][slice_start:slice_end],
        "tech_array": bundle["tech_array"][slice_start:slice_end],
        "timestamps": pd.DataFrame({"datetime": sliced_timestamps.astype(str)}),
        "metadata": metadata,
    }


def slice_bundle_for_evaluation_start(bundle: dict, evaluation_start: str, lookback: int) -> dict:
    return slice_bundle_for_evaluation_window(
        bundle=bundle,
        evaluation_start=evaluation_start,
        lookback=lookback,
    )
