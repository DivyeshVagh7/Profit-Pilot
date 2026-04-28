from __future__ import annotations

import argparse
from pathlib import Path
import time

import ccxt
import pandas as pd

from profit_pilot.config import get_binance_credentials, get_exchange_name, load_project_config
from profit_pilot.utils.io import symbol_slug


OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def create_exchange(exchange_name: str, sandbox: bool = False):
    api_key, api_secret = get_binance_credentials()
    exchange_class = getattr(ccxt, exchange_name)
    params = {"enableRateLimit": True}

    if api_key and api_secret:
        params["apiKey"] = api_key
        params["secret"] = api_secret

    exchange = exchange_class(params)
    if sandbox and hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(True)
    return exchange


def fetch_symbol_ohlcv(
    exchange,
    symbol: str,
    timeframe: str,
    since: str,
    until: str,
    limit: int,
) -> pd.DataFrame:
    since_ms = exchange.parse8601(since)
    until_ms = exchange.parse8601(until)
    cursor = since_ms
    rows: list[list[float]] = []

    while cursor is not None and cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break

        rows.extend(batch)
        last_timestamp = batch[-1][0]
        if last_timestamp >= until_ms or last_timestamp == cursor:
            break

        cursor = last_timestamp + 1
        time.sleep(exchange.rateLimit / 1000)

    frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    if frame.empty:
        raise ValueError(f"No data returned for {symbol}.")

    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    frame = frame[frame["timestamp"] <= until_ms].copy()
    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["symbol"] = symbol
    return frame


def save_symbol_csv(frame: pd.DataFrame, raw_dir: Path, exchange_name: str, timeframe: str, symbol: str) -> Path:
    target_dir = raw_dir / exchange_name / timeframe
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{symbol_slug(symbol)}.csv"
    frame.to_csv(target_path, index=False)
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV candles from Binance or another CCXT exchange.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to the project config YAML file.")
    parser.add_argument("--sandbox", action="store_true", help="Enable sandbox mode when supported by the exchange.")
    args = parser.parse_args()

    config = load_project_config(args.config)
    exchange_name = get_exchange_name(config.project.exchange)
    exchange = create_exchange(exchange_name, sandbox=args.sandbox)
    raw_root = Path(config.data.raw_dir)

    print(f"Using exchange: {exchange_name}")
    if not get_binance_credentials()[0]:
        print("No API key found. Using public market-data endpoints only.")

    for symbol in config.data.symbols:
        print(f"Downloading {symbol} {config.data.timeframe} candles...")
        frame = fetch_symbol_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe=config.data.timeframe,
            since=config.data.since,
            until=config.data.until,
            limit=config.data.download_limit,
        )
        target = save_symbol_csv(frame, raw_root, exchange_name, config.data.timeframe, symbol)
        print(f"Saved {len(frame)} rows to {target}")

    if hasattr(exchange, "close"):
        exchange.close()


if __name__ == "__main__":
    main()
