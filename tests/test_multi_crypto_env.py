from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from profit_pilot.env.multi_crypto_env import MultiCryptoTradingEnv


def make_env(price_array: np.ndarray, **overrides) -> MultiCryptoTradingEnv:
    defaults = {
        "price_array": price_array.astype(np.float32),
        "tech_array": np.ones((price_array.shape[0], 4), dtype=np.float32),
        "tickers": ["BTC/USDT", "ETH/USDT"],
        "lookback": 3,
        "initial_cash": 1_000.0,
        "reward_mode": "return_cost_risk_volatility",
        "max_position_fraction": 0.50,
        "max_trade_fraction": 0.50,
        "stop_loss_pct": 0.02,
        "max_drawdown_pct": 0.20,
        "min_trade_quantity": 1e-8,
    }
    defaults.update(overrides)
    return MultiCryptoTradingEnv(**defaults)


def test_stop_loss_liquidates_position() -> None:
    prices = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [97.0, 100.0],
        ]
    )
    env = make_env(prices)
    env.reset()
    env.cash = 900.0
    env.holdings[0] = 1.0
    env.entry_prices[0] = 100.0
    env.portfolio_value = 1_000.0

    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert env.holdings[0] == 0.0
    assert info["stop_loss_symbols"] == ["BTC/USDT"]


def test_drawdown_halt_liquidates_and_truncates() -> None:
    prices = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [70.0, 100.0],
        ]
    )
    env = make_env(prices)
    env.reset()
    env.cash = 0.0
    env.holdings[0] = 10.0
    env.entry_prices[0] = 100.0
    env.portfolio_value = 1_000.0
    env.peak_portfolio_value = 1_000.0

    _, _, _, truncated, info = env.step(np.zeros(2, dtype=np.float32))

    assert truncated
    assert info["risk_halted"]
    assert np.all(env.holdings == 0.0)


def test_position_cap_trims_oversized_holding() -> None:
    prices = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
        ]
    )
    env = make_env(prices)
    env.reset()
    env.cash = 200.0
    env.holdings[0] = 8.0
    env.entry_prices[0] = 100.0
    env.portfolio_value = 1_000.0

    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    max_allowed_value = info["portfolio_value"] * env.max_position_fraction
    assert info["position_cap_symbols"] == ["BTC/USDT"]
    assert env.holdings[0] * prices[-1, 0] <= max_allowed_value + 1.0


def test_volatility_reward_is_bounded() -> None:
    prices = np.array(
        [
            [100.0, 100.0],
            [101.0, 100.0],
            [102.0, 100.0],
            [103.0, 100.0],
        ]
    )
    env = make_env(prices)
    env.reset()
    env.cash = 900.0
    env.holdings[0] = 1.0
    env.entry_prices[0] = 102.0
    env.portfolio_value = 1_002.0

    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert -1.0 <= info["risk_adjusted_return"] <= 1.0


if __name__ == "__main__":
    test_stop_loss_liquidates_position()
    test_drawdown_halt_liquidates_and_truncates()
    test_position_cap_trims_oversized_holding()
    test_volatility_reward_is_bounded()
    print("multi_crypto_env risk/reward checks passed")
