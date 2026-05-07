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
        "action_mode": "target_allocation",
        "max_position_fraction": 0.50,
        "max_gross_exposure": 1.0,
        "max_trade_fraction": 1.0,
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


def test_positive_actions_buy_high_and_low_priced_assets() -> None:
    prices = np.array(
        [
            [50_000.0, 2_000.0],
            [50_000.0, 2_000.0],
            [50_000.0, 2_000.0],
            [50_000.0, 2_000.0],
        ]
    )
    env = make_env(prices, min_trade_quantity=0.0001)
    env.reset()

    _, _, _, _, info = env.step(np.ones(2, dtype=np.float32))

    assert env.holdings[0] > 0.0001
    assert env.holdings[1] > 0.0001
    assert env.holdings[0] * prices[-1, 0] <= info["portfolio_value"] * env.max_position_fraction + 1.0
    assert env.holdings[1] * prices[-1, 1] <= info["portfolio_value"] * env.max_position_fraction + 1.0


def test_target_allocation_rebalances_once_then_holds() -> None:
    prices = np.ones((8, 2), dtype=np.float32) * 100.0
    env = make_env(
        prices,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        min_trade_notional=1.0,
        rebalance_threshold=0.03,
    )
    env.reset()

    action = np.array([0.4, 0.2], dtype=np.float32)
    _, _, _, _, first_info = env.step(action)
    first_weights = first_info["asset_weights"]
    _, _, _, _, second_info = env.step(action)

    assert np.allclose(first_weights, np.array([0.4, 0.2]), atol=0.01)
    assert first_info["total_fees"] > 0.0
    assert second_info["total_fees"] == 0.0
    assert second_info["turnover"] == 0.0


def test_cash_target_allocation_can_choose_cash() -> None:
    prices = np.ones((8, 2), dtype=np.float32) * 100.0
    env = make_env(
        prices,
        action_mode="cash_target_allocation",
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
        max_position_fraction=1.0,
        max_gross_exposure=1.0,
        min_trade_notional=1.0,
    )
    observation, _ = env.reset()

    _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    assert env.action_space.shape == (3,)
    assert observation.shape == env.observation_space.shape
    assert np.allclose(info["asset_weights"], np.zeros(2), atol=1e-6)
    assert info["target_cash_weight"] == 1.0
    assert info["total_fees"] == 0.0


def test_cash_target_allocation_splits_scores_with_cash() -> None:
    prices = np.ones((8, 2), dtype=np.float32) * 100.0
    env = make_env(
        prices,
        action_mode="cash_target_allocation",
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
        max_position_fraction=1.0,
        max_gross_exposure=1.0,
        min_trade_notional=1.0,
    )
    env.reset()

    _, _, _, _, info = env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))

    assert np.allclose(info["target_weights"], np.array([1 / 3, 1 / 3], dtype=np.float32), atol=1e-6)
    assert np.allclose(info["asset_weights"], np.array([1 / 3, 1 / 3], dtype=np.float32), atol=1e-6)
    assert abs(info["cash_weight"] - 1 / 3) < 1e-6


def test_observation_contains_cash_drawdown_return_and_weights() -> None:
    prices = np.ones((8, 2), dtype=np.float32) * 100.0
    env = make_env(prices)

    observation, _ = env.reset()

    assert observation.shape == env.observation_space.shape
    assert np.isfinite(observation).all()
    assert observation[0] == 1.0
    assert observation[1] == 0.0
    assert observation[2] == 0.0
    assert np.allclose(observation[3 : 3 + env.n_assets], 0.0)


def test_cooldown_expires_after_sell() -> None:
    prices = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
        ]
    )
    env = make_env(prices, action_mode="trade_fraction", cooldown_steps=2, max_trade_fraction=1.0)
    env.reset()

    env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.holdings[0] > 0

    env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert env.holdings[0] == 0.0

    env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.holdings[0] == 0.0

    env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.holdings[0] > 0.0


def test_random_start_uses_fixed_episode_window() -> None:
    prices = np.ones((30, 2), dtype=np.float32) * 100.0
    env = make_env(prices, random_start=True, episode_length=5)

    env.reset(seed=7)
    assert env.lookback - 1 <= env.time <= env.max_step - env.episode_length
    assert env.episode_end_time == env.time + env.episode_length

    steps = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        done = terminated or truncated
        steps += 1

    assert steps == 5
    assert terminated


def test_deadband_and_min_notional_skip_tiny_trades() -> None:
    prices = np.ones((8, 2), dtype=np.float32) * 100.0

    deadband_env = make_env(prices, trade_deadband=0.10, max_trade_fraction=1.0, action_mode="trade_fraction")
    deadband_env.reset()
    deadband_env.step(np.array([0.05, 0.0], dtype=np.float32))
    assert deadband_env.holdings[0] == 0.0

    notional_env = make_env(prices, min_trade_notional=25.0, max_trade_fraction=0.01, action_mode="trade_fraction")
    notional_env.reset()
    notional_env.step(np.array([0.20, 0.0], dtype=np.float32))
    assert notional_env.holdings[0] == 0.0


def test_turnover_penalty_is_reported_and_reduces_reward() -> None:
    prices = np.ones((8, 2), dtype=np.float32) * 100.0
    env = make_env(
        prices,
        cost_penalty_weight=0.0,
        risk_penalty_weight=0.0,
        volatility_reward_weight=0.0,
        turnover_penalty_weight=0.1,
        reward_mode="net_log_return_alpha",
    )
    env.reset()

    _, reward, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

    assert info["turnover_penalty"] > 0.0
    assert reward < info["net_log_return"]


def test_cost_penalty_reduces_net_log_alpha_reward() -> None:
    prices = np.ones((8, 2), dtype=np.float32) * 100.0
    env = make_env(
        prices,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        cost_penalty_weight=1.0,
        risk_penalty_weight=0.0,
        turnover_penalty_weight=0.0,
        drawdown_penalty_weight=0.0,
        alpha_reward_weight=0.0,
        reward_mode="net_log_return_alpha",
    )
    env.reset()

    _, reward, _, _, info = env.step(np.array([0.5, 0.0], dtype=np.float32))

    assert info["cost_penalty"] > 0.0
    assert reward < info["net_log_return"]


def test_risk_halt_penalty_reduces_reward() -> None:
    prices = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [70.0, 100.0],
        ],
        dtype=np.float32,
    )
    env = make_env(
        prices,
        reward_mode="net_log_return_alpha",
        risk_halt_penalty_weight=0.5,
        stop_loss_pct=0.50,
    )
    env.reset()
    env.cash = 0.0
    env.holdings[0] = 10.0
    env.entry_prices[0] = 100.0
    env.portfolio_value = 1_000.0
    env.peak_portfolio_value = 1_000.0

    _, reward, _, truncated, info = env.step(np.zeros(2, dtype=np.float32))

    assert truncated
    assert info["risk_halted"]
    assert reward <= info["net_log_return"] - 0.5


def test_flat_market_hold_reward_is_near_zero() -> None:
    prices = np.ones((8, 2), dtype=np.float32) * 100.0
    env = make_env(prices, reward_mode="net_log_return_alpha")
    env.reset()

    _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert info["total_fees"] == 0.0
    assert abs(reward) < 1e-12


def test_uptrend_allocation_gets_positive_net_reward() -> None:
    prices = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [102.0, 100.0],
            [104.0, 100.0],
        ],
        dtype=np.float32,
    )
    env = make_env(
        prices,
        reward_mode="net_log_return_alpha",
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
        turnover_penalty_weight=0.0,
        drawdown_penalty_weight=0.0,
        alpha_reward_weight=0.0,
    )
    env.reset()
    env.step(np.array([0.5, 0.0], dtype=np.float32))

    _, reward, _, _, info = env.step(np.array([0.5, 0.0], dtype=np.float32))

    assert info["net_log_return"] > 0.0
    assert reward > 0.0


if __name__ == "__main__":
    test_stop_loss_liquidates_position()
    test_drawdown_halt_liquidates_and_truncates()
    test_position_cap_trims_oversized_holding()
    test_volatility_reward_is_bounded()
    test_positive_actions_buy_high_and_low_priced_assets()
    test_target_allocation_rebalances_once_then_holds()
    test_cash_target_allocation_can_choose_cash()
    test_cash_target_allocation_splits_scores_with_cash()
    test_observation_contains_cash_drawdown_return_and_weights()
    test_cooldown_expires_after_sell()
    test_random_start_uses_fixed_episode_window()
    test_deadband_and_min_notional_skip_tiny_trades()
    test_turnover_penalty_is_reported_and_reduces_reward()
    test_cost_penalty_reduces_net_log_alpha_reward()
    test_risk_halt_penalty_reduces_reward()
    test_flat_market_hold_reward_is_near_zero()
    test_uptrend_allocation_gets_positive_net_reward()
    print("multi_crypto_env risk/reward checks passed")
