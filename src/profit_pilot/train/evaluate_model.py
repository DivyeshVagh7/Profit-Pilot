from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from profit_pilot.config import load_project_config
from profit_pilot.env.multi_crypto_env import MultiCryptoTradingEnv
from profit_pilot.utils.io import load_processed_bundle, save_json


def compute_max_drawdown(values: list[float]) -> float:
    peaks = np.maximum.accumulate(np.asarray(values, dtype=np.float64))
    drawdowns = (peaks - np.asarray(values, dtype=np.float64)) / np.maximum(peaks, 1e-8)
    return float(np.max(drawdowns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO or Recurrent PPO model.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to the project config YAML file.")
    parser.add_argument("--use-lstm", action="store_true", help="Load the recurrent PPO model instead of the feed-forward one.")
    args = parser.parse_args()

    config = load_project_config(args.config)
    bundle = load_processed_bundle(
        processed_root=config.data.processed_dir,
        timeframe=config.data.timeframe,
        symbols=config.data.symbols,
    )

    env = MultiCryptoTradingEnv(
        price_array=bundle["price_array"],
        tech_array=bundle["tech_array"],
        tickers=config.data.symbols,
        lookback=config.environment.lookback,
        initial_cash=config.environment.initial_cash,
        buy_cost_pct=config.environment.buy_cost_pct,
        sell_cost_pct=config.environment.sell_cost_pct,
        cash_norm=config.environment.cash_norm,
        holdings_norm=config.environment.holdings_norm,
        tech_norm=config.environment.tech_norm,
        reward_scaling=config.environment.reward_scaling,
        reward_mode=config.environment.reward_mode,
        cost_penalty_weight=config.environment.cost_penalty_weight,
        risk_penalty_weight=config.environment.risk_penalty_weight,
        min_trade_quantity=config.environment.min_trade_quantity,
        cooldown_steps=config.environment.cooldown_steps,
    )

    use_lstm = args.use_lstm or config.training.use_lstm
    model_name = "profit_pilot_recurrent_ppo" if use_lstm else "profit_pilot_ppo"
    model_path = Path(config.training.model_dir) / model_name
    report_dir = Path(config.training.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    model = RecurrentPPO.load(model_path) if use_lstm else PPO.load(model_path)

    observation, _ = env.reset()
    done = False
    episode_rewards: list[float] = []
    portfolio_values: list[float] = [env.portfolio_value]
    lstm_state = None
    episode_start = np.array([True])

    while not done:
        if use_lstm:
            action, lstm_state = model.predict(
                observation,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True,
            )
        else:
            action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_rewards.append(float(reward))
        portfolio_values.append(float(info["portfolio_value"]))
        episode_start = np.array([done])

    metrics = {
        "model_name": model_name,
        "final_portfolio_value": portfolio_values[-1],
        "return_pct": ((portfolio_values[-1] / portfolio_values[0]) - 1.0) * 100,
        "total_reward": float(np.sum(episode_rewards)),
        "max_drawdown": compute_max_drawdown(portfolio_values),
        "num_steps": len(episode_rewards),
    }
    save_json(metrics, report_dir / f"{model_name}_metrics.json")

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label="Agent Portfolio")
    plt.title("Profit-Pilot Portfolio Curve")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / f"{model_name}_portfolio_curve.png", dpi=200)
    plt.close()

    print(f"Saved evaluation metrics to {report_dir / f'{model_name}_metrics.json'}")
    print(f"Saved portfolio curve to {report_dir / f'{model_name}_portfolio_curve.png'}")


if __name__ == "__main__":
    main()
