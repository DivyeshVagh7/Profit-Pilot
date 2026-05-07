from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from profit_pilot.config import load_project_config
from profit_pilot.env.multi_crypto_env import MultiCryptoTradingEnv
from profit_pilot.train.train_ppo import model_name_for_config
from profit_pilot.utils.io import load_processed_bundle, save_json, slice_bundle_for_evaluation_window, symbol_slug


def compute_max_drawdown(values: list[float]) -> float:
    peaks = np.maximum.accumulate(np.asarray(values, dtype=np.float64))
    drawdowns = (peaks - np.asarray(values, dtype=np.float64)) / np.maximum(peaks, 1e-8)
    return float(np.max(drawdowns))


def summarize_curve(values: list[float], returns: list[float]) -> dict:
    return_array = np.asarray(returns, dtype=np.float64)
    volatility = float(return_array.std(ddof=0))
    sharpe = float(return_array.mean() / volatility) if volatility > 1e-12 else 0.0
    return {
        "initial_value": float(values[0]),
        "final_value": float(values[-1]),
        "return_pct": ((float(values[-1]) / max(float(values[0]), 1e-8)) - 1.0) * 100,
        "max_drawdown_pct": compute_max_drawdown(values) * 100,
        "mean_step_return": float(return_array.mean()) if return_array.size else 0.0,
        "step_return_volatility": volatility,
        "sharpe_per_step": sharpe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO or Recurrent PPO model.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to the project config YAML file.")
    parser.add_argument("--use-lstm", action="store_true", help="Load the recurrent PPO model instead of the feed-forward one.")
    parser.add_argument("--model-name", default=None, help="Optional model artifact name without .zip.")
    args = parser.parse_args()

    config = load_project_config(args.config)
    bundle = load_processed_bundle(
        processed_root=config.data.processed_dir,
        timeframe=config.data.timeframe,
        symbols=config.data.symbols,
    )
    if config.data.evaluation_start:
        bundle = slice_bundle_for_evaluation_window(
            bundle=bundle,
            evaluation_start=config.data.evaluation_start,
            lookback=config.environment.lookback,
            evaluation_end=config.data.evaluation_end,
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
        action_mode=config.environment.action_mode,
        cost_penalty_weight=config.environment.cost_penalty_weight,
        risk_penalty_weight=config.environment.risk_penalty_weight,
        alpha_reward_weight=config.environment.alpha_reward_weight,
        drawdown_penalty_weight=config.environment.drawdown_penalty_weight,
        risk_halt_penalty_weight=config.environment.risk_halt_penalty_weight,
        volatility_reward_weight=config.environment.volatility_reward_weight,
        volatility_window=config.environment.volatility_window,
        risk_adjusted_return_clip=config.environment.risk_adjusted_return_clip,
        max_position_fraction=config.environment.max_position_fraction,
        max_gross_exposure=config.environment.max_gross_exposure,
        max_trade_fraction=config.environment.max_trade_fraction,
        stop_loss_pct=config.environment.stop_loss_pct,
        max_drawdown_pct=config.environment.max_drawdown_pct,
        min_trade_quantity=config.environment.min_trade_quantity,
        cooldown_steps=config.environment.cooldown_steps,
        random_start=False,
        episode_length=None,
        trade_deadband=config.environment.trade_deadband,
        rebalance_threshold=config.environment.rebalance_threshold,
        min_trade_notional=config.environment.min_trade_notional,
        turnover_penalty_weight=config.environment.turnover_penalty_weight,
    )

    use_lstm = args.use_lstm or config.training.use_lstm
    model_name = args.model_name or model_name_for_config(config, use_lstm)
    model_path = Path(config.training.model_dir) / model_name
    report_dir = Path(config.training.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    model = RecurrentPPO.load(model_path) if use_lstm else PPO.load(model_path)

    timestamps = pd.to_datetime(bundle["timestamps"]["datetime"], utc=True)
    observation, info = env.reset()
    done = False
    episode_rewards: list[float] = []
    portfolio_values: list[float] = [env.portfolio_value]
    benchmark_values: list[float] = [float(info["benchmark_value"])]
    portfolio_returns: list[float] = []
    benchmark_returns: list[float] = []
    rows = [
        {
            "step": 0,
            "datetime": timestamps.iloc[env.time],
            "portfolio_value": env.portfolio_value,
            "benchmark_value": float(info["benchmark_value"]),
            "cash": float(env.cash),
            "reward": 0.0,
            "portfolio_return": 0.0,
            "benchmark_return": 0.0,
            "total_fees": 0.0,
            "turnover_penalty": 0.0,
            "drawdown": 0.0,
            "risk_halted": False,
        }
    ]
    lstm_state = None
    episode_start = np.array([True])
    step = 0

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
        step += 1
        episode_rewards.append(float(reward))
        portfolio_values.append(float(info["portfolio_value"]))
        benchmark_values.append(float(info["benchmark_value"]))
        portfolio_return = float(info["portfolio_return"])
        benchmark_return = (benchmark_values[-1] / max(benchmark_values[-2], 1e-8)) - 1.0
        portfolio_returns.append(portfolio_return)
        benchmark_returns.append(benchmark_return)

        row = {
            "step": step,
            "datetime": timestamps.iloc[env.time],
            "portfolio_value": float(info["portfolio_value"]),
            "benchmark_value": float(info["benchmark_value"]),
            "cash": float(info["cash"]),
            "reward": float(reward),
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "total_fees": float(info["total_fees"]),
            "turnover_penalty": float(info["turnover_penalty"]),
            "drawdown": float(info["drawdown"]),
            "risk_halted": bool(info["risk_halted"]),
        }
        executed_actions = np.asarray(info.get("executed_actions", action)).reshape(-1)
        action_array = np.asarray(action).reshape(-1)
        asset_action_offset = 1 if config.environment.action_mode == "cash_target_allocation" else 0
        if asset_action_offset:
            row["action_cash"] = float(action_array[0])
            row["target_cash_weight"] = float(info.get("target_cash_weight", 0.0))
        for index, symbol in enumerate(config.data.symbols):
            slug = symbol_slug(symbol)
            price = float(env.price_array[env.time, index])
            holding = float(env.holdings[index])
            row[f"price_{slug}"] = price
            row[f"holding_{slug}"] = holding
            row[f"position_value_{slug}"] = holding * price
            row[f"action_{slug}"] = float(action_array[index + asset_action_offset])
            row[f"target_weight_{slug}"] = float(info.get("target_weights", np.zeros(len(config.data.symbols)))[index])
            row[f"executed_action_{slug}"] = float(executed_actions[index])
        rows.append(row)
        episode_start = np.array([done])

    results = pd.DataFrame(rows)
    position_cols = [f"position_value_{symbol_slug(symbol)}" for symbol in config.data.symbols]
    holding_cols = [f"holding_{symbol_slug(symbol)}" for symbol in config.data.symbols]
    results["gross_exposure"] = results[position_cols].sum(axis=1) / results["portfolio_value"].clip(lower=1e-12)
    for symbol in config.data.symbols:
        slug = symbol_slug(symbol)
        results[f"exposure_{slug}"] = results[f"position_value_{slug}"] / results["portfolio_value"].clip(lower=1e-12)

    trade_events = int((results[holding_cols].diff().abs().sum(axis=1) > 1e-8).sum())
    active_assets = int(sum((results[column].abs() > 1e-12).any() for column in holding_cols))

    agent_metrics = summarize_curve(portfolio_values, portfolio_returns)
    agent_metrics.update(
        {
            "strategy": "agent",
            "total_reward": float(np.sum(episode_rewards)),
            "num_steps": len(episode_rewards),
            "total_fees": float(results["total_fees"].sum()),
            "trade_events": trade_events,
            "active_assets_traded": active_assets,
            "mean_gross_exposure_pct": float(results["gross_exposure"].mean() * 100),
            "risk_halted": bool(results["risk_halted"].any()),
        }
    )
    benchmark_metrics = summarize_curve(benchmark_values, benchmark_returns)
    benchmark_metrics.update(
        {
            "strategy": "equal_weight_benchmark",
            "total_reward": 0.0,
            "num_steps": len(episode_rewards),
            "total_fees": 0.0,
            "trade_events": 0,
            "active_assets_traded": len(config.data.symbols),
            "mean_gross_exposure_pct": 100.0,
            "risk_halted": False,
        }
    )
    agent_metrics["alpha_vs_benchmark_pct"] = agent_metrics["return_pct"] - benchmark_metrics["return_pct"]
    benchmark_metrics["alpha_vs_benchmark_pct"] = 0.0

    metrics = {
        "model_name": model_name,
        "model_path": str(model_path) + ".zip",
        "action_semantics": config.environment.action_mode,
        "metrics": [agent_metrics, benchmark_metrics],
    }
    save_json(metrics, report_dir / f"{model_name}_metrics.json")
    results.to_csv(report_dir / f"{model_name}_results.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label="Agent Portfolio")
    plt.plot(benchmark_values, label="Equal-Weight Benchmark")
    plt.title("Profit-Pilot Portfolio Curve")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / f"{model_name}_portfolio_curve.png", dpi=200)
    plt.close()

    print(f"Saved evaluation metrics to {report_dir / f'{model_name}_metrics.json'}")
    print(f"Saved evaluation results to {report_dir / f'{model_name}_results.csv'}")
    print(f"Saved portfolio curve to {report_dir / f'{model_name}_portfolio_curve.png'}")


if __name__ == "__main__":
    main()
