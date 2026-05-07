from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from profit_pilot.config import load_project_config
from profit_pilot.env.multi_crypto_env import MultiCryptoTradingEnv
from profit_pilot.utils.io import bundle_name, load_processed_bundle, save_json


def model_name_for_config(config, use_lstm: bool) -> str:
    algorithm = "recurrent_ppo" if use_lstm else "ppo"
    action_suffix = "_cash_allocation" if config.environment.action_mode == "cash_target_allocation" else ""
    dataset_name = bundle_name(config.data.timeframe, config.data.symbols)
    return f"profit_pilot_{algorithm}{action_suffix}_multi_asset_{dataset_name}"


def build_env(config_path: str):
    config = load_project_config(config_path)
    bundle = load_processed_bundle(
        processed_root=config.data.processed_dir,
        timeframe=config.data.timeframe,
        symbols=config.data.symbols,
    )

    def _make_env():
        return MultiCryptoTradingEnv(
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
            random_start=config.environment.random_start,
            episode_length=config.environment.episode_length,
            trade_deadband=config.environment.trade_deadband,
            rebalance_threshold=config.environment.rebalance_threshold,
            min_trade_notional=config.environment.min_trade_notional,
            turnover_penalty_weight=config.environment.turnover_penalty_weight,
        )

    return config, bundle, _make_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO or Recurrent PPO on the processed Profit-Pilot dataset.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to the project config YAML file.")
    parser.add_argument("--use-lstm", action="store_true", help="Override the config and use Recurrent PPO.")
    parser.add_argument("--model-name", default=None, help="Optional model artifact name without .zip.")
    args = parser.parse_args()

    config, bundle, env_factory = build_env(args.config)
    use_lstm = args.use_lstm or config.training.use_lstm

    vec_env = DummyVecEnv([env_factory])
    vec_env = VecMonitor(vec_env)

    model_dir = Path(config.training.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(
        env=vec_env,
        verbose=1,
        learning_rate=config.training.learning_rate,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_range=config.training.clip_range,
        ent_coef=config.training.ent_coef,
        device=config.training.device,
    )
    if config.training.target_kl is not None:
        common_kwargs["target_kl"] = config.training.target_kl

    if use_lstm:
        model = RecurrentPPO("MlpLstmPolicy", **common_kwargs)
    else:
        model = PPO("MlpPolicy", **common_kwargs)
    model_name = args.model_name or model_name_for_config(config, use_lstm)

    model.learn(total_timesteps=config.training.total_timesteps)
    model_path = model_dir / model_name
    model.save(model_path)

    summary = {
        "model_path": str(model_path),
        "model_name": model_name,
        "use_lstm": use_lstm,
        "total_timesteps": config.training.total_timesteps,
        "symbols": config.data.symbols,
        "timeframe": config.data.timeframe,
        "price_array_shape": list(bundle["price_array"].shape),
        "tech_array_shape": list(bundle["tech_array"].shape),
        "action_semantics": config.environment.action_mode,
        "environment": {
            "random_start": config.environment.random_start,
            "episode_length": config.environment.episode_length,
            "action_mode": config.environment.action_mode,
            "reward_mode": config.environment.reward_mode,
            "max_gross_exposure": config.environment.max_gross_exposure,
            "max_trade_fraction": config.environment.max_trade_fraction,
            "trade_deadband": config.environment.trade_deadband,
            "rebalance_threshold": config.environment.rebalance_threshold,
            "min_trade_notional": config.environment.min_trade_notional,
            "alpha_reward_weight": config.environment.alpha_reward_weight,
            "drawdown_penalty_weight": config.environment.drawdown_penalty_weight,
            "risk_halt_penalty_weight": config.environment.risk_halt_penalty_weight,
            "turnover_penalty_weight": config.environment.turnover_penalty_weight,
        },
    }
    save_json(summary, model_dir / f"{model_name}_summary.json")
    print(f"Saved model to {model_path}.zip")


if __name__ == "__main__":
    main()
