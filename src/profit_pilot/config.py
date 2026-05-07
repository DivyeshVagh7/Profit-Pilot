from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import yaml
from dotenv import load_dotenv


@dataclass
class ProjectSection:
    name: str
    exchange: str


@dataclass
class DataSection:
    raw_dir: str
    processed_dir: str
    symbols: list[str]
    timeframe: str
    since: str
    until: str
    download_limit: int
    normalization_stats_dir: str | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None


@dataclass
class EnvironmentSection:
    lookback: int
    initial_cash: float
    buy_cost_pct: float
    sell_cost_pct: float
    cash_norm: float
    holdings_norm: float
    tech_norm: float
    reward_scaling: float
    reward_mode: str
    cost_penalty_weight: float
    risk_penalty_weight: float
    volatility_reward_weight: float
    volatility_window: int
    risk_adjusted_return_clip: float
    max_position_fraction: float
    max_trade_fraction: float
    stop_loss_pct: float
    max_drawdown_pct: float
    min_trade_quantity: float
    cooldown_steps: int
    random_start: bool = False
    episode_length: int | None = None
    trade_deadband: float = 0.0
    min_trade_notional: float = 0.0
    turnover_penalty_weight: float = 0.0
    action_mode: str = "target_allocation"
    alpha_reward_weight: float = 0.25
    drawdown_penalty_weight: float = 0.05
    risk_halt_penalty_weight: float = 0.0
    max_gross_exposure: float = 0.80
    rebalance_threshold: float = 0.03


@dataclass
class TrainingSection:
    total_timesteps: int
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    device: str
    use_lstm: bool
    model_dir: str
    report_dir: str
    target_kl: float | None = None


@dataclass
class ProjectConfig:
    project: ProjectSection
    data: DataSection
    environment: EnvironmentSection
    training: TrainingSection


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_project_config(config_path: str | Path) -> ProjectConfig:
    load_dotenv()
    path = Path(config_path)
    raw = _read_yaml(path)

    project = ProjectSection(**raw["project"])
    data = DataSection(**raw["data"])
    environment = EnvironmentSection(**raw["environment"])
    training = TrainingSection(**raw["training"])

    return ProjectConfig(
        project=project,
        data=data,
        environment=environment,
        training=training,
    )


def get_binance_credentials() -> tuple[str | None, str | None]:
    api_key = os.getenv("BINANCE_API_KEY") or None
    api_secret = os.getenv("BINANCE_API_SECRET") or None
    return api_key, api_secret


def get_exchange_name(default_name: str) -> str:
    return os.getenv("BINANCE_EXCHANGE", default_name)
