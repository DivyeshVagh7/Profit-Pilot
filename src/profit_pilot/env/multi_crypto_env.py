from __future__ import annotations

import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MultiCryptoTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        price_array: np.ndarray,
        tech_array: np.ndarray,
        tickers: list[str],
        lookback: int = 50,
        initial_cash: float = 10_000.0,
        buy_cost_pct: float = 0.001,
        sell_cost_pct: float = 0.001,
        cash_norm: float = 1e-4,
        holdings_norm: float = 1.0,
        tech_norm: float = 1.0,
        reward_scaling: float = 1.0,
        reward_mode: str = "return_cost_risk",
        cost_penalty_weight: float = 1.0,
        risk_penalty_weight: float = 0.05,
        min_trade_quantity: float = 1e-4,
        cooldown_steps: int = 0,
    ) -> None:
        super().__init__()

        self.price_array = np.asarray(price_array, dtype=np.float32)
        self.tech_array = np.asarray(tech_array, dtype=np.float32)
        self.tickers = tickers
        self.lookback = lookback
        self.initial_cash = float(initial_cash)
        self.buy_cost_pct = float(buy_cost_pct)
        self.sell_cost_pct = float(sell_cost_pct)
        self.cash_norm = float(cash_norm)
        self.holdings_norm = float(holdings_norm)
        self.tech_norm = float(tech_norm)
        self.reward_scaling = float(reward_scaling)
        self.reward_mode = reward_mode
        self.cost_penalty_weight = float(cost_penalty_weight)
        self.risk_penalty_weight = float(risk_penalty_weight)
        self.min_trade_quantity = float(min_trade_quantity)
        self.cooldown_steps = int(cooldown_steps)

        self.n_assets = self.price_array.shape[1]
        self.n_tech_features = self.tech_array.shape[1]
        self.state_dim = 1 + self.n_assets + self.n_tech_features * self.lookback
        self.max_step = self.price_array.shape[0] - 1

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.action_norm_vector = self._build_action_norm_vector()

        self.time = self.lookback - 1
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.cooldowns = np.zeros(self.n_assets, dtype=np.int32)
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.equal_weight_holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.equal_weight_value = self.initial_cash

    def _build_action_norm_vector(self) -> np.ndarray:
        action_norm = []
        for price in self.price_array[0]:
            order = max(0, math.floor(math.log(max(price, 1e-8), 10)))
            action_norm.append(1 / (10**order))
        return np.asarray(action_norm, dtype=np.float32)

    def _portfolio_value(self, prices: np.ndarray) -> float:
        return float(self.cash + (self.holdings * prices).sum())

    def _benchmark_value(self, prices: np.ndarray) -> float:
        return float((self.equal_weight_holdings * prices).sum())

    def _get_state(self) -> np.ndarray:
        state = [self.cash * self.cash_norm]
        state.extend((self.holdings * self.holdings_norm).tolist())

        for offset in range(self.lookback):
            index = max(0, self.time - offset)
            tech_row = self.tech_array[index] * self.tech_norm
            state.extend(tech_row.tolist())

        return np.asarray(state, dtype=np.float32)

    def _apply_sell_actions(self, actions: np.ndarray, prices: np.ndarray) -> float:
        total_sell_fees = 0.0
        sell_indices = np.where(actions < -self.min_trade_quantity)[0]
        for index in sell_indices:
            if self.holdings[index] <= 0:
                continue
            sell_amount = min(self.holdings[index], float(-actions[index]))
            gross_value = float(prices[index] * sell_amount)
            fee_paid = gross_value * self.sell_cost_pct
            self.holdings[index] -= sell_amount
            self.cash += gross_value - fee_paid
            self.cooldowns[index] = 0
            total_sell_fees += fee_paid
        return total_sell_fees

    def _apply_buy_actions(self, actions: np.ndarray, prices: np.ndarray) -> float:
        total_buy_fees = 0.0
        buy_indices = np.where(actions > self.min_trade_quantity)[0]
        for index in buy_indices:
            if self.cooldown_steps and self.cooldowns[index] < self.cooldown_steps:
                continue

            fee_adjusted_cash = self.cash / (1 + self.buy_cost_pct)
            max_affordable = (fee_adjusted_cash / prices[index]) * 0.95
            buy_amount = min(max_affordable, float(actions[index]))
            if buy_amount < self.min_trade_quantity:
                continue

            gross_value = float(prices[index] * buy_amount)
            fee_paid = gross_value * self.buy_cost_pct
            self.holdings[index] += buy_amount
            self.cash -= gross_value + fee_paid
            total_buy_fees += fee_paid
        return total_buy_fees

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.time = self.lookback - 1
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.cooldowns = np.full(self.n_assets, self.cooldown_steps, dtype=np.int32)

        initial_prices = self.price_array[self.time]
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.equal_weight_holdings = (
            np.ones(self.n_assets, dtype=np.float32)
            * (self.initial_cash / self.n_assets)
            / initial_prices
        )
        self.equal_weight_value = self._benchmark_value(initial_prices)

        state = self._get_state()
        info = {"portfolio_value": self.portfolio_value, "benchmark_value": self.equal_weight_value}
        return state, info

    def step(self, action: np.ndarray):
        if self.time >= self.max_step:
            raise RuntimeError("Episode is done. Call reset() before step().")

        previous_portfolio_value = self.portfolio_value
        self.time += 1
        prices = self.price_array[self.time]

        self.cooldowns = np.where(self.holdings > 0, self.cooldowns + 1, self.cooldowns)

        scaled_actions = np.clip(action, -1.0, 1.0).astype(np.float32) * self.action_norm_vector
        sell_fees = self._apply_sell_actions(scaled_actions, prices)
        buy_fees = self._apply_buy_actions(scaled_actions, prices)
        total_fees = sell_fees + buy_fees

        self.portfolio_value = self._portfolio_value(prices)
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        self.equal_weight_value = self._benchmark_value(prices)

        delta_portfolio = self.portfolio_value - previous_portfolio_value
        portfolio_return = delta_portfolio / max(previous_portfolio_value, 1e-8)
        cost_penalty = total_fees / max(previous_portfolio_value, 1e-8)
        drawdown = (self.peak_portfolio_value - self.portfolio_value) / max(self.peak_portfolio_value, 1e-8)

        reward = (
            portfolio_return
            - (self.cost_penalty_weight * cost_penalty)
            - (self.risk_penalty_weight * drawdown)
        ) * self.reward_scaling

        terminated = self.time >= self.max_step
        truncated = False
        state = self._get_state()
        info = {
            "portfolio_value": self.portfolio_value,
            "benchmark_value": self.equal_weight_value,
            "cash": float(self.cash),
            "holdings": self.holdings.copy(),
            "portfolio_return": float(portfolio_return),
            "cost_penalty": float(cost_penalty),
            "drawdown": float(drawdown),
            "total_fees": float(total_fees),
        }
        return state, float(reward), terminated, truncated, info
