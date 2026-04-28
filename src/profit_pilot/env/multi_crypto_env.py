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
        volatility_reward_weight: float = 0.01,
        volatility_window: int = 50,
        risk_adjusted_return_clip: float = 5.0,
        max_position_fraction: float = 0.50,
        max_trade_fraction: float = 0.25,
        stop_loss_pct: float = 0.08,
        max_drawdown_pct: float = 0.20,
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
        self.volatility_reward_weight = float(volatility_reward_weight)
        self.volatility_window = int(volatility_window)
        self.risk_adjusted_return_clip = float(risk_adjusted_return_clip)
        self.max_position_fraction = float(max_position_fraction)
        self.max_trade_fraction = float(max_trade_fraction)
        self.stop_loss_pct = float(stop_loss_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.min_trade_quantity = float(min_trade_quantity)
        self.cooldown_steps = int(cooldown_steps)

        self._validate_settings()

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
        self.entry_prices = np.zeros(self.n_assets, dtype=np.float32)
        self.cooldowns = np.zeros(self.n_assets, dtype=np.int32)
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.risk_halted = False
        self.equal_weight_holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.equal_weight_value = self.initial_cash

    def _validate_settings(self) -> None:
        supported_reward_modes = {"return_cost_risk", "return_cost_risk_volatility"}
        if self.reward_mode not in supported_reward_modes:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")
        if self.risk_adjusted_return_clip <= 0:
            raise ValueError("risk_adjusted_return_clip must be greater than 0")
        if not 0.0 < self.max_position_fraction <= 1.0:
            raise ValueError("max_position_fraction must be in the range (0, 1]")
        if not 0.0 < self.max_trade_fraction <= 1.0:
            raise ValueError("max_trade_fraction must be in the range (0, 1]")
        if not 0.0 < self.stop_loss_pct < 1.0:
            raise ValueError("stop_loss_pct must be in the range (0, 1)")
        if not 0.0 < self.max_drawdown_pct < 1.0:
            raise ValueError("max_drawdown_pct must be in the range (0, 1)")

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

    def _rolling_return_volatility(self) -> float:
        start = max(1, self.time - self.volatility_window + 1)
        price_window = self.price_array[start - 1 : self.time + 1]
        if price_window.shape[0] < 2:
            return 1e-4

        returns = np.diff(price_window, axis=0) / np.maximum(price_window[:-1], 1e-8)
        if returns.size == 0:
            return 1e-4
        return float(max(np.std(returns, axis=0).mean(), 1e-4))

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
            if self.holdings[index] <= self.min_trade_quantity:
                self.holdings[index] = 0.0
                self.entry_prices[index] = 0.0
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
            portfolio_value = self._portfolio_value(prices)
            max_position_value = portfolio_value * self.max_position_fraction
            current_position_value = float(self.holdings[index] * prices[index])
            remaining_position_value = max(0.0, max_position_value - current_position_value)
            max_trade_value = portfolio_value * self.max_trade_fraction
            max_affordable_value = fee_adjusted_cash * 0.95
            buy_value_cap = min(remaining_position_value, max_trade_value, max_affordable_value)
            max_affordable = buy_value_cap / max(float(prices[index]), 1e-8)
            buy_amount = min(max_affordable, float(actions[index]))
            if buy_amount < self.min_trade_quantity:
                continue

            gross_value = float(prices[index] * buy_amount)
            fee_paid = gross_value * self.buy_cost_pct
            previous_quantity = float(self.holdings[index])
            previous_entry_value = previous_quantity * float(self.entry_prices[index])
            self.holdings[index] += buy_amount
            self.entry_prices[index] = (previous_entry_value + gross_value) / max(float(self.holdings[index]), 1e-8)
            self.cash -= gross_value + fee_paid
            total_buy_fees += fee_paid
        return total_buy_fees

    def _liquidate_position(self, index: int, prices: np.ndarray) -> float:
        if self.holdings[index] <= 0:
            return 0.0

        gross_value = float(prices[index] * self.holdings[index])
        fee_paid = gross_value * self.sell_cost_pct
        self.cash += gross_value - fee_paid
        self.holdings[index] = 0.0
        self.entry_prices[index] = 0.0
        self.cooldowns[index] = 0
        return fee_paid

    def _apply_stop_losses(self, prices: np.ndarray) -> tuple[float, list[int], list[str]]:
        total_stop_fees = 0.0
        triggered_indices = []
        triggered_symbols = []
        for index, ticker in enumerate(self.tickers):
            if self.holdings[index] <= 0 or self.entry_prices[index] <= 0:
                continue
            stop_price = float(self.entry_prices[index]) * (1 - self.stop_loss_pct)
            if prices[index] <= stop_price:
                total_stop_fees += self._liquidate_position(index, prices)
                triggered_indices.append(index)
                triggered_symbols.append(ticker)
        return total_stop_fees, triggered_indices, triggered_symbols

    def _trim_position_caps(self, prices: np.ndarray) -> tuple[float, list[str]]:
        total_cap_fees = 0.0
        trimmed_symbols = []
        portfolio_value = self._portfolio_value(prices)
        max_position_value = portfolio_value * self.max_position_fraction

        for index, ticker in enumerate(self.tickers):
            position_value = float(self.holdings[index] * prices[index])
            excess_value = position_value - max_position_value
            if excess_value <= 0:
                continue

            sell_amount = min(float(self.holdings[index]), excess_value / max(float(prices[index]), 1e-8))
            if sell_amount < self.min_trade_quantity:
                continue

            gross_value = float(prices[index] * sell_amount)
            fee_paid = gross_value * self.sell_cost_pct
            self.holdings[index] -= sell_amount
            self.cash += gross_value - fee_paid
            if self.holdings[index] <= self.min_trade_quantity:
                self.holdings[index] = 0.0
                self.entry_prices[index] = 0.0
            total_cap_fees += fee_paid
            trimmed_symbols.append(ticker)

        return total_cap_fees, trimmed_symbols

    def _liquidate_all_positions(self, prices: np.ndarray) -> float:
        total_fees = 0.0
        for index in range(self.n_assets):
            total_fees += self._liquidate_position(index, prices)
        return total_fees

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.time = self.lookback - 1
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.entry_prices = np.zeros(self.n_assets, dtype=np.float32)
        self.cooldowns = np.full(self.n_assets, self.cooldown_steps, dtype=np.int32)
        self.risk_halted = False

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

        stop_loss_fees, stop_loss_indices, stop_loss_symbols = self._apply_stop_losses(prices)
        position_cap_fees, position_cap_symbols = self._trim_position_caps(prices)
        pre_trade_value = self._portfolio_value(prices)
        self.peak_portfolio_value = max(self.peak_portfolio_value, pre_trade_value)
        pre_trade_drawdown = (
            (self.peak_portfolio_value - pre_trade_value) / max(self.peak_portfolio_value, 1e-8)
        )

        halt_fees = 0.0
        if pre_trade_drawdown >= self.max_drawdown_pct:
            halt_fees = self._liquidate_all_positions(prices)
            self.risk_halted = True

        sell_fees = 0.0
        buy_fees = 0.0
        if not self.risk_halted:
            scaled_actions = np.clip(action, -1.0, 1.0).astype(np.float32) * self.action_norm_vector
            if stop_loss_indices:
                scaled_actions[stop_loss_indices] = np.minimum(scaled_actions[stop_loss_indices], 0.0)
            sell_fees = self._apply_sell_actions(scaled_actions, prices)
            buy_fees = self._apply_buy_actions(scaled_actions, prices)
        total_fees = stop_loss_fees + position_cap_fees + halt_fees + sell_fees + buy_fees

        self.portfolio_value = self._portfolio_value(prices)
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        self.equal_weight_value = self._benchmark_value(prices)

        delta_portfolio = self.portfolio_value - previous_portfolio_value
        portfolio_return = delta_portfolio / max(previous_portfolio_value, 1e-8)
        cost_penalty = total_fees / max(previous_portfolio_value, 1e-8)
        drawdown = (self.peak_portfolio_value - self.portfolio_value) / max(self.peak_portfolio_value, 1e-8)
        rolling_volatility = self._rolling_return_volatility()
        raw_risk_adjusted_return = portfolio_return / rolling_volatility
        clipped_risk_adjusted_return = float(
            np.clip(raw_risk_adjusted_return, -self.risk_adjusted_return_clip, self.risk_adjusted_return_clip)
        )
        risk_adjusted_return = float(np.tanh(clipped_risk_adjusted_return))

        reward = portfolio_return
        if self.reward_mode == "return_cost_risk_volatility":
            reward += self.volatility_reward_weight * risk_adjusted_return
        reward -= self.cost_penalty_weight * cost_penalty
        reward -= self.risk_penalty_weight * drawdown
        reward *= self.reward_scaling

        terminated = self.time >= self.max_step
        truncated = self.risk_halted
        state = self._get_state()
        info = {
            "portfolio_value": self.portfolio_value,
            "benchmark_value": self.equal_weight_value,
            "cash": float(self.cash),
            "holdings": self.holdings.copy(),
            "entry_prices": self.entry_prices.copy(),
            "portfolio_return": float(portfolio_return),
            "risk_adjusted_return": float(risk_adjusted_return),
            "clipped_risk_adjusted_return": float(clipped_risk_adjusted_return),
            "raw_risk_adjusted_return": float(raw_risk_adjusted_return),
            "rolling_volatility": float(rolling_volatility),
            "cost_penalty": float(cost_penalty),
            "drawdown": float(drawdown),
            "total_fees": float(total_fees),
            "stop_loss_symbols": stop_loss_symbols,
            "position_cap_symbols": position_cap_symbols,
            "risk_halted": self.risk_halted,
        }
        return state, float(reward), terminated, truncated, info
