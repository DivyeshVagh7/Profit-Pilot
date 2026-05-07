from __future__ import annotations

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
        reward_mode: str = "net_log_return_alpha",
        action_mode: str = "target_allocation",
        cost_penalty_weight: float = 1.0,
        risk_penalty_weight: float = 0.05,
        alpha_reward_weight: float = 0.25,
        drawdown_penalty_weight: float = 0.05,
        risk_halt_penalty_weight: float = 0.0,
        volatility_reward_weight: float = 0.01,
        volatility_window: int = 50,
        risk_adjusted_return_clip: float = 5.0,
        max_position_fraction: float = 0.60,
        max_gross_exposure: float = 0.80,
        max_trade_fraction: float = 1.0,
        stop_loss_pct: float = 0.08,
        max_drawdown_pct: float = 0.20,
        min_trade_quantity: float = 1e-4,
        cooldown_steps: int = 0,
        random_start: bool = False,
        episode_length: int | None = None,
        trade_deadband: float = 0.0,
        rebalance_threshold: float = 0.03,
        min_trade_notional: float = 0.0,
        turnover_penalty_weight: float = 0.002,
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
        self.action_mode = action_mode
        self.cost_penalty_weight = float(cost_penalty_weight)
        self.risk_penalty_weight = float(risk_penalty_weight)
        self.alpha_reward_weight = float(alpha_reward_weight)
        self.drawdown_penalty_weight = float(drawdown_penalty_weight)
        self.risk_halt_penalty_weight = float(risk_halt_penalty_weight)
        self.volatility_reward_weight = float(volatility_reward_weight)
        self.volatility_window = int(volatility_window)
        self.risk_adjusted_return_clip = float(risk_adjusted_return_clip)
        self.max_position_fraction = float(max_position_fraction)
        self.max_gross_exposure = float(max_gross_exposure)
        self.max_trade_fraction = float(max_trade_fraction)
        self.stop_loss_pct = float(stop_loss_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.min_trade_quantity = float(min_trade_quantity)
        self.cooldown_steps = int(cooldown_steps)
        self.random_start = bool(random_start)
        self.episode_length = None if episode_length is None else int(episode_length)
        self.trade_deadband = float(trade_deadband)
        self.rebalance_threshold = float(rebalance_threshold)
        self.min_trade_notional = float(min_trade_notional)
        self.turnover_penalty_weight = float(turnover_penalty_weight)

        self._validate_settings()

        self.n_assets = self.price_array.shape[1]
        self.n_tech_features = self.tech_array.shape[1]
        self.account_state_dim = 3 + self.n_assets
        self.state_dim = self.account_state_dim + self.n_tech_features * self.lookback
        self.max_step = self.price_array.shape[0] - 1
        self.episode_end_time = self.max_step

        allocation_modes = {"target_allocation", "cash_target_allocation"}
        action_low = 0.0 if self.action_mode in allocation_modes else -1.0
        action_dim = self.n_assets + 1 if self.action_mode == "cash_target_allocation" else self.n_assets
        self.action_space = spaces.Box(low=action_low, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.time = self.lookback - 1
        self.episode_start_time = self.time
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.entry_prices = np.zeros(self.n_assets, dtype=np.float32)
        self.cooldowns = np.zeros(self.n_assets, dtype=np.int32)
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.previous_portfolio_return = 0.0
        self.risk_halted = False
        self.equal_weight_holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.equal_weight_value = self.initial_cash

    def _validate_settings(self) -> None:
        supported_action_modes = {"target_allocation", "cash_target_allocation", "trade_fraction"}
        supported_reward_modes = {"return_cost_risk", "return_cost_risk_volatility", "net_log_return_alpha"}
        if self.price_array.ndim != 2:
            raise ValueError("price_array must be a 2D array with shape (steps, assets)")
        if self.tech_array.ndim != 2:
            raise ValueError("tech_array must be a 2D array with shape (steps, features)")
        if self.price_array.shape[0] != self.tech_array.shape[0]:
            raise ValueError("price_array and tech_array must have the same number of timesteps")
        if self.price_array.shape[1] != len(self.tickers):
            raise ValueError("tickers length must match the number of price_array assets")
        if self.lookback < 1 or self.lookback >= self.price_array.shape[0]:
            raise ValueError("lookback must be at least 1 and smaller than the number of timesteps")
        if not np.isfinite(self.price_array).all() or np.any(self.price_array <= 0):
            raise ValueError("price_array must contain only positive finite prices")
        if not np.isfinite(self.tech_array).all():
            raise ValueError("tech_array must contain only finite values")
        if self.reward_mode not in supported_reward_modes:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")
        if self.action_mode not in supported_action_modes:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")
        if self.risk_adjusted_return_clip <= 0:
            raise ValueError("risk_adjusted_return_clip must be greater than 0")
        if not 0.0 < self.max_position_fraction <= 1.0:
            raise ValueError("max_position_fraction must be in the range (0, 1]")
        if not 0.0 < self.max_gross_exposure <= 1.0:
            raise ValueError("max_gross_exposure must be in the range (0, 1]")
        if not 0.0 < self.max_trade_fraction <= 1.0:
            raise ValueError("max_trade_fraction must be in the range (0, 1]")
        if not 0.0 < self.stop_loss_pct < 1.0:
            raise ValueError("stop_loss_pct must be in the range (0, 1)")
        if not 0.0 < self.max_drawdown_pct < 1.0:
            raise ValueError("max_drawdown_pct must be in the range (0, 1)")
        if self.episode_length is not None:
            max_episode_length = self.price_array.shape[0] - self.lookback
            if self.episode_length < 1:
                raise ValueError("episode_length must be at least 1 when provided")
            if self.episode_length > max_episode_length:
                raise ValueError(
                    "episode_length is too large for the dataset and lookback: "
                    f"episode_length={self.episode_length}, max={max_episode_length}"
                )
        if self.random_start and self.episode_length is None:
            raise ValueError("random_start requires episode_length so training episodes have stable length")
        if not 0.0 <= self.trade_deadband < 1.0:
            raise ValueError("trade_deadband must be in the range [0, 1)")
        if not 0.0 <= self.rebalance_threshold < 1.0:
            raise ValueError("rebalance_threshold must be in the range [0, 1)")
        if self.min_trade_notional < 0.0:
            raise ValueError("min_trade_notional must be non-negative")
        if self.turnover_penalty_weight < 0.0:
            raise ValueError("turnover_penalty_weight must be non-negative")
        if self.alpha_reward_weight < 0.0:
            raise ValueError("alpha_reward_weight must be non-negative")
        if self.drawdown_penalty_weight < 0.0:
            raise ValueError("drawdown_penalty_weight must be non-negative")
        if self.risk_halt_penalty_weight < 0.0:
            raise ValueError("risk_halt_penalty_weight must be non-negative")

    def _portfolio_value(self, prices: np.ndarray) -> float:
        return float(self.cash + (self.holdings * prices).sum())

    def _benchmark_value(self, prices: np.ndarray) -> float:
        return float((self.equal_weight_holdings * prices).sum())

    def _asset_weights(self, prices: np.ndarray, portfolio_value: float | None = None) -> np.ndarray:
        value = self._portfolio_value(prices) if portfolio_value is None else portfolio_value
        return (self.holdings * prices / max(value, 1e-8)).astype(np.float32)

    def _cash_weight(self, portfolio_value: float | None = None) -> float:
        value = self.portfolio_value if portfolio_value is None else portfolio_value
        return float(self.cash / max(value, 1e-8))

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
        prices = self.price_array[self.time]
        portfolio_value = self._portfolio_value(prices)
        drawdown = (self.peak_portfolio_value - portfolio_value) / max(self.peak_portfolio_value, 1e-8)
        state = [
            self._cash_weight(portfolio_value),
            float(drawdown),
            float(self.previous_portfolio_return),
        ]
        state.extend(self._asset_weights(prices, portfolio_value).tolist())

        for offset in range(self.lookback):
            index = max(0, self.time - offset)
            tech_row = self.tech_array[index] * self.tech_norm
            state.extend(tech_row.tolist())

        return np.asarray(state, dtype=np.float32)

    def _target_weights_from_action(self, action: np.ndarray) -> np.ndarray:
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_mode == "cash_target_allocation":
            expected_shape = self.n_assets + 1
            if raw_action.shape[0] != expected_shape:
                raise ValueError(f"action must have shape ({expected_shape},) for cash_target_allocation")
            allocation_scores = np.clip(raw_action, 0.0, 1.0)
            total_score = float(allocation_scores.sum())
            if total_score <= 1e-8:
                target_weights = np.zeros(self.n_assets, dtype=np.float32)
            else:
                target_weights = allocation_scores[1:] / total_score
        else:
            target_weights = np.clip(raw_action, 0.0, 1.0)
            if target_weights.shape[0] != self.n_assets:
                raise ValueError(f"action must have shape ({self.n_assets},)")
        target_weights = np.minimum(target_weights, self.max_position_fraction)
        gross_target = float(target_weights.sum())
        if gross_target > self.max_gross_exposure:
            target_weights *= self.max_gross_exposure / max(gross_target, 1e-8)
        return target_weights.astype(np.float32)

    def _apply_sell_actions(self, actions: np.ndarray, prices: np.ndarray) -> float:
        total_sell_fees = 0.0
        sell_indices = np.where(actions < -1e-8)[0]
        for index in sell_indices:
            if self.holdings[index] <= 0:
                continue
            portfolio_value = self._portfolio_value(prices)
            requested_trade_value = float(-actions[index]) * self.max_trade_fraction * portfolio_value
            current_position_value = float(self.holdings[index] * prices[index])
            sell_value = min(current_position_value, requested_trade_value)
            if sell_value < self.min_trade_notional:
                continue
            sell_amount = sell_value / max(float(prices[index]), 1e-8)
            if sell_amount < self.min_trade_quantity:
                continue
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
        buy_indices = np.where(actions > 1e-8)[0]
        for index in buy_indices:
            if self.cooldown_steps and self.cooldowns[index] < self.cooldown_steps:
                continue

            portfolio_value = self._portfolio_value(prices)
            requested_trade_value = float(actions[index]) * self.max_trade_fraction * portfolio_value
            fee_adjusted_cash = self.cash / (1 + self.buy_cost_pct)
            max_position_value = portfolio_value * self.max_position_fraction
            current_position_value = float(self.holdings[index] * prices[index])
            remaining_position_value = max(0.0, max_position_value - current_position_value)
            buy_value = min(requested_trade_value, remaining_position_value, fee_adjusted_cash)
            if buy_value < self.min_trade_notional:
                continue
            buy_amount = buy_value / max(float(prices[index]), 1e-8)
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

    def _apply_target_allocation_actions(self, target_weights: np.ndarray, prices: np.ndarray) -> tuple[float, float, np.ndarray]:
        target_weights = np.asarray(target_weights, dtype=np.float32).reshape(-1)
        if target_weights.shape[0] != self.n_assets:
            raise ValueError(f"target_weights must have shape ({self.n_assets},)")
        portfolio_value = self._portfolio_value(prices)
        current_weights = self._asset_weights(prices, portfolio_value)
        weight_deltas = target_weights - current_weights
        weight_deltas[np.abs(weight_deltas) < self.rebalance_threshold] = 0.0

        total_fees = 0.0
        traded_value = 0.0
        max_step_trade_value = self.max_trade_fraction * portfolio_value

        for index in np.where(weight_deltas < -1e-8)[0]:
            current_position_value = float(self.holdings[index] * prices[index])
            desired_trade_value = min(float(-weight_deltas[index]) * portfolio_value, max_step_trade_value)
            sell_value = min(current_position_value, desired_trade_value)
            if sell_value < self.min_trade_notional:
                continue
            sell_amount = sell_value / max(float(prices[index]), 1e-8)
            if sell_amount < self.min_trade_quantity:
                continue
            gross_value = float(prices[index] * sell_amount)
            fee_paid = gross_value * self.sell_cost_pct
            self.holdings[index] -= sell_amount
            self.cash += gross_value - fee_paid
            if self.holdings[index] <= self.min_trade_quantity:
                self.holdings[index] = 0.0
                self.entry_prices[index] = 0.0
                self.cooldowns[index] = 0
            total_fees += fee_paid
            traded_value += gross_value

        portfolio_value = self._portfolio_value(prices)
        max_step_trade_value = self.max_trade_fraction * portfolio_value
        for index in np.where(weight_deltas > 1e-8)[0]:
            if self.cooldown_steps and self.cooldowns[index] < self.cooldown_steps:
                continue
            current_position_value = float(self.holdings[index] * prices[index])
            max_position_value = portfolio_value * self.max_position_fraction
            remaining_position_value = max(0.0, max_position_value - current_position_value)
            desired_trade_value = min(float(weight_deltas[index]) * portfolio_value, max_step_trade_value)
            fee_adjusted_cash = self.cash / (1 + self.buy_cost_pct)
            buy_value = min(desired_trade_value, remaining_position_value, fee_adjusted_cash)
            if buy_value < self.min_trade_notional:
                continue
            buy_amount = buy_value / max(float(prices[index]), 1e-8)
            if buy_amount < self.min_trade_quantity:
                continue
            gross_value = float(prices[index] * buy_amount)
            fee_paid = gross_value * self.buy_cost_pct
            previous_quantity = float(self.holdings[index])
            previous_entry_value = previous_quantity * float(self.entry_prices[index])
            self.holdings[index] += buy_amount
            self.entry_prices[index] = (previous_entry_value + gross_value) / max(float(self.holdings[index]), 1e-8)
            self.cash -= gross_value + fee_paid
            total_fees += fee_paid
            traded_value += gross_value

        executed_weights = self._asset_weights(prices)
        return total_fees, traded_value / max(portfolio_value, 1e-8), executed_weights

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
        earliest_start = self.lookback - 1
        if self.episode_length is None:
            self.time = earliest_start
            self.episode_end_time = self.max_step
        elif self.random_start:
            latest_start = self.max_step - self.episode_length
            self.time = int(self.np_random.integers(earliest_start, latest_start + 1))
            self.episode_end_time = min(self.time + self.episode_length, self.max_step)
        else:
            self.time = earliest_start
            self.episode_end_time = min(self.time + self.episode_length, self.max_step)
        self.episode_start_time = self.time

        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.entry_prices = np.zeros(self.n_assets, dtype=np.float32)
        self.cooldowns = np.full(self.n_assets, self.cooldown_steps, dtype=np.int32)
        self.risk_halted = False

        initial_prices = self.price_array[self.time]
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.previous_portfolio_return = 0.0
        self.equal_weight_holdings = (
            np.ones(self.n_assets, dtype=np.float32)
            * (self.initial_cash / self.n_assets)
            / initial_prices
        )
        self.equal_weight_value = self._benchmark_value(initial_prices)

        state = self._get_state()
        info = {
            "portfolio_value": self.portfolio_value,
            "benchmark_value": self.equal_weight_value,
            "episode_start_time": int(self.episode_start_time),
            "episode_end_time": int(self.episode_end_time),
        }
        return state, info

    def step(self, action: np.ndarray):
        if self.time >= self.episode_end_time:
            raise RuntimeError("Episode is done. Call reset() before step().")

        previous_portfolio_value = self.portfolio_value
        previous_benchmark_value = self.equal_weight_value
        self.time += 1
        prices = self.price_array[self.time]

        self.cooldowns = np.minimum(self.cooldowns + 1, self.cooldown_steps)

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
        allocation_fees = 0.0
        turnover = 0.0
        clipped_actions = np.zeros(self.n_assets, dtype=np.float32)
        executed_actions = np.zeros(self.n_assets, dtype=np.float32)
        if not self.risk_halted:
            if self.action_mode in {"target_allocation", "cash_target_allocation"}:
                clipped_actions = self._target_weights_from_action(action)
                if stop_loss_indices:
                    clipped_actions[stop_loss_indices] = 0.0
                allocation_fees, turnover, executed_actions = self._apply_target_allocation_actions(clipped_actions, prices)
            else:
                clipped_actions = np.clip(action, -1.0, 1.0).astype(np.float32)
                clipped_actions[np.abs(clipped_actions) < self.trade_deadband] = 0.0
                if stop_loss_indices:
                    clipped_actions[stop_loss_indices] = np.minimum(clipped_actions[stop_loss_indices], 0.0)
                sell_fees = self._apply_sell_actions(clipped_actions, prices)
                buy_fees = self._apply_buy_actions(clipped_actions, prices)
                average_cost_pct = max((self.buy_cost_pct + self.sell_cost_pct) / 2.0, 1e-8)
                turnover = (sell_fees + buy_fees) / max(previous_portfolio_value * average_cost_pct, 1e-8)
                executed_actions = clipped_actions.copy()
        total_fees = stop_loss_fees + position_cap_fees + halt_fees + sell_fees + buy_fees + allocation_fees

        self.portfolio_value = self._portfolio_value(prices)
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        self.equal_weight_value = self._benchmark_value(prices)

        delta_portfolio = self.portfolio_value - previous_portfolio_value
        portfolio_return = delta_portfolio / max(previous_portfolio_value, 1e-8)
        benchmark_return = (self.equal_weight_value / max(previous_benchmark_value, 1e-8)) - 1.0
        cost_penalty = total_fees / max(previous_portfolio_value, 1e-8)
        average_cost_pct = max((self.buy_cost_pct + self.sell_cost_pct) / 2.0, 1e-8)
        turnover_penalty = turnover if self.action_mode in {"target_allocation", "cash_target_allocation"} else cost_penalty / average_cost_pct
        drawdown = (self.peak_portfolio_value - self.portfolio_value) / max(self.peak_portfolio_value, 1e-8)
        rolling_volatility = self._rolling_return_volatility()
        raw_risk_adjusted_return = portfolio_return / rolling_volatility
        clipped_risk_adjusted_return = float(
            np.clip(raw_risk_adjusted_return, -self.risk_adjusted_return_clip, self.risk_adjusted_return_clip)
        )
        risk_adjusted_return = float(np.tanh(clipped_risk_adjusted_return))

        if self.reward_mode == "net_log_return_alpha":
            net_log_return = float(np.log(max(self.portfolio_value, 1e-8) / max(previous_portfolio_value, 1e-8)))
            reward = net_log_return
            reward += self.alpha_reward_weight * (portfolio_return - benchmark_return)
            reward -= self.cost_penalty_weight * cost_penalty
            reward -= self.risk_penalty_weight * drawdown
            reward -= self.drawdown_penalty_weight * drawdown
            reward -= self.turnover_penalty_weight * turnover_penalty
        else:
            net_log_return = float(np.log1p(portfolio_return))
            reward = portfolio_return
            if self.reward_mode == "return_cost_risk_volatility":
                reward += self.volatility_reward_weight * risk_adjusted_return
            reward -= self.cost_penalty_weight * cost_penalty
            reward -= self.turnover_penalty_weight * turnover_penalty
            reward -= self.risk_penalty_weight * drawdown
        if self.risk_halted:
            reward -= self.risk_halt_penalty_weight
        reward *= self.reward_scaling
        self.previous_portfolio_return = float(portfolio_return)

        terminated = self.time >= self.episode_end_time
        truncated = self.risk_halted
        state = self._get_state()
        info = {
            "portfolio_value": self.portfolio_value,
            "benchmark_value": self.equal_weight_value,
            "cash": float(self.cash),
            "holdings": self.holdings.copy(),
            "entry_prices": self.entry_prices.copy(),
            "portfolio_return": float(portfolio_return),
            "benchmark_return": float(benchmark_return),
            "net_log_return": float(net_log_return),
            "risk_adjusted_return": float(risk_adjusted_return),
            "clipped_risk_adjusted_return": float(clipped_risk_adjusted_return),
            "raw_risk_adjusted_return": float(raw_risk_adjusted_return),
            "rolling_volatility": float(rolling_volatility),
            "cost_penalty": float(cost_penalty),
            "turnover_penalty": float(turnover_penalty),
            "turnover": float(turnover),
            "drawdown": float(drawdown),
            "total_fees": float(total_fees),
            "target_weights": clipped_actions.copy(),
            "target_cash_weight": float(max(0.0, 1.0 - clipped_actions.sum())),
            "executed_actions": executed_actions.copy(),
            "cash_weight": float(self._cash_weight(self.portfolio_value)),
            "asset_weights": self._asset_weights(prices, self.portfolio_value),
            "stop_loss_symbols": stop_loss_symbols,
            "position_cap_symbols": position_cap_symbols,
            "risk_halted": self.risk_halted,
            "episode_start_time": int(self.episode_start_time),
            "episode_end_time": int(self.episode_end_time),
        }
        return state, float(reward), terminated, truncated, info
