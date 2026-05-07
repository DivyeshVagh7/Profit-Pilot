# Methodology

This document is written so you can directly use it in your report and presentation.

## 1. Project Goal

The goal of `Profit-Pilot` is to build a reinforcement-learning based crypto trading agent that learns a profitable portfolio-allocation policy from historical market data. The agent observes market conditions, selects trading actions, receives reward from portfolio performance, and gradually improves its policy through repeated interaction with the environment.

## 2. Why Reinforcement Learning

Reinforcement learning is suitable for trading because financial markets are sequential decision-making problems:

- the current action affects future portfolio value
- market dynamics are uncertain and time-dependent
- the objective is long-term cumulative reward, not only one-step accuracy

Unlike supervised learning, RL directly models the decision loop of trading.

## 3. Why We Selected FinRL_Crypto as the Main Reference

Among the studied repositories, `FinRL_Crypto` was selected because it provides the strongest end-to-end academic pipeline.

### Justification

- it already follows the proper workflow of data download, environment construction, training, validation, and backtesting
- it uses crypto-focused data and exchange-oriented processing
- it includes an anti-overfitting story, which is important in financial ML
- it is easier to justify academically than a purely live-trading or demo-style repo

## 4. Our Project Architecture

The project follows this pipeline:

1. Download historical OHLCV data
2. Build technical indicators
3. Convert processed data into arrays
4. Feed arrays into a custom trading environment
5. Train a PPO agent
6. Evaluate portfolio performance on unseen data
7. Add one or more original extensions

### Why this architecture

- it separates data, features, environment, model, and evaluation cleanly
- it is easy to explain in slides
- it makes debugging and extension easier

## 5. Trading Environment

The environment is formulated as a Markov Decision Process.

### Core functions

- `reset()`: starts a new episode with initial cash and zero holdings
- `step(action)`: executes the action, updates holdings and cash, computes reward, and returns the next state
- terminal condition: episode ends when the final timestamp is reached

### Why a custom environment is used

- it gives full control over state, reward, and trading rules
- it keeps the project original instead of being only a copied script
- it allows later additions such as sentiment, risk limits, and multi-timeframe inputs

## 6. State Representation

Our state is inspired directly by the real `FinRL_Crypto` environment structure.

### State components

- normalized cash balance
- normalized current holdings for each crypto asset
- flattened lookback window of technical features

### Technical features per asset

- open
- high
- low
- close
- volume
- MACD
- MACD signal
- MACD histogram
- RSI
- CCI
- ADX

### Final state form

If there are `N` assets and `F` technical features per timestep, then:

`state_dim = 1 + N + (lookback x N x F)`

### Why this state is used

- cash tells the agent how much buying power remains
- holdings tell the agent current exposure
- lookback features capture temporal patterns
- technical indicators help the model detect trend, momentum, and volatility

## 7. Action Space

The base design uses a continuous action vector with one action per asset.

### Action meaning

- positive value: buy that asset
- negative value: sell that asset
- near zero: hold

In the current implementation, action magnitude is interpreted as a notional trade fraction, not a raw coin quantity. For example, a positive action buys up to `action x max_trade_fraction x portfolio_value`, subject to cash, fees, and position caps. This keeps the action interface consistent across assets with very different prices.

### Why continuous actions are used

- this matches the main design of `FinRL_Crypto`
- it allows position sizing, not only direction prediction
- it is more realistic for portfolio allocation than only `buy/sell/hold`

### Simpler explanation for presentation

If needed, you can explain each action as:

- increase position
- decrease position
- maintain position

## 8. Reward Function

The implemented reward is based on portfolio return, transaction costs, drawdown risk, and a small clipped volatility-adjusted return bonus:

`reward_t = portfolio_return + beta x tanh(clip(portfolio_return / rolling_volatility)) - cost_weight x cost_penalty - risk_weight x drawdown`

Where:

- `portfolio_return` = change in portfolio value divided by previous portfolio value
- `rolling_volatility` = mean rolling standard deviation of asset returns over the configured volatility window
- `cost_penalty` = trading fees divided by previous portfolio value
- `drawdown` = current peak-to-trough portfolio decline
- `beta` = volatility reward weight
- `clip(...)` is bounded by `risk_adjusted_return_clip`, then `tanh(...)` compresses the term to avoid unstable PPO updates during unusually low-volatility windows

### Why this reward is better than raw PnL only

- raw PnL can reward lucky exposure during broad market rallies
- transaction-cost penalties discourage excessive churn
- drawdown penalties discourage unstable equity curves
- the volatility-adjusted term rewards cleaner returns without letting it dominate the base return signal
- clipping the volatility-adjusted term keeps the reward numerically stable

### Transaction-cost handling

- buy cost is subtracted when buying
- sell cost is subtracted when selling
- forced risk exits also pay sell costs
- this makes the simulation more realistic

## 9. Risk Control

The environment now includes a lightweight deterministic risk layer inspired by practical trading-bot rules.

### Implemented risk additions

- maximum position fraction per asset
- maximum trade fraction per step
- fixed stop-loss based on each asset's average entry price
- drawdown-based halt that liquidates positions and ends the episode when the portfolio breaches the configured limit
- optional cooldown after selling

### Why risk control matters

- a profit-maximizing RL agent can otherwise become unstable
- position caps prevent a single asset from dominating the portfolio
- stop-loss logic prevents one position from silently becoming a large loss
- drawdown halts teach the policy that catastrophic equity curves are unacceptable

## 10. Model Choice

The baseline model is `PPO`.

### Why PPO is selected

- stable and widely used in RL
- works well with noisy continuous-control tasks
- easier to explain than many off-policy alternatives
- strong baseline before trying recurrent architectures

### LSTM decision

The team will decide later whether to add LSTM.

Recommended approach:

1. Build PPO baseline first
2. Validate and present the working baseline
3. Upgrade to recurrent PPO only if time permits

### Why LSTM is optional

- LSTM can capture temporal dependencies better
- but it increases implementation complexity and debugging effort
- for a student project, a strong baseline is more important than premature complexity

## 11. Training Procedure

The training pipeline is:

1. Download data
2. Build features
3. Train PPO on historical data
4. Evaluate on a later unseen time range
5. Compare final portfolio value against benchmark

### Training justification

- chronological splitting prevents leakage from future data
- backtesting on unseen data checks whether the learned policy generalizes
- saved plots and metrics make presentation easier

## 12. Prediction and Inference

During inference:

1. The environment provides the current state
2. The trained PPO policy outputs an action vector
3. The environment executes the action
4. Portfolio value is updated
5. The next state is returned

### Why this is important to explain

- it shows the full RL loop clearly
- it answers "how does the model actually trade?"

## 13. Expected Results

We do not claim guaranteed profit. The expected outcomes are:

- the agent should learn a non-random trading policy
- the portfolio curve should be comparable to or better than a benchmark on selected periods
- the project should demonstrate a full RL trading pipeline with measurable outputs

## 14. Original Contributions Planned for This Project

At least one of the following will be implemented:

1. Additional 5-minute market-regime features
2. Sentiment score as an extra state feature
3. Drawdown-based risk halt
4. LSTM-based recurrent PPO

## 15. Short Report-Ready Version

This project adopts a reinforcement-learning based approach for cryptocurrency trading, using `FinRL_Crypto` as the primary reference architecture. Historical OHLCV market data are collected from Binance, transformed into technical-indicator enriched features, and fed into a custom multi-asset trading environment. The environment represents the state using normalized cash, current holdings, and a lookback window of market and technical features. The action space is defined as a continuous allocation vector, where positive values increase positions, negative values reduce positions, and near-zero values correspond to holding. The reward function combines portfolio return, transaction-cost penalties, drawdown penalties, and a small volatility-adjusted return component, encouraging profitable but smoother portfolio growth. PPO is selected as the baseline RL algorithm because of its stability and suitability for noisy financial environments. The model is trained on historical data and evaluated through out-of-sample backtesting. To make the project original and practically meaningful, the current implementation adds deterministic risk controls, while future extensions such as sentiment integration, multi-timeframe features, and optional LSTM-based recurrent PPO remain available.

## Literature Pointers

- Moody and Saffell, 2001: direct reinforcement for trading
- FinRL_Crypto paper: <https://arxiv.org/abs/2209.05559>
- Recent ensemble stock-trading RL literature: <https://arxiv.org/abs/2511.12120>
