# Presentation Slides

This deck is arranged so you can directly convert each section into one slide.

## Slide 1: Title

- Project title: `Profit-Pilot: Reinforcement Learning for Cryptocurrency Trading`
- Course project on financial machine learning and sequential decision making
- Team members: add names and roll numbers

What to say: Our project uses reinforcement learning to learn a trading policy from historical crypto market data.

## Slide 2: Problem Statement

- Crypto markets are highly volatile and operate 24/7
- Manual trading is emotional and inconsistent
- We need an agent that can make sequential decisions under uncertainty
- Objective: maximize long-term portfolio growth using RL

What to say: Trading is not a one-step prediction problem. Every action changes future opportunities, so RL is a natural fit.

## Slide 3: Why Reinforcement Learning

- RL learns by interaction with an environment
- The agent observes state, takes action, gets reward, and improves over time
- This matches the trading loop naturally
- RL is more suitable than simple classification for dynamic portfolio decisions

What to say: Instead of only predicting up or down, RL directly learns what action should be taken.

## Slide 4: Literature Review

- Moody and Saffell, 2001 introduced direct reinforcement for trading
- `FinRL_Crypto` provides a full DRL pipeline for cryptocurrency trading
- `RLTrader` shows a clear Gym-based backtesting environment
- Nick Kaparinos' work shows that LSTM can help sequence-based trading models
- Recent RL trading literature such as arXiv `2511.12120` supports the use of DRL in financial decision systems

What to say: We studied multiple repositories and papers, then selected the one that gives the best academic and implementation balance.

## Slide 5: Why We Chose FinRL_Crypto

- complete pipeline: data, environment, training, validation, backtest
- crypto-focused design
- academically stronger than demo-style projects
- easier to justify in a project presentation
- allows original extensions like sentiment, risk layer, and LSTM

What to say: We used FinRL_Crypto as the main blueprint, then simplified the structure locally for better understanding.

## Slide 6: Overall Project Architecture

- historical OHLCV data download
- technical-indicator generation
- processed arrays passed to custom trading environment
- PPO agent interacts with environment
- trained model is evaluated on unseen data

What to say: The project follows a clear end-to-end pipeline from raw data to final portfolio results.

## Slide 7: Data Collection

- data source: Binance OHLCV candles through CCXT
- assets: initially `BTC/USDT` and `ETH/USDT`
- timeframe: start with `1h`
- date range: for example `2022-01-01` to `2024-12-31`

What to say: We start with a small but realistic dataset so the project remains manageable and easy to explain.

## Slide 8: Feature Engineering

- raw features: open, high, low, close, volume
- technical indicators: MACD, MACD signal, MACD histogram, RSI, CCI, ADX
- features are aligned across assets and stored as model-ready arrays

What to say: Technical indicators help the model recognize trend, momentum, and volatility patterns.

## Slide 9: Trading Environment

- custom multi-asset trading environment
- functions: `reset()` and `step(action)`
- tracks cash, holdings, transaction costs, and portfolio value
- episode ends at the end of the historical time series

What to say: The environment acts like the market simulator in which the RL agent learns.

## Slide 10: State Representation

- normalized cash balance
- current holdings of each crypto
- flattened lookback window of technical features
- state dimension depends on number of assets and lookback size

What to say: The state contains both market information and portfolio information, so the agent knows both the market and its own condition.

## Slide 11: Action Space

- continuous action vector with one value per asset
- positive action means increase position
- negative action means reduce position
- near-zero action means hold

What to say: Continuous actions are more flexible than only buy, sell, or hold because they also represent position size.

## Slide 12: Reward Function

- reward is based on portfolio-value change
- transaction costs and drawdown penalties are included
- a clipped volatility-adjusted component rewards smoother returns

What to say: This reward is better than raw profit only because it discourages overtrading, penalizes unstable equity curves, and gives a small bonus to returns earned with lower market volatility.

## Slide 13: Risk Management

- buy and sell costs included in the environment
- maximum position and trade-size caps limit concentration
- stop-loss and drawdown halt protect against large losses

What to say: Risk control is important because RL can otherwise learn overly aggressive behavior, so deterministic guardrails keep the simulation closer to practical trading.

## Slide 14: Model Selection

- baseline model: PPO
- reason: stable policy updates and good performance on noisy environments
- optional future upgrade: recurrent PPO with LSTM

What to say: We are starting with PPO because it is the easiest stable baseline, and LSTM will be tested later if time allows.

## Slide 15: Training Process

- download data
- build features
- train PPO on historical data
- test on a later unseen time period
- compare against benchmark

What to say: This keeps the workflow chronological and avoids using future information during training.

## Slide 16: How Prediction Works

- current state is passed to the trained policy
- model outputs action vector
- environment executes the action
- portfolio updates and moves to the next timestamp

What to say: At test time the model only performs inference; there is no learning during the final backtest.

## Slide 17: Our New Contributions

- cleaner local project structure for easier explanation
- optional LSTM version
- planned risk-control layer inspired by another repo
- possible sentiment or multi-timeframe extension

What to say: This is how we make the project original rather than just copying an online repository.

## Slide 18: Expected Results

- a working RL trading pipeline
- portfolio curve and backtest metrics
- comparison with benchmark
- evidence that the agent learns a non-random policy

What to say: We do not claim guaranteed profit. Our expected result is a valid and measurable RL trading system.

## Slide 19: Individual Contributions

- Member 1: data collection and preprocessing
- Member 2: environment and reward implementation
- Member 3: PPO training and evaluation
- Member 4: presentation, analysis, and result visualization

What to say: Assign real names here according to actual work done by your group.

## Slide 20: Conclusion

- RL is suitable for trading because it models sequential decisions
- FinRL_Crypto gave the best academic base
- PPO baseline is our first implementation
- LSTM and additional risk features remain future extensions

What to say: The project is practical, academically justified, and extensible.
