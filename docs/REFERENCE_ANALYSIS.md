# Reference Analysis

This file summarizes what the four reference repositories actually contain after inspecting their code, not just their README claims.

## Final Recommendation

Use `FinRL_Crypto` as the main academic base, then borrow selected ideas from the other repositories:

- borrow risk rules from `Crypto-RL-Trading-Bot`
- borrow Gym explainability from `RLTrader`
- keep LSTM as an optional extension from `Automated-Cryptocurrency-trading-using-Deep-RL`

## Repo 1: oyi77/Crypto-RL-Trading-Bot

Reference repository: <https://github.com/oyi77/Crypto-RL-Trading-Bot>

### What it is good at

- clear paper-trading story
- real-time Binance data idea
- strong risk-manager design
- easy to explain operational flow

### What is actually ready-made

- a detailed risk manager in `src/lib/risk/riskManager.ts`
- a paper-trading runner in `src/scripts/runPaperTrading.ts`
- market-analysis and dashboard utilities

### What is weak or misleading

- the file `src/lib/rl/rlAgent.ts` is not a true PPO actor-critic implementation
- it uses a simple dense network with epsilon exploration and Q-value style target updates
- `src/lib/rl/tradingEnvironment.ts` uses placeholder trade simulation with random win/loss behavior
- this makes it weak as the main academic RL base

### Best thing to borrow

- risk rules such as max risk per trade, drawdown guard, stop-loss, take-profit, and max open positions

## Repo 2: notadamking/RLTrader

Reference repository: <https://github.com/notadamking/RLTrader>

### What it is good at

- clean Gym-style environment
- simple training story
- backtesting orientation
- reward strategies are easy to explain

### What is actually ready-made

- `lib/env/TradingEnv.py` implements a proper Gym environment
- action space is discrete and defaults to 24 actions
- reward strategies are modular in `lib/env/reward/`
- optimization pipeline exists in `optimize.py`

### Core environment idea

- state = current market observation plus normalized account history
- actions = buy, sell, hold with different size bins
- reward = incremental profit or weighted unrealized profit

### Limitation

- older stack
- mainly historical backtesting
- less professional validation story than FinRL_Crypto

### Best thing to borrow

- Gym environment clarity and modular reward design

## Repo 3: AI4Finance-Foundation/FinRL_Crypto

Reference repository: <https://github.com/AI4Finance-Foundation/FinRL_Crypto>

### Why this is the best base

- full pipeline already exists: download, optimize, validate, backtest
- multiple DRL algorithms are supported
- it is presented as an academic project, so it fits classroom expectations well
- it has a strong anti-overfitting story through CPCV, walk-forward, and validation scripts

### What is actually ready-made

- `processor_Binance.py` downloads data and builds technical indicators
- `environment_CCXT.py` defines the trading environment
- `1_optimize_cpcv.py`, `1_optimize_kcv.py`, and `1_optimize_wf.py` handle tuning workflows
- `2_validate.py` analyzes the Optuna study
- `4_backtest.py` evaluates trained agents

### Real environment details from the code

- state dimension is `1 + number_of_assets + tech_feature_count * lookback`
- action dimension is `number_of_assets`
- actions are continuous, not simple buy/sell/hold labels
- reward is based on portfolio-value change minus equal-weight benchmark change
- transaction costs are applied during buy and sell operations

### Important weakness

- the original repo forces Binance API-key setup even for data download
- the stack is more complex than needed for a student project
- some dependency choices are older and less beginner-friendly than a modern SB3 setup

### Best thing to borrow

- the academic pipeline and state-action-reward design

## Repo 4: NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL

Reference repository: <https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL>

### What it is good at

- sequence modeling focus
- multi-asset portfolio switching idea
- LSTM justification is clear

### What is actually ready-made

- DQN-based training in `main_DQN.py`
- 5-action setup: one action per asset plus USD
- state includes 24-hour price statistics and one-hot portfolio encoding

### Limitation

- DQN architecture is older for this use case
- the code is less modular than FinRL_Crypto
- not the easiest main repo for a class project

### Best thing to borrow

- LSTM extension idea once the PPO baseline is already working

## Decision

For `Profit-Pilot`, the best structure is:

1. Main reference logic from `FinRL_Crypto`
2. Cleaner local implementation using our own project structure
3. PPO baseline first
4. LSTM later only if time allows
5. Risk layer inspired by Repo 1

## Sources

- FinRL_Crypto: <https://github.com/AI4Finance-Foundation/FinRL_Crypto>
- RLTrader: <https://github.com/notadamking/RLTrader>
- Crypto-RL-Trading-Bot: <https://github.com/oyi77/Crypto-RL-Trading-Bot>
- Automated-Cryptocurrency-trading-using-Deep-RL: <https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL>
