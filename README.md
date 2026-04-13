# Profit-Pilot

`Profit-Pilot` is a student-friendly crypto trading project scaffold inspired mainly by `FinRL_Crypto`, with a cleaner local structure for understanding, presenting, and extending the work.

The goal is not to blindly copy an online repository. The goal is to build an understandable project with:

- historical crypto data download
- feature engineering with technical indicators
- a custom multi-asset trading environment
- PPO training and backtesting
- a clear place to add your own originality later

## Why We Chose FinRL_Crypto

From the reference repositories, `FinRL_Crypto` is the best base for this project because it already demonstrates the full academic pipeline:

1. Download data
2. Build technical indicators
3. Train a DRL agent
4. Validate and backtest
5. Compare against a benchmark

We still keep useful ideas from the other repositories:

- `Crypto-RL-Trading-Bot`: strong risk management ideas
- `RLTrader`: clean Gym-style environment structure
- `Automated-Cryptocurrency-trading-using-Deep-RL`: LSTM-based sequence modeling idea

## Folder Layout

```text
Profit-Pilot/
├── config/
├── data/
├── docs/
├── models/
├── reports/
├── src/profit_pilot/
└── references/
```

`references/` contains the cloned public repositories used for study. Your actual project work should happen in the root structure, not inside those reference repos.

## Quick Start

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
pip install -e .
```

### 3. Create your environment file

```powershell
Copy-Item .env.example .env
```

For historical OHLCV downloads, API keys are optional because market-data endpoints are public. For live trading or account endpoints, API keys are required.

### 4. Download historical data

```powershell
python -m profit_pilot.data.download_ohlcv --config config/project_config.yaml
```

### 5. Build features and arrays

```powershell
python -m profit_pilot.features.build_features --config config/project_config.yaml
```

### 6. Train PPO

```powershell
python -m profit_pilot.train.train_ppo --config config/project_config.yaml
```

### 7. Evaluate the model

```powershell
python -m profit_pilot.train.evaluate_model --config config/project_config.yaml
```

### 8. Optional LSTM version later

You can switch to recurrent PPO later:

```powershell
python -m profit_pilot.train.train_ppo --config config/project_config.yaml --use-lstm
```

## Data Source

Default source: Binance OHLCV candles through `ccxt`.

- Good for: BTC, ETH, BNB, SOL, and other crypto pairs
- Best for class project: public historical candles, no private order permissions needed
- Alternative if Binance access fails in your region: switch exchange name to `binanceus`, or replace the downloader with another public source

## What To Read Next

- [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)
- [docs/REFERENCE_ANALYSIS.md](docs/REFERENCE_ANALYSIS.md)
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
- [docs/PRESENTATION_SLIDES.md](docs/PRESENTATION_SLIDES.md)

## Important Note

This scaffold keeps the project understandable. The reference `FinRL_Crypto` repo uses ElegantRL internals and more complex validation scripts. For class work, this local version keeps the same overall logic but is easier to explain and extend.
