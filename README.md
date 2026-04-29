# Profit-Pilot: Reinforcement Learning for Cryptocurrency Trading

Profit-Pilot is a research-grade framework for developing, training, and evaluating reinforcement learning (RL) agents for multi-asset cryptocurrency portfolio management. It is designed for academic, educational, and practical experimentation with RL-based trading strategies, using real historical market data and robust evaluation metrics.

---

## 🚀 Project Goal

Build a reinforcement learning agent that learns a profitable portfolio-allocation policy from historical crypto market data. The agent observes market conditions, selects trading actions, receives reward from portfolio performance, and improves its policy through repeated interaction with a custom environment.

## 📚 Why Reinforcement Learning?
- Trading is a sequential decision-making problem: actions affect future outcomes.
- RL optimizes long-term cumulative reward, not just one-step prediction.
- RL directly models the trading loop: observe → act → reward → learn.

## 🏗️ Project Architecture

1. **Data Download**: Fetch historical OHLCV data for multiple crypto assets (e.g., BTC/USDT, ETH/USDT) from Binance.
2. **Feature Engineering**: Compute technical indicators and process data into arrays.
3. **Custom Environment**: Feed arrays into a Gymnasium-style multi-asset trading environment with realistic constraints (fees, slippage, risk limits).
4. **Agent Training**: Train a PPO (Proximal Policy Optimization) agent on the environment.
5. **Evaluation**: Test on unseen data, compare to benchmarks, and analyze risk/return metrics.
6. **Extensions**: Add original features (e.g., LSTM, risk overlays, new reward functions).

## 🧩 Key Features

- Modular pipeline: data, features, environment, agent, evaluation
- Multi-asset support (BTC, ETH, etc.)
- Realistic trading constraints (fees, max drawdown, stop-loss, cooldown)
- PPO baseline (easily extensible to LSTM, other RL algorithms)
- Academic references and methodology included
- Ready-to-run Colab and local Jupyter notebooks
- Reproducible experiments and artifact tracking

## 📊 Example Results

**Colab 5m PPO Agent vs. Equal-Weight Benchmark (2024-2026 test set):**

| Metric                | PPO Agent | Equal-Weight |
|-----------------------|-----------|--------------|
| Final Value           | $10,179   | $8,200       |
| Total Return (%)      | 1.79      | -17.99       |
| Annualized Return (%) | 6.10      | -48.37       |
| Sharpe Ratio          | 0.39      | -0.78        |
| Max Drawdown (%)      | 9.55      | 43.20        |
| Trade Events          | 28,246    | 0            |

See `reports/colab_5m_2024_2026/profit_pilot_ppo_5m_t4_test_metrics.json` for full details.

## 📦 Directory Structure

- `src/profit_pilot/` — Core package (data, env, features, train, utils)
- `data/` — Raw and processed market data bundles
- `models/` — Saved agent checkpoints and training summaries
- `reports/` — Evaluation metrics, plots, and experiment artifacts
- `notebooks/` — Jupyter/Colab notebooks for training and analysis
- `docs/` — Methodology, setup guide, reference analysis, slides
- `config/` — Project and experiment configuration files
- `tests/` — Unit tests

## ⚡ Quickstart

1. **Clone the repo and set up the environment:**
   ```powershell
   git clone <repo-url>
   cd ProfitPilot
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install -e .
   ```
2. **Configure API keys:**
   - Copy `.env.example` to `.env` and fill in your Binance API credentials (or use sandbox mode).
3. **Run a notebook:**
   - Open `notebooks/colab_t4_5m_profit_pilot_training.ipynb` or `notebooks/Local Run/Local_run_5m.ipynb` in Jupyter/Colab.
   - Follow the cells to train and evaluate the agent.

## 🛠️ Main Modules

- `src/profit_pilot/data/download_ohlcv.py` — Download historical OHLCV data
- `src/profit_pilot/features/build_features.py` — Compute technical indicators
- `src/profit_pilot/env/multi_crypto_env.py` — Custom multi-asset trading environment
- `src/profit_pilot/train/train_ppo.py` — PPO agent training loop
- `src/profit_pilot/train/evaluate_model.py` — Evaluation and metrics
- `src/profit_pilot/utils/io.py` — I/O utilities

## 🧪 Testing

- Run `pytest` or check `tests/test_multi_crypto_env.py` for environment tests.

## 📖 Documentation

- `docs/METHODOLOGY.md` — Project methodology and academic rationale
- `docs/SETUP_GUIDE.md` — Step-by-step setup instructions
- `docs/REFERENCE_ANALYSIS.md` — Analysis of reference repositories
- `docs/PRESENTATION_SLIDES.md` — Slide-ready project summary

## 📝 References

- [FinRL_Crypto](https://github.com/AI4Finance-Foundation/FinRL_Crypto)
- [oyi77/Crypto-RL-Trading-Bot](https://github.com/oyi77/Crypto-RL-Trading-Bot)
- [notadamking/RLTrader](https://github.com/notadamking/RLTrader)
- [Automated-Cryptocurrency-trading-using-Deep-RL](https://github.com/llSourcell/Automated-Cryptocurrency-trading-using-Deep-RL)
- Moody & Saffell (2001), "Learning to Trade via Direct Reinforcement"
- Recent RL trading literature (see `docs/METHODOLOGY.md`)

## 👥 Authors & Acknowledgements

- Divyesh, Utsker, Smit

## 📄 License

Specify your license here (MIT, Apache 2.0, etc.)

---

For questions, open an issue or see the documentation in the `docs/` folder.
