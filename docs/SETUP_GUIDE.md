# Setup Guide

This is the practical step-by-step guide for building your project in this folder.

## Recommended Path

Start with `PPO without LSTM`.

Why this is best first:

- easiest to understand
- easiest to debug
- easiest to explain in viva or presentation
- gives you a strong baseline before adding complexity

Add `LSTM` only after the non-LSTM pipeline works.

## Step 1: Create the Python environment

From the `Profit-Pilot` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Step 2: Create your `.env` file

```powershell
Copy-Item .env.example .env
```

The file contains:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_EXCHANGE`
- `BINANCE_SANDBOX`

## Step 3: Decide whether you need API keys immediately

### Historical data only

You do not need API keys immediately if you are only downloading public OHLCV candles.

This is the recommended starting mode for your project.

### Live trading or account access

You need API keys only when you want:

- account balances
- real orders
- testnet/private endpoint access
- user-data or signed endpoints

## Step 4: Where to get the data

### Recommended source

Use Binance market-data candles through `ccxt`.

Why:

- crypto-specific
- easy to automate
- same market family as the reference repo
- good enough for BTC and ETH class experiments

### Commands

```powershell
python -m profit_pilot.data.download_ohlcv --config config/project_config.yaml
```

This will download OHLCV candles for the symbols listed in `config/project_config.yaml`.

### If Binance access is blocked

Use one of these backups:

1. Change `exchange` in `config/project_config.yaml` from `binance` to `binanceus`
2. Replace the downloader source with another CCXT-supported exchange
3. Use CSV data from CryptoDataDownload or Kaggle, then adapt the feature builder

## Step 5: How to get Binance API keys

You only need this later if you move beyond public historical data.

### Main account keys

1. Log in to Binance.
2. Open API Management.
3. Create a new API key.
4. Save the API key and secret securely.
5. Put them in `.env`.

### Safety rules

- never enable withdrawals for a class project
- use read-only or the minimum permissions required
- never push `.env` to GitHub
- rotate keys if you accidentally expose them

### Testnet

For safer experimentation, Binance Spot Testnet exists and uses:

- base endpoint: `https://testnet.binance.vision/api`

Testnet is better than real trading for demonstration because no real funds are used.

## Step 6: Edit the project configuration

Open `config/project_config.yaml` and adjust:

- symbols
- timeframe
- date range
- lookback window
- initial cash
- training timesteps

Recommended first experiment:

- symbols: `BTC/USDT`, `ETH/USDT`
- timeframe: `1h`
- since: `2022-01-01`
- until: `2024-12-31`
- lookback: `50`

## Step 7: Build technical indicators

After raw data download:

```powershell
python -m profit_pilot.features.build_features --config config/project_config.yaml
```

This step creates:

- aligned timestamps
- close-price array
- technical-feature array
- metadata file describing symbols and feature columns

## Step 8: Train the PPO baseline

```powershell
python -m profit_pilot.train.train_ppo --config config/project_config.yaml
```

This is the recommended baseline for the project presentation.

## Step 9: Evaluate and create outputs

```powershell
python -m profit_pilot.train.evaluate_model --config config/project_config.yaml
```

This will save:

- portfolio curve plot
- basic backtest metrics JSON

## Step 10: Optional LSTM version

When the baseline works, then try recurrent PPO:

```powershell
python -m profit_pilot.train.train_ppo --config config/project_config.yaml --use-lstm
python -m profit_pilot.train.evaluate_model --config config/project_config.yaml --use-lstm
```

Use this only as an extension, not as the starting point.

## Step 11: How to add originality

You must not submit a plain copy of an online repo. Add at least one clear improvement.

Best options:

1. Add a sentiment score to the state
2. Add multi-timeframe features such as `15m + 1h`
3. Add a drawdown-based trading halt
4. Add LSTM after the PPO baseline works
5. Compare `raw PnL reward` versus `benchmark-adjusted reward`

## Step 12: If you want to run the original FinRL_Crypto pipeline too

The reference copy is already in:

`references/FinRL_Crypto`

Typical reference flow there is:

```powershell
cd references\FinRL_Crypto
python 0_dl_trainval_data.py
python 1_optimize_cpcv.py
python 2_validate.py
python 4_backtest.py
```

Before that, you would need to configure `references/FinRL_Crypto/config_api.py`.

For your own submission, the root `Profit-Pilot` scaffold is cleaner and easier to explain.

## Suggested Work Sequence for Your Team

1. Finish the non-LSTM PPO baseline
2. Prepare slides and methodology using the baseline
3. Add one originality feature
4. Re-run results
5. If time remains, try LSTM

## Common Problems

### Problem: downloader fails

- check internet access
- try `binanceus` instead of `binance`
- shorten the date range first

### Problem: no processed features are created

- check that raw CSV files exist under `data/raw`
- confirm symbol names match your config

### Problem: PPO training is too slow

- reduce date range
- reduce symbols from 2 to 1
- reduce total timesteps
- start with `1h` instead of `5m`

### Problem: LSTM is unstable

- go back to non-LSTM baseline
- fix the base pipeline first

## Official References

- Binance market-data endpoints: <https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints>
- Binance request security: <https://developers.binance.com/docs/binance-spot-api-docs/rest-api/request-security>
- Binance Spot Testnet: <https://developers.binance.com/docs/binance-spot-api-docs/testnet/rest-api>
