# HMM Crypto QuantConnect Strategy

This repository scaffolds a Hidden Markov Model (HMM) driven crypto trading workflow designed to run inside QuantConnect Lean. The layout mirrors the end-to-end pipeline described in the project plan: external data ingestion, normalization, feature engineering, model training, probability export, and QuantConnect algorithm integration.

## Repository Layout

```text
config/                 # YAML & JSON configuration for assets, features, schedules
qcsrc/                  # Code deployed to QuantConnect (algorithm, custom data, models, etc.)
data/                   # Raw, interim, processed datasets and model artifacts
qc_data/                # Custom data exports formatted for QuantConnect consumption
notebooks/              # Research notebooks for diagnostics and EDA
tests/                  # Pytest suite scaffolding for pipelines and utilities
```

## Environment Setup

1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `config/credentials.example.json` to `secrets/credentials.json` and fill in API keys.
4. Populate a `.env` file (see `.env.example`) when running pipelines locally.

## QuantConnect Cloud Deployment Notes

QuantConnect supplies many scientific Python libraries, but custom dependencies and artifacts must be uploaded to the project for cloud execution. To ensure the cloud environment receives the latest versions:

- Package any libraries not provided by QuantConnect into wheels or include their source in the project before syncing.
- Upload updated files via the QuantConnect web IDE or `lean cloud push` after every change to `requirements.txt`, model artifacts under `data/models/`, or custom data in `qc_data/`.
- When using the Lean CLI, run `lean cloud push` from the repo root and confirm the manifest shows modified dependencies and artifacts before launching a backtest or live job.

Document updates to this process alongside strategy changes so future deployments stay reproducible.

## Lean CLI (optional)

If you run the pipeline locally with the Lean CLI, configure `lean.json` (to be added) to point at `qcsrc/HMMCryptoAlgorithm.py` and relevant data directories.

### Providing API keys to Lean CLI runs

When invoking the QuantConnect algorithm or any of the pipeline commands through the Lean CLI, ensure the same credentials used locally are available to the spawned Python process:

1. Keep your populated `secrets/credentials.json` file in place so the helper at `qcsrc/util/secrets.py` can resolve provider keys.
2. Alternatively (or additionally) export the environment variables immediately before launching Lean. For example:
   ```bash
   export CRYPTOQUANT_API_KEY="<your-key>"
   export COINSTATS_API_KEY="<your-key>"
   # Binance is optional for the current depth endpoint, but include if you add private calls
   export BINANCE_API_KEY="<your-key>"
   export BINANCE_API_SECRET="<your-secret>"
   lean backtest "HMM_CRYPTO_QC" --data-provider local
   ```
   You can also inline them on a single command: `CRYPTOQUANT_API_KEY=... COINSTATS_API_KEY=... lean backtest ...`.
3. If you rely on a `.env` file, the pipeline entry points already load it via `python-dotenv`. For Lean CLI executions, call the helper script through a small wrapper that sources the `.env` file (e.g., `set -a; source .env; set +a; lean backtest ...`).

These approaches keep sensitive values outside version control while making them available for both offline data pulls and CLI-managed backtests.

## Testing

Pytest scaffolding is in `tests/`; add unit tests as pipeline modules are implemented:

```bash
pytest
```

## Running the external data pipeline

To execute all external fetchers locally, run:

```bash
python -m qcsrc.pipeline.run_all
```

The script loads symbols from `config/assets.yaml`, requests hourly data for the
last 24 hours, and writes CSV files into `data/external/` for each source. After
every fetch cycle it aligns the raw datasets into hourly UTC frames under
`data/interim/` and generates scaled feature matrices plus fitted scalers under
`data/processed/` and `data/models/hmm/` respectively. When enough hourly bars are
available the orchestrator also trains a Gaussian HMM per asset, writing model
artifacts to `data/models/hmm/<SYMBOL>_hmm.pkl` and diagnostics (log-likelihood,
posterior probabilities, and error metrics) to `data/models/diagnostics/`. The
trained models are then reloaded to score the latest feature matrices, and the
resulting posterior probabilities are exported to QuantConnect-ready CSV files
under `qc_data/custom/HMMStateProba/<symbol>.csv` with hourly timestamps for warm
start continuity.

## QuantConnect custom data definitions

Three PythonData classes live under `qcsrc/custom_data/` to ingest the exported
CSV feeds once the strategy is running on QuantConnect:

- `LiquidityBitAsk` parses order book depth, spread, and imbalance metrics stored
  under `qc_data/custom/LiquidityBitAsk/`.
- `MarketSentiment` exposes CoinStats fear-and-greed scores and confidence values
  under `qc_data/custom/MarketSentiment/`.
- `HMMStateProba` reads posterior probabilities from
  `qc_data/custom/HMMStateProba/` and offers a `GetSignal(threshold)` helper that
  returns `long`, `short`, or `flat` based on the configured probability
  threshold (default 0.7).

Each class returns a one-hour bar ending at `Time + 1 hour`, aligning with the
hourly cadence used throughout the pipeline.

### Data source endpoints

The fetchers are wired to the user's preferred provider endpoints:

- **CryptoQuant OHLCV** – `https://api.cryptoquant.com/v1/<asset>/market-data/price-ohlcv` with hourly windows, spot market, and `all_exchange` aggregation.
- **CoinStats sentiment** – `https://openapiv1.coinstats.app/insights/fear-and-greed`, authenticated via the `COINSTATS_API_KEY` header.
- **Binance order book liquidity** – `https://api3.binance.com/api/v3/depth`, accessed without authentication using the symbols defined under `assets.*.binance.symbol`.

Populate `CRYPTOQUANT_API_KEY` and `COINSTATS_API_KEY` in your environment (or `secrets/credentials.json`) before running the pipeline to authenticate with the private APIs.

## QuantConnect algorithm integration

`qcsrc/HMMCryptoAlgorithm.py` wires the exported probability series into a
QuantConnect-ready trading loop:

- Each configured crypto asset is added at minute resolution with an hourly
  consolidator for diagnostics.
- Custom data subscriptions load liquidity, sentiment, and HMM posterior
  probabilities from the `qc_data/custom/**` directories.
- Incoming probability bars are converted into trading signals. If the bullish
  probability is at least the configured threshold (default `0.7`), the strategy
  opens or maintains a long position; if the bearish probability breaches the
  threshold it flips short; if the consolidation probability dominates, the
  position is flattened. Signals persist until a different regime exceeds the
  threshold, matching the probability management rules in the project brief.
- Retraining checkpoints are logged according to `config/settings.yaml` so you
  can schedule offline retraining and re-upload refreshed models and
  probabilities.

To backtest locally with the Lean CLI:

```bash
lean backtest "HMM_CRYPTO_QC" --data-provider local
```

Ensure the `qc_data/custom/**` exports and model artifacts under `data/models/`
are synced to the QuantConnect cloud project before starting live or cloud
backtests so the algorithm reads the latest probabilities and scalers.

## Backtesting and evaluation

- Run the full offline pipeline first to fetch data, align features, train the
  HMM, export probabilities, and emit diagnostic parquet/CSV artifacts.
- Execute a local backtest via Lean CLI (or on QuantConnect cloud) to generate
  trading statistics. The algorithm logs per-symbol error metrics and persists
  them to `data/models/diagnostics/<symbol>_metrics.json` at the end of each run
  when the filesystem is writable.
- For reproducible offline evaluation, run:

  ```bash
  python -m qcsrc.pipeline.evaluate
  ```

  This command compares exported probability expectations against the next
  hour's realized returns, computing MAE, RMSE, MAPE, and directional accuracy
  and writing the results to `data/models/diagnostics/`.
- Inspect the generated JSON files and QuantConnect performance report to tune
  the HMM thresholds (`config/settings.yaml`) or feature definitions
  (`config/features.yaml`).

