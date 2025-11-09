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
last 24 hours, and writes CSV files into `data/external/` for each source.

### Data source endpoints

The fetchers are wired to the user's preferred provider endpoints:

- **CryptoQuant OHLCV** – `https://api.cryptoquant.com/v1/<asset>/market-data/price-ohlcv` with hourly windows, spot market, and `all_exchange` aggregation.
- **CoinStats sentiment** – `https://openapiv1.coinstats.app/insights/fear-and-greed`, authenticated via the `COINSTATS_API_KEY` header.
- **Binance order book liquidity** – `https://api3.binance.com/api/v3/depth`, accessed without authentication using the symbols defined under `assets.*.binance.symbol`.

Populate `CRYPTOQUANT_API_KEY` and `COINSTATS_API_KEY` in your environment (or `secrets/credentials.json`) before running the pipeline to authenticate with the private APIs.

