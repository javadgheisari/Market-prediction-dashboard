# FreqAI forecasting and signals

This directory runs the official FreqAI implementation in Freqtrade's Docker
images. It is intentionally isolated from the dashboard virtual environment:
the dashboard pins TensorFlow 2.10/Python 3.10, while current Freqtrade requires
Python 3.11 or newer.

Everything is configured for dry-run trading. No exchange API key is required
to download public candles, train models, or backtest. Review the strategy and
backtest results before considering any live configuration.

## Included strategies

- `MarketForecastStrategy` trains a regressor on technical, calendar,
  multi-timeframe, shifted-candle, and correlated-pair features. Its target is
  the mean return over the next 24 one-hour candles. The predicted return also
  drives guarded entry and exit signals.
- `MarketSignalStrategy` trains a classifier whose target is `up`, `flat`, or
  `down` 24 candles ahead. A 0.4% dead band prevents tiny price changes from
  being labelled as directional signals.

Both strategies accept predictions only when FreqAI's `do_predict` flag is 1.
The configured Dissimilarity Index therefore suppresses out-of-distribution
predictions.

## Download data

From this directory:

```bash
docker compose run --rm freqai download-data \
  --config /freqtrade/user_data/config.forecast.json \
  --timeframes 1h 4h --days 180
```

Edit both JSON configs to change `pair_whitelist` and
`include_corr_pairlist`. FreqAI/Freqtrade consumes exchange pairs such as
`BTC/USDT`; the dashboard's broker CSV symbols are not interchangeable with
exchange pair data.

## Forecast models

Run a chronological FreqAI backtest with either built-in tree regressor:

```bash
docker compose run --rm freqai backtesting \
  --config /freqtrade/user_data/config.forecast.json \
  --strategy MarketForecastStrategy \
  --freqaimodel LightGBMRegressor

docker compose run --rm freqai backtesting \
  --config /freqtrade/user_data/config.forecast.json \
  --strategy MarketForecastStrategy \
  --freqaimodel XGBoostRegressor
```

The first run trains models and can take several minutes. Change the FreqAI
`identifier` whenever features, labels, or important training parameters
change, so stale cached models are not reused.

For the PyTorch MLP regressor, use the larger torch image:

```bash
docker compose --profile torch run --rm freqai-torch backtesting \
  --config /freqtrade/user_data/config.forecast.json \
  --strategy MarketForecastStrategy \
  --freqaimodel PyTorchMLPRegressor
```

## Signal classifiers

The classifier strategy can use either supported tree classifier:

```bash
docker compose run --rm freqai backtesting \
  --config /freqtrade/user_data/config.signal.json \
  --strategy MarketSignalStrategy \
  --freqaimodel LightGBMClassifier

docker compose run --rm freqai backtesting \
  --config /freqtrade/user_data/config.signal.json \
  --strategy MarketSignalStrategy \
  --freqaimodel XGBoostClassifier
```

## Dry-run adaptive forecasting

Start the default LightGBM return forecaster with periodic retraining:

```bash
docker compose up freqai
```

The default service retrains every six hours and refuses models older than 12
hours. Logs, trained models, downloaded candles, databases, and backtest output
are runtime artifacts and are excluded from Git.

To run another model continuously, override the service command using the same
arguments shown in the backtest examples, replacing `backtesting` with
`trade`. Live trading is deliberately not configured.
