# Market-prediction-dashboard
prediction app market like Forex, Crypto and Stocks (United States America)

#### Datasets:
 - dashboard/data/new
 - [dukascopy datasets](https://www.dukascopy.com/plugins/fxMarketWatch/?historical_datat)

#### Models Used:
1) Lstm
2) Gru

#### Train the improved models

The GRU and LSTM entry points share the same leakage-safe forecasting pipeline.
Pass the price CSV followed by the news CSV, or omit both paths to open two file
pickers at startup:

```bash
venv/bin/python code/improved_gru.py prices.csv daily-news.csv
venv/bin/python code/improved_lstm.py prices.csv daily-news.csv
```

The news CSV needs a publication column such as `published_at`, `date`, or
`timestamp`. For any asset, the recommended format supplies an asset-specific
impact score rather than generic emotional sentiment:

```csv
published_at,headline,asset_impact,relevance,confidence
2026-09-01 10:30,Federal Reserve signals lower rates,0.8,1.0,0.9
2026-09-01 14:00,Dollar strengthens after jobs data,-0.7,0.9,0.8
```

`asset_impact` ranges from `-1` (bearish for the selected asset) through `0`
(neutral) to `+1` (bullish). `relevance`, `importance`, `source_quality`, and
`confidence` are optional values from 0 to 1. For filenames containing `gold`
or `XAU`, headline/body text can be scored by the built-in gold phrase fallback
when no impact column exists. Explicit scores are recommended for production.

Timestamped articles affect only candles after their publication time. Rows
containing a date but no time are conservatively made available on the following
day to avoid look-ahead leakage. News influence decays with age; configure it
using `--news-half-life-hours` and `--news-lookback-hours`.

Use `--help` to change the input window, forecast horizon, split ratios, model
size, training settings, timestamp columns, news decay, or candle interval. Models and their
preprocessing metadata are saved under `models/`; plots and held-out metrics are
saved under `plots/` and `models/`, respectively.

#### Web Framework:
 - Django
