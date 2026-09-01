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
Run either script with a CSV path, or omit the path to use the file picker:

```bash
venv/bin/python code/improved_gru.py data/forex/EUR-USD.csv
venv/bin/python code/improved_lstm.py data/forex/EUR-USD.csv
```

Use `--help` to change the input window, forecast horizon, split ratios, model
size, training settings, timestamp column, or candle interval. Models and their
preprocessing metadata are saved under `models/`; plots and held-out metrics are
saved under `plots/` and `models/`, respectively.

#### Web Framework:
 - Django
