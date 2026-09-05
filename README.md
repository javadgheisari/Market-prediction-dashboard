# Market Prediction Dashboard

A research and demonstration project for forecasting Forex, cryptocurrency,
precious-metal, and US stock prices with LSTM and GRU neural networks. The
repository contains:

- an improved, leakage-aware command-line forecasting pipeline that combines
  market candles with timestamped asset news;
- a Django dashboard for selecting an asset and running the original LSTM or
  GRU demonstration; and
- sample OHLCV datasets for the symbols displayed by the dashboard.

> **Important:** The forecasts are experimental and are not financial advice.
> Historical test results do not guarantee future performance.

## Requirements

- Python 3.10 is recommended. The pinned TensorFlow 2.10 dependency does not
  support newer Python releases such as Python 3.11.
- `pip` and `venv`.
- Enough memory and disk space to train TensorFlow models. A GPU is optional;
  TensorFlow can train on a CPU, but it will be slower.
- A graphical desktop only if CSV paths are omitted and the optional file
  pickers are used. Supplying paths on the command line works in headless
  environments.

## Installation

From the repository root:

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.\venv\Scripts\Activate.ps1
```

## Data requirements

### Market-price CSV

The improved trainers require a timestamp column and a positive close-price
column. `Open`, `High`, `Low`, and `Volume` are optional, but they add useful
features when present.

```csv
Date,Open,High,Low,Close,Volume
2022-08-01 01:30,1764.809,1766.110,1762.898,1763.134,450080.0073
2022-08-01 03:30,1763.124,1763.968,1760.338,1762.574,1592889.9646
```

Rows are sorted by timestamp, duplicate timestamps are reduced to their last
value, and rows without a valid positive close are discarded. The dataset must
also be long enough for the configured input window, forecast horizon, and
train/validation/test partitions. If it is too short, reduce `--window` or
`--forecast-steps`, or provide more observations.

Bundled dashboard data is stored in `mysite/dashboard/data/new/`. Additional
historical data can be obtained from the
[Dukascopy historical data feed](https://www.dukascopy.com/plugins/fxMarketWatch/?historical_data).

### News CSV

News must overlap the price dataset's training period and include a publication
column named one of `published_at`, `published`, `datetime`, `timestamp`,
`date`, `Date`, or `time`. The recommended format is:

```csv
published_at,headline,asset_impact,relevance,importance,source_quality,confidence
2022-08-01 00:15,Federal Reserve signals lower rates,0.8,1.0,0.9,0.9,0.9
2022-08-01 02:00,Dollar strengthens after jobs data,-0.7,0.9,0.8,0.9,0.8
```

`asset_impact` describes the expected directional effect on the selected asset:

- `-1` is strongly bearish;
- `0` is neutral; and
- `+1` is strongly bullish.

Numeric values or labels such as `bullish`, `neutral`, and `bearish` are
accepted. Optional `relevance`, `importance`, `source_quality`, and `confidence`
columns must contain values from 0 to 1 and are combined into a quality weight.

For files whose name contains `gold` or `XAU`, the trainer can estimate impact
from headline/body text with a small built-in phrase scorer when no impact
column is supplied. Other assets require an explicit asset-impact column.
Explicit, asset-specific scores are recommended for serious experiments.

Timestamped articles affect only candles at or after publication. A row that
contains only a date is conservatively delayed until the next day to prevent
look-ahead leakage. News impact decays over time according to
`--news-half-life-hours` and is ignored after `--news-lookback-hours`.

See [`data/news/gold_news_sample.csv`](data/news/gold_news_sample.csv) for a
complete example. Its dates must be adjusted or paired with market data from
the same period before training.

### Collect current news

`code/news_crawler.py` collects headline metadata from RSS search results and
updates one CSV per asset. It does not download or scrape article bodies. Run
it from the repository root, for example:

```bash
python code/news_crawler.py --assets XAU BTC AAPL --days 7
```

The output files are `data/news/XAU_news.csv`, `data/news/BTC_news.csv`, and
`data/news/AAPL_news.csv`. Running the command again merges new articles and
deduplicates them by URL. Omit `--assets` to collect every symbol in
`config/news_assets.json`; edit that file to add assets or refine search terms.

The crawler assigns a deliberately low-confidence, keyword-based
`asset_impact` estimate so its output can be read by the forecasting pipeline.
Treat it as a baseline: review the scores or replace them with a dedicated
asset-impact model before relying on them in an experiment. To use a different
RSS search service, pass a URL template containing `{query}` via
`--feed-template`.

## Train an improved model

The LSTM and GRU entry points use the shared pipeline in
`code/sequence_forecaster.py`. From the repository root, run either:

```bash
venv/bin/python code/improved_lstm.py \
  mysite/dashboard/data/new/XAU.csv data/news/gold_news_sample.csv \
  --date-col Date

venv/bin/python code/improved_gru.py \
  mysite/dashboard/data/new/XAU.csv data/news/gold_news_sample.csv \
  --date-col Date
```

The example files currently cover different date ranges, so they demonstrate
the expected command and schemas but will not train together until their dates
overlap. Replace either path with matching historical price and news data.

Both positional paths are optional. If omitted, the program opens two file
pickers: first for prices and then optionally for news. Cancel the news picker to
train using price data only.

```bash
venv/bin/python code/improved_lstm.py --date-col Date
```

Common options include:

```text
--date-col NAME              Price timestamp column (use Date for bundled data)
--close-col NAME             Close-price column (default: Close)
--news-date-col NAME         Override automatic news-date detection
--news-impact-col NAME       Override automatic impact-column detection
--window N                   Input sequence length (default: 60)
--forecast-steps N           Direct forecast horizon (default: 24)
--split-ratio FLOAT          Train/validation share (default: 0.80)
--validation-ratio FLOAT     Validation share within training data (default: 0.20)
--epochs N                   Maximum epochs (default: 150)
--batch-size N               Batch size (default: 32)
--interval-minutes FLOAT     Future candle spacing; inferred when omitted
--show                       Display the saved plot after training (default)
--no-show                    Save the plot without displaying it
```

Run `venv/bin/python code/improved_lstm.py --help` for every option.

The pipeline predicts a complete horizon of log returns, fits scalers only on
the training partition, uses a chronological validation/test split, compares
one-step test forecasts with a last-price baseline, and selects return
shrinkage using validation data only.

### Generated outputs

Training creates these ignored, local artifacts:

```text
models/
  lstm_<asset>.h5                 trained Keras model
  lstm_<asset>_preprocessing.pkl  scalers, feature names, and configuration
  lstm_<asset>_metrics.json       validation/test metadata and metrics
plots/<asset>/
  <date>_LSTM.png                 forecast chart
  <date>_LSTM.pkl                 serialized Matplotlib figure
```

GRU runs use `gru`/`GRU` in the corresponding names. Reported metrics include
MSE, MAE, RMSE, MAPE, directional accuracy, and skill versus the naive
last-price baseline.

## Run the Django dashboard

The dashboard is a separate, original demonstration pipeline. Start it from the
`mysite` directory so its relative dataset paths resolve correctly:

```bash
source venv/bin/activate
cd mysite
python manage.py migrate
python manage.py runserver
```

Open <http://127.0.0.1:8000/> and select a symbol. The first prediction request
may train a model and can take some time.

The Django code is configured for local development (`DEBUG=True`) and must not
be deployed publicly without production settings, a private secret key, and
appropriate allowed hosts. The legacy model-save fallback in
`dashboard/views.py` also contains a machine-specific Windows path; change it
to a valid local path before relying on automatic model persistence.

## Project layout

```text
code/                           training and utility scripts
data/news/                      sample news input
models/                         generated models and metrics (ignored)
plots/                          generated forecast plots (ignored)
mysite/                         Django project
  dashboard/data/new/           bundled market-price CSV files
requirements.txt                pinned Python dependencies
```

## Known limitations

- Forecast quality depends heavily on data coverage, timestamp consistency,
  asset-specific news scoring, and hyperparameter selection.
- Future forecasts assume no new news after the final observed candle.
- The improved command-line models are not automatically loaded by the legacy
  Django prediction view; the two workflows currently save and consume models
  in different formats and locations.
- Several scripts in `code/` are experiments or older utilities. For new
  training runs, prefer `improved_lstm.py` or `improved_gru.py`.

## License

No license file is currently included. Add one before redistributing or
accepting external contributions.
