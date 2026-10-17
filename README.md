# Market Prediction Dashboard

A research and demonstration project for forecasting Forex, cryptocurrency,
commodity, and US stock prices with LSTM and GRU models plus a
validation-selected XGBoost portfolio. The repository contains:

- an improved, leakage-aware command-line forecasting pipeline that combines
  market candles with timestamped asset news;
- a Django dashboard for selecting an asset and running the improved LSTM or
  GRU pipeline or the XGBoost portfolio, plus a ranked ensemble that combines
  all three models with the independent news signal; and
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
accepted. Optional `relevance`, `importance`, `source_quality`, `confidence`, and
`novelty` columns must contain values from 0 to 1 and are combined with a
geometric-mean quality weight. `event_type` and `impact_horizon_hours` add
event-regime features and event-specific causal decay when available.

For files whose name contains `gold` or `XAU`, the trainer can estimate impact
from headline/body text with a small built-in phrase scorer when no impact
column is supplied. Other assets require an explicit asset-impact column.
Explicit, asset-specific scores are recommended for serious experiments.

Timestamped articles affect only candles at or after publication. A row that
contains only a date is conservatively delayed until the next day to prevent
look-ahead leakage. News impact decays over time according to
`--news-half-life-hours` and is ignored after `--news-lookback-hours`.

The dashboard also includes a **News-only buy decision** model. It does not
load price data. It combines active articles using their asset impact,
relevance, importance, source quality, confidence, novelty, recency, and impact
horizon, then returns either `BUY` or `DO NOT BUY`. A buy requires both a
positive weighted signal and sufficient consistent evidence. This output is an
experimental research signal, not financial advice.

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
`asset_impact` estimate by default. For asset-specific structured scoring, set
an API key and explicitly select the optional OpenAI scorer:

```bash
export OPENAI_API_KEY="..."
python code/news_crawler.py --assets XAU BTC AAPL --days 7 --scorer openai
```

The scorer uses GPT-6 Luna by default and records impact, relevance, importance,
confidence, novelty, expected impact horizon, event type, and scoring method.
Only the symbol, asset description, local article ID, and headline are sent.
Set `OPENAI_NEWS_MODEL` or pass `--openai-model` to change the model. The local
lexical path remains the default and requires no API key or API spend. For the
Django crawl button, opt in with `NEWS_SCORER=openai`; otherwise it stays local.
To use a different RSS service, pass a URL template containing `{query}` via
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

### Train XGBoost

XGBoost uses the same causal market/news features and chronological evaluation,
but trains much faster on the relatively small bundled datasets:

```bash
venv/bin/python code/improved_xgboost.py \
  data/crypto/btc/BTC-USD_1Hour_BID_2026-08-05_to_2026-09-05_Asia_Tehran\ \(1\).csv \
  --date-col Asia/Tehran --forecast-steps 24
```

It is also available as **XGBoost Portfolio (recommended)** in the Django model
selector. The CPU-only package is pinned to avoid unnecessary GPU libraries.

The pipeline trains three differently regularized members, applies smooth
recency weighting, and uses the chronological validation partition to choose a
member, a blend, or the zero-return naive forecast independently for every
horizon. The untouched test partition remains a walk-forward evaluation. This
prevents the dashboard from forcing a weak trained model into the result. Use
`--portfolio-size 1` for the old single-model cost profile, or adjust
`--recency-half-life` after profiling regime sensitivity. Early stopping and
return shrinkage use validation data only; the default single-worker setting
avoids CPU oversubscription in web requests.

### Chronos-2 in the dashboard

Chronos-2 is an optional dashboard model because it adds PyTorch/Transformers
and downloads model weights on first use. Install it after the base environment:

```bash
venv/bin/python -m pip install -r requirements-chronos.txt
```

Then choose **Chronos-2** in the model selector. Its advanced controls select the
checkpoint, context length, device, and chronological split. The dashboard runs
a causal held-out evaluation, displays p10/p50/p90 forecast paths, and reports
whether Chronos beats the last-price baseline. When news is enabled, only news
known at the forecast origin enters the context; future news inputs are zero.

A global Temporal Fusion Transformer remains a later experiment: it should be
trained across sufficiently deep histories for all assets, not fitted
independently to the small per-asset CSVs currently bundled.

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
last-price baseline. Future charts include an 80% interval bootstrapped from
complete validation-horizon residuals. This interval describes historical
model error; it is not a guarantee that future prices will remain inside it.

## Run the Django dashboard

The dashboard uses the same improved pipeline as `improved_lstm.py` and
`improved_gru.py`. Start it from the `mysite` directory:

```bash
source venv/bin/activate
cd mysite
python manage.py migrate
python manage.py runserver
```

Open <http://127.0.0.1:8000/>, select an asset, choose the recommended XGBoost
portfolio, Chronos-2, Improved GRU, or Improved LSTM, and enter the forecast
horizon. Expand **Advanced model configuration** for bounded model-specific
controls; fields that do not apply to the selected model are hidden. One step is one candle interval
in the selected CSV (one hour for the bundled hourly datasets). The request
trains a model, so it can take several minutes on a CPU. The result page shows
actual prices, one-step test predictions, future predictions, error metrics,
directional accuracy, and estimated bullish/bearish probabilities.

The dashboard includes the seven USD FX majors: AUD (`AUDUSD`), CAD (`USDCAD`),
CHF (`USDCHF`), EUR (`EURUSD`), GBP (`GBPUSD`), JPY (`USDJPY`), and NZD
(`NZDUSD`). It also includes gold (XAU), silver (XAG), Brent crude, and WTI
crude (USOIL). FX majors, metals, and Brent can download one-hour ForexSB
history automatically. WTI requires an uploaded historical CSV; BiQuote adds
its latest closed candles before training.

For historical prices, select **Use project /data CSV** to use the bundled file
for that asset, or **Choose CSV from this computer** to upload a CSV of up to
50 MB. Uploaded price files are used temporarily for one training request and
are deleted afterward.

Use **Download CSV** beside any dashboard asset to export its latest completed
one-hour candles. Where available, the export combines deeper ForexSB history
with newer BiQuote candles; other assets use BiQuote's recent H1 history.

Use **Crawl news** beside an asset, or on its forecast page, to fetch the latest
seven days of relevant RSS headlines into `data/news/<SYMBOL>_news.csv`. Existing
rows are retained and deduplicated. Use **Refresh news & rank buys** on a market
card to clear the saved news for every asset in that market, crawl fresh
headlines, and open a confidence-adjusted ranking of the resulting news-only buy
signals. A later training run automatically includes that news file when
**Price + crawled news** is selected and its dates overlap the market dataset.
Select **Price data only** to train while ignoring news.

Use **Build a model + news buy ranking** for the general recommendation flow.
Choose a market, optionally narrow it to specific assets, and use the default
ensemble to combine independent news, XGBoost, GRU, and LSTM evidence. Its four
confidence-adjusted directional signals are weighted equally, and a `BUY`
requires at least three positive components, 55% average confidence, and a
validated model edge. Individual XGBoost, Chronos-2, GRU, LSTM, and news-only
rankings remain available, including the option to refresh news first.
Assets without automatic price history or usable news remain in the table with
an actionable error instead of aborting the rest of the market ranking. Use
**Export ranking CSV** to download the current ranked results, including failed
assets, without rerunning the selected model or refreshing news.

Each asset page lists saved headlines with publication time, source, link, and
the crawler's asset-impact score. Pagination provides access to the complete
saved history, 25 articles at a time.

The Django code is configured for local development (`DEBUG=True`) and must not
be deployed publicly without production settings, a private secret key, and
appropriate allowed hosts.

## Project layout

```text
code/                           training and utility scripts
data/news/                      sample news input
models/                         generated models and metrics (ignored)
plots/                          generated forecast plots (ignored)
mysite/                         Django project
data/                           bundled market-price and news CSV files
requirements.txt                pinned Python dependencies
```

## Known limitations

- Forecast quality depends heavily on data coverage, timestamp consistency,
  asset-specific news scoring, and hyperparameter selection.
- Future forecasts assume no new news after the final observed candle.
- Future timestamps reuse the sessions observed in the input CSV. Supply
  `--interval-minutes` when the inferred candle spacing is not appropriate.
- Dashboard training is synchronous. For a multi-user or production deployment,
  move training into a background job queue and return job progress to the UI.
- Several scripts in `code/` are experiments or older utilities. For new
  training runs, prefer `improved_lstm.py`, `improved_gru.py`, or
  `improved_xgboost.py`.

## License

No license file is currently included. Add one before redistributing or
accepting external contributions.
