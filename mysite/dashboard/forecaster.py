import sys
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.data_paths import DATA_DIR, DATE_COLUMNS, resolve_data_csv
from dashboard.market_data_service import refreshed_price_csv

CODE_DIR = Path(__file__).resolve().parents[2] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

MODEL_KINDS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "xgboost": "XGBOOST",
}

UTC = "UTC"
CHART_TIMEZONE = "Asia/Tehran"
NEWS_DATE_COLUMNS = (
    "published_at",
    "published",
    "datetime",
    "timestamp",
    "date",
    "Date",
    "time",
)


def _chart_dates_in_tehran(values: list[str]) -> list[str]:
    converted = []
    for value in values:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(UTC)
        else:
            timestamp = timestamp.tz_convert(UTC)
        converted.append(
            timestamp.tz_convert(CHART_TIMEZONE).strftime("%Y-%m-%d %H:%M")
        )
    return converted


def resolve_date_column(path: Path) -> str:
    # Uploaded broker "CSV" files may actually be tab- or semicolon-separated.
    columns = pd.read_csv(path, nrows=0, sep=None, engine="python").columns
    for column in DATE_COLUMNS:
        if column in columns:
            return column
    raise ValueError(
        f"No date column found in {path}. Expected one of {DATE_COLUMNS}."
    )


def resolve_news_csv(symbol: str) -> Path | None:
    symbol = symbol.upper()
    candidates = (
        DATA_DIR / "news" / f"{symbol}_news.csv",
        DATA_DIR / "news" / f"{symbol}-USD_news.csv",
    )
    return next((path for path in candidates if path.is_file()), None)


@contextmanager
def news_overlapping_price_csv(price_csv: Path, news_csv: Path):
    """Yield prices beginning at the first valid crawled-news timestamp.

    Both inputs are interpreted as UTC, matching the dashboard upload contract.
    Cropping before the chronological split ensures the training partition can
    actually learn from news instead of placing all available articles in the
    validation or test partitions.
    """
    date_col = resolve_date_column(price_csv)
    prices = pd.read_csv(price_csv)
    news = pd.read_csv(news_csv)
    news_date_col = next(
        (column for column in NEWS_DATE_COLUMNS if column in news.columns), None
    )
    if news_date_col is None:
        raise ValueError(
            f"No publication date column found in {news_csv.name}."
        )

    price_dates = pd.to_datetime(prices[date_col], errors="coerce", utc=True)
    news_dates = pd.to_datetime(news[news_date_col], errors="coerce", utc=True)
    first_news = news_dates.min()
    if pd.isna(first_news):
        raise ValueError("The crawled news file contains no valid publication dates.")

    overlapping = prices.loc[price_dates >= first_news].copy()
    if overlapping.empty:
        raise ValueError(
            "The crawled news begins after the final price candle. Complete the "
            "price CSV with newer candles or crawl news matching its date range."
        )

    with tempfile.TemporaryDirectory(prefix="market-news-overlap-") as directory:
        cropped_path = Path(directory) / price_csv.name
        overlapping.to_csv(cropped_path, index=False)
        yield cropped_path


def build_plotly_chart(metrics: dict[str, Any]) -> dict[str, Any]:
    chart = metrics["chart"].copy()
    for date_key in ("actual_dates", "test_dates", "future_dates"):
        if date_key in chart:
            chart[date_key] = _chart_dates_in_tehran(chart[date_key])
    chart["timezone"] = CHART_TIMEZONE
    return_scales = metrics.get(
        "validation_selected_return_scales",
        [metrics.get("validation_selected_return_scale", 1.0)],
    )
    return {
        **chart,
        "metrics": {
            "mae": metrics["model_metrics"]["mae"],
            "rmse": metrics["model_metrics"]["rmse"],
            "directional_accuracy_percent": metrics["model_metrics"][
                "directional_accuracy_percent"
            ],
            "mae_skill_vs_naive_percent": metrics["model_metrics"].get(
                "mae_skill_vs_naive_percent", 0.0
            ),
            "bullish_percent": metrics["forecast_direction_probability"][
                "bullish_percent"
            ],
            "bearish_percent": metrics["forecast_direction_probability"][
                "bearish_percent"
            ],
            "news_enabled": metrics["news"] is not None,
            "forecast_is_naive": all(scale == 0 for scale in return_scales),
        },
    }


def run_dashboard_forecast(
    symbol: str,
    model_name: str,
    predict_steps: int,
    price_csv: Path | None,
    use_news: bool,
) -> dict[str, Any]:
    """Train the selected improved model and return JSON-safe chart data."""
    if model_name not in MODEL_KINDS:
        raise ValueError(f"Unsupported model '{model_name}'.")

    price_csv = price_csv or resolve_data_csv(symbol)
    news_csv = resolve_news_csv(symbol) if use_news else None
    if use_news and news_csv is None:
        raise FileNotFoundError(
            f"No crawled news file exists for {symbol}. Click 'Crawl latest "
            f"{symbol} news' first, then train again with news."
        )
    with refreshed_price_csv(price_csv, symbol) as normalized_price_csv:
        price_context = (
            news_overlapping_price_csv(normalized_price_csv, news_csv)
            if news_csv is not None
            else nullcontext(normalized_price_csv)
        )
        with price_context as training_price_csv:
            if model_name == "xgboost":
                from xgboost_forecaster import train_and_forecast_xgboost

                metrics = train_and_forecast_xgboost(
                    csv_path=training_price_csv,
                    date_col=resolve_date_column(training_price_csv),
                    close_col="Close",
                    forecast_steps=predict_steps,
                    news_csv_path=news_csv,
                )
            else:
                from sequence_forecaster import ForecastConfig, train_and_forecast

                config = replace(
                    ForecastConfig(epochs=50, patience=10),
                    forecast_steps=predict_steps,
                )
                metrics = train_and_forecast(
                    kind=MODEL_KINDS[model_name],
                    csv_path=training_price_csv,
                    date_col=resolve_date_column(training_price_csv),
                    close_col="Close",
                    config=config,
                    news_csv_path=news_csv,
                    show_plot=False,
                    save_plots=False,
                )
    return build_plotly_chart(metrics)
