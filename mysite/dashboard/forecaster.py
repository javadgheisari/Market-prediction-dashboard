import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.data_paths import DATA_DIR, DATE_COLUMNS, resolve_data_csv

CODE_DIR = Path(__file__).resolve().parents[2] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

MODEL_KINDS = {
    "lstm": "LSTM",
    "gru": "GRU",
}


def resolve_date_column(path: Path) -> str:
    columns = pd.read_csv(path, nrows=0).columns
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


def build_plotly_chart(metrics: dict[str, Any]) -> dict[str, Any]:
    chart = metrics["chart"]
    return {
        **chart,
        "metrics": {
            "mae": metrics["model_metrics"]["mae"],
            "rmse": metrics["model_metrics"]["rmse"],
            "directional_accuracy_percent": metrics["model_metrics"][
                "directional_accuracy_percent"
            ],
            "bullish_percent": metrics["forecast_direction_probability"][
                "bullish_percent"
            ],
            "bearish_percent": metrics["forecast_direction_probability"][
                "bearish_percent"
            ],
            "news_enabled": metrics["news"] is not None,
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

    # TensorFlow is intentionally imported only when prediction is requested,
    # keeping the dashboard and development server quick to start.
    from sequence_forecaster import ForecastConfig, train_and_forecast

    price_csv = price_csv or resolve_data_csv(symbol)
    news_csv = resolve_news_csv(symbol) if use_news else None
    if use_news and news_csv is None:
        raise FileNotFoundError(
            f"No crawled news file exists for {symbol}. Click 'Crawl latest "
            f"{symbol} news' first, then train again with news."
        )
    config = replace(
        ForecastConfig(epochs=50, patience=10),
        forecast_steps=predict_steps,
    )
    metrics = train_and_forecast(
        kind=MODEL_KINDS[model_name],
        csv_path=price_csv,
        date_col=resolve_date_column(price_csv),
        close_col="Close",
        config=config,
        news_csv_path=news_csv,
        show_plot=False,
        save_plots=False,
    )
    return build_plotly_chart(metrics)
