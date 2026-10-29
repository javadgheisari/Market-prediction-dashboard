import sys
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.data_paths import DATA_DIR, DATE_COLUMNS, resolve_data_csv
from dashboard.forexsb_service import forexsb_h1_csv
from dashboard.market_data_service import refreshed_price_csv

CODE_DIR = Path(__file__).resolve().parents[2] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

MODEL_KINDS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "xgboost": "XGBOOST_PORTFOLIO",
    "chronos": "CHRONOS_2",
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
NEWS_PRICE_CONTEXT_ROWS = 60


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


def _typical_candle_interval(values: list[str]) -> str | None:
    """Describe the usual spacing without letting market closures dominate it."""
    if len(values) < 4:
        return None
    dates = pd.to_datetime(values[-201:], errors="coerce", utc=True)
    differences = pd.Series(dates).dropna().drop_duplicates().sort_values().diff()
    minutes = differences.dropna().dt.total_seconds().div(60)
    minutes = minutes[minutes > 0]
    if len(minutes) < 3:
        return None
    typical = int(round(minutes.median()))
    if typical < 1:
        return None
    for unit_minutes, unit_name in ((1440, "day"), (60, "hour")):
        if typical >= unit_minutes and typical % unit_minutes == 0:
            count = typical // unit_minutes
            return f"{count} {unit_name}{'' if count == 1 else 's'}"
    return f"{typical} minute{'' if typical == 1 else 's'}"


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
    """Yield news-overlapping prices with pre-news model context.

    Both inputs are interpreted as UTC, matching the dashboard upload contract.
    Keeping one input window before the first article gives the model enough
    price context while cropping still ensures news reaches the training split.
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

    overlap_positions = (price_dates >= first_news).to_numpy().nonzero()[0]
    if not len(overlap_positions):
        raise ValueError(
            "The crawled news begins after the final price candle. Complete the "
            "price CSV with newer candles or crawl news matching its date range."
        )
    first_overlap = int(overlap_positions[0])
    context_start = max(0, first_overlap - NEWS_PRICE_CONTEXT_ROWS)
    overlapping = prices.iloc[context_start:].copy()

    with tempfile.TemporaryDirectory(prefix="market-news-overlap-") as directory:
        cropped_path = Path(directory) / price_csv.name
        overlapping.to_csv(cropped_path, index=False)
        yield cropped_path


def build_plotly_chart(metrics: dict[str, Any]) -> dict[str, Any]:
    chart = metrics["chart"].copy()
    actual_dates = chart.get("actual_dates", [])
    chart["candle_interval"] = _typical_candle_interval(actual_dates)
    for date_key in ("actual_dates", "test_dates", "future_dates"):
        if date_key in chart:
            chart[date_key] = _chart_dates_in_tehran(chart[date_key])
    chart["last_observed_at"] = (
        chart["actual_dates"][-1] if actual_dates else None
    )
    future_dates = chart.get("future_dates", [])
    chart["forecast_ends_at"] = future_dates[-1] if future_dates else None
    chart["timezone"] = CHART_TIMEZONE
    return_scales = metrics.get(
        "validation_selected_return_scales",
        [metrics.get("validation_selected_return_scale", 1.0)],
    )
    report_card = metrics.get("horizon_backtest")
    if report_card is not None:
        report_card = report_card.copy()
        report_card.pop("by_horizon", None)
        for key in (
            "first_origin_at", "last_origin_at", "first_target_at", "last_target_at"
        ):
            if report_card.get(key):
                report_card[key] = _chart_dates_in_tehran([report_card[key]])[0]
        report_card["data_rows"] = metrics.get("rows")
        report_card["test_candles"] = metrics.get("test_samples")
        report_card["data_last_at"] = chart["last_observed_at"]
    return {
        **chart,
        "report_card": report_card,
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
            "diagnostic_label": (
                "Single-origin path diagnostics"
                if metrics.get("model") == "CHRONOS_2"
                else "One-candle walk-forward diagnostics"
            ),
        },
    }


def run_dashboard_forecast(
    symbol: str,
    model_name: str,
    predict_steps: int,
    price_csv: Path | None,
    use_news: bool,
    price_source: str = "bundled",
    advanced_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the selected improved model and return JSON-safe chart data."""
    return run_dashboard_forecasts(
        symbol=symbol,
        model_names=(model_name,),
        predict_steps=predict_steps,
        price_csv=price_csv,
        use_news=use_news,
        price_source=price_source,
        advanced_config=advanced_config,
    )[model_name]


def run_dashboard_forecasts(
    symbol: str,
    model_names: tuple[str, ...],
    predict_steps: int,
    price_csv: Path | None,
    use_news: bool,
    price_source: str = "bundled",
    advanced_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Train multiple models against one shared, refreshed data snapshot."""
    if not model_names:
        raise ValueError("Choose at least one forecast model.")
    unsupported = [name for name in model_names if name not in MODEL_KINDS]
    if unsupported:
        raise ValueError(f"Unsupported model '{unsupported[0]}'.")

    if price_source == "forexsb":
        source_context = forexsb_h1_csv(symbol)
    else:
        source_context = nullcontext(price_csv or resolve_data_csv(symbol))
    news_csv = resolve_news_csv(symbol) if use_news else None
    if use_news and news_csv is None:
        raise FileNotFoundError(
            f"No crawled news file exists for {symbol}. Click 'Crawl latest "
            f"{symbol} news' first, then train again with news."
        )

    config = advanced_config or {}
    charts = {}
    with source_context as source_price_csv:
        with refreshed_price_csv(source_price_csv, symbol) as normalized_price_csv:
            price_context = (
                news_overlapping_price_csv(normalized_price_csv, news_csv)
                if news_csv is not None
                else nullcontext(normalized_price_csv)
            )
            with price_context as training_price_csv:
                for model_name in model_names:
                    charts[model_name] = build_plotly_chart(
                        _train_model(
                            model_name=model_name,
                            training_price_csv=training_price_csv,
                            predict_steps=predict_steps,
                            news_csv=news_csv,
                            advanced_config=config,
                        )
                    )
    return charts


def _train_model(
    model_name: str,
    training_price_csv: Path,
    predict_steps: int,
    news_csv: Path | None,
    advanced_config: dict[str, Any],
) -> dict[str, Any]:
    """Run a model against an already downloaded and refreshed price file."""
    if model_name == "xgboost":
        from xgboost_forecaster import train_and_forecast_xgboost

        return train_and_forecast_xgboost(
            csv_path=training_price_csv,
            date_col=resolve_date_column(training_price_csv),
            close_col="Close",
            forecast_steps=predict_steps,
            news_csv_path=news_csv,
            window=int(advanced_config.get("window", 60)),
            split_ratio=float(advanced_config.get("split_ratio", 0.8)),
            validation_ratio=float(advanced_config.get("validation_ratio", 0.2)),
            portfolio_size=int(advanced_config.get("portfolio_size", 3)),
            recency_half_life=float(
                advanced_config.get("recency_half_life", 0.5)
            ),
            n_estimators=int(advanced_config.get("n_estimators", 300)),
            max_depth=int(advanced_config.get("max_depth", 3)),
            learning_rate=float(advanced_config.get("learning_rate", 0.03)),
        )
    if model_name == "chronos":
        from chronos_challenger import train_and_forecast_chronos

        return train_and_forecast_chronos(
            csv_path=training_price_csv,
            date_col=resolve_date_column(training_price_csv),
            close_col="Close",
            forecast_steps=predict_steps,
            news_csv_path=news_csv,
            split_ratio=float(advanced_config.get("split_ratio", 0.8)),
            context_length=int(
                advanced_config.get("chronos_context_length", 512)
            ),
            model_id=str(
                advanced_config.get("chronos_model_id", "amazon/chronos-2")
            ),
            device_map=str(advanced_config.get("chronos_device", "cpu")),
        )
    else:
        from sequence_forecaster import ForecastConfig, train_and_forecast

        epochs = int(advanced_config.get("epochs", 50))
        config = replace(
            ForecastConfig(
                window=int(advanced_config.get("window", 60)),
                split_ratio=float(advanced_config.get("split_ratio", 0.8)),
                validation_ratio=float(
                    advanced_config.get("validation_ratio", 0.2)
                ),
                epochs=epochs,
                patience=max(5, min(15, epochs // 3)),
            ),
            forecast_steps=predict_steps,
        )
        return train_and_forecast(
            kind=MODEL_KINDS[model_name],
            csv_path=training_price_csv,
            date_col=resolve_date_column(training_price_csv),
            close_col="Close",
            config=config,
            news_csv_path=news_csv,
            show_plot=False,
            save_plots=False,
        )
