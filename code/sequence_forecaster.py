"""Shared training pipeline for the improved GRU and LSTM scripts.

The model predicts future log returns instead of absolute price levels. Returns
are more stable across price regimes, and predicting the whole horizon directly
avoids recursively feeding predictions back into the network.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM, LayerNormalization
from tensorflow.keras.optimizers import Adam


ModelKind = Literal["GRU", "LSTM"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEWS_DATE_COLUMNS = (
    "published_at",
    "published",
    "datetime",
    "timestamp",
    "date",
    "Date",
    "time",
)
NEWS_TEXT_COLUMNS = (
    "headline",
    "title",
    "summary",
    "description",
    "body",
    "content",
    "text",
)
NEWS_IMPACT_COLUMNS = (
    "asset_impact",
    "impact_score",
    "gold_impact",
    "sentiment_score",
    "sentiment",
)

# These phrases describe likely pressure on gold, not the emotional tone of an
# article. For other assets, supplying an explicit asset_impact column is much
# more reliable than this fallback scorer.
GOLD_BULLISH_PHRASES = {
    "rate cut": 2.0,
    "dovish": 1.5,
    "lower yields": 1.5,
    "yields fall": 1.5,
    "dollar weakens": 1.5,
    "weak dollar": 1.5,
    "geopolitical tension": 1.5,
    "geopolitical risk": 1.5,
    "war": 1.0,
    "conflict escalates": 1.5,
    "safe haven": 1.5,
    "inflation rises": 1.2,
    "inflation fears": 1.2,
    "central bank buying": 2.0,
    "gold purchases": 1.5,
    "recession fears": 1.5,
    "economic uncertainty": 1.0,
    "banking crisis": 1.5,
}
GOLD_BEARISH_PHRASES = {
    "rate hike": 2.0,
    "hawkish": 1.5,
    "higher yields": 1.5,
    "yields rise": 1.5,
    "dollar strengthens": 1.5,
    "strong dollar": 1.5,
    "ceasefire": 1.2,
    "de-escalation": 1.2,
    "risk appetite": 1.0,
    "inflation cools": 1.2,
    "central bank selling": 2.0,
    "strong jobs": 1.2,
    "strong economy": 1.0,
}


@dataclass(frozen=True)
class ForecastConfig:
    window: int = 60
    forecast_steps: int = 24
    split_ratio: float = 0.80
    validation_ratio: float = 0.20
    units: int = 64
    dropout: float = 0.15
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 1e-3
    patience: int = 15
    seed: int = 42
    news_half_life_hours: float = 24.0
    news_lookback_hours: float = 168.0


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow."""
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def parse_datetimes(values: pd.Series) -> pd.Series:
    """Parse ISO timestamps and the broker's day-first timezone format."""
    try:
        return pd.to_datetime(
            values, format="mixed", dayfirst=True, errors="coerce"
        )
    except (TypeError, ValueError):
        return pd.to_datetime(values, dayfirst=True, errors="coerce")


def load_market_data(path: Path, date_col: str, close_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {date_col, close_col}.difference(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing {sorted(missing)}. Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[date_col] = parse_datetimes(df[date_col])
    numeric_candidates = {"Open", "High", "Low", "Close", "Volume", close_col}
    for column in numeric_candidates.intersection(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=[date_col, close_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[date_col], keep="last")
        .reset_index(drop=True)
    )
    df = df[df[close_col] > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("The CSV contains no valid, positive close prices.")
    return df


def resolve_column(
    df: pd.DataFrame,
    requested: str | None,
    candidates: tuple[str, ...],
    description: str,
) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"News CSV has no '{requested}' {description} column. "
                f"Available columns: {list(df.columns)}"
            )
        return requested
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not detect the news {description} column. Expected one of "
        f"{list(candidates)}; available columns: {list(df.columns)}"
    )


def score_gold_news(text: str) -> float:
    """Return a bounded estimate of an article's directional impact on gold."""
    normalized = " ".join(text.lower().replace("-", " ").split())
    bullish = sum(
        weight for phrase, weight in GOLD_BULLISH_PHRASES.items() if phrase in normalized
    )
    bearish = sum(
        weight for phrase, weight in GOLD_BEARISH_PHRASES.items() if phrase in normalized
    )
    return float(np.tanh((bullish - bearish) / 3.0))


def parse_impact_value(value: Any) -> float:
    labels = {
        "very bullish": 1.0,
        "bullish": 0.7,
        "positive": 0.7,
        "up": 0.7,
        "neutral": 0.0,
        "mixed": 0.0,
        "bearish": -0.7,
        "negative": -0.7,
        "down": -0.7,
        "very bearish": -1.0,
    }
    if pd.isna(value):
        return 0.0
    label = str(value).strip().lower()
    if label in labels:
        return labels[label]
    try:
        return float(np.clip(float(value), -1.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def align_news_timezone(
    news_dates: pd.Series, market_dates: pd.Series
) -> pd.Series:
    """Put news timestamps in the market timestamp's timezone representation."""
    market_timezone = market_dates.dt.tz
    news_timezone = news_dates.dt.tz
    if market_timezone is None and news_timezone is not None:
        return news_dates.dt.tz_localize(None)
    if market_timezone is not None and news_timezone is None:
        return news_dates.dt.tz_localize(
            market_timezone, ambiguous="NaT", nonexistent="shift_forward"
        )
    if market_timezone is not None and news_timezone is not None:
        return news_dates.dt.tz_convert(market_timezone)
    return news_dates


def load_news_data(
    path: Path,
    market_dates: pd.Series,
    asset_name: str,
    date_col: str | None = None,
    impact_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load news and produce publication time, impact, and confidence columns."""
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("The selected news CSV is empty.")
    resolved_date_col = resolve_column(raw, date_col, NEWS_DATE_COLUMNS, "date")
    original_dates = raw[resolved_date_col].astype(str)
    parsed_dates = parse_datetimes(raw[resolved_date_col])
    parsed_dates = align_news_timezone(parsed_dates, market_dates)

    # A date without a publication time is only considered known on the next
    # day. This conservative rule prevents a daily summary from leaking news
    # published later that same day into earlier price candles.
    date_only = ~original_dates.str.contains(r"\d{1,2}:\d{2}", regex=True, na=False)
    delayed_rows = date_only & parsed_dates.notna()
    if delayed_rows.any():
        parsed_dates = parsed_dates.copy()
        parsed_dates.loc[delayed_rows] = (
            parsed_dates.loc[delayed_rows] + pd.Timedelta(days=1)
        )

    resolved_impact_col: str | None = None
    for candidate in ((impact_col,) if impact_col else NEWS_IMPACT_COLUMNS):
        if candidate and candidate in raw.columns:
            resolved_impact_col = candidate
            break

    text_columns = [column for column in NEWS_TEXT_COLUMNS if column in raw.columns]
    if resolved_impact_col:
        impacts = raw[resolved_impact_col].map(parse_impact_value).astype(float)
        scoring_method = f"explicit:{resolved_impact_col}"
    else:
        is_gold = "gold" in asset_name.lower() or "xau" in asset_name.lower()
        if not is_gold:
            raise ValueError(
                "News for non-gold assets needs an asset_impact/impact_score column "
                "with values from -1 (bearish) to +1 (bullish)."
            )
        if not text_columns:
            raise ValueError(
                "News CSV needs a headline/title/body/content column when no "
                "explicit impact score is provided."
            )
        combined_text = raw[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
        impacts = combined_text.map(score_gold_news)
        scoring_method = "gold_phrase_fallback"

    quality = pd.Series(1.0, index=raw.index)
    quality_columns: list[str] = []
    for column in ("relevance", "importance", "source_quality", "confidence"):
        if column in raw.columns:
            values = pd.to_numeric(raw[column], errors="coerce").fillna(1.0)
            quality *= values.clip(0.0, 1.0)
            quality_columns.append(column)

    news = pd.DataFrame(
        {
            "published_at": parsed_dates,
            "impact": impacts.clip(-1.0, 1.0),
            "quality": quality.clip(0.0, 1.0),
        }
    )
    if text_columns:
        news["deduplication_text"] = (
            raw[text_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        )
    news = news.dropna(subset=["published_at"])
    if "deduplication_text" in news:
        news = news.drop_duplicates(
            subset=["published_at", "deduplication_text"], keep="last"
        )
    news = news.sort_values("published_at").reset_index(drop=True)
    if news.empty:
        raise ValueError("The news CSV contains no valid publication dates.")

    metadata: dict[str, Any] = {
        "path": str(path),
        "rows_loaded": len(news),
        "date_column": resolved_date_col,
        "impact_column": resolved_impact_col,
        "text_columns": text_columns,
        "quality_columns": quality_columns,
        "scoring_method": scoring_method,
        "date_only_rows_delayed_to_next_day": int(date_only.sum()),
    }
    return news[["published_at", "impact", "quality"]], metadata


def build_news_features(
    market_dates: pd.Series,
    news: pd.DataFrame,
    half_life_hours: float,
    lookback_hours: float,
) -> pd.DataFrame:
    """Align decayed news signals to candles using publication-time cutoffs."""
    if half_life_hours <= 0 or lookback_hours <= 0:
        raise ValueError("News half-life and lookback must be positive.")

    market_ns = market_dates.array.asi8
    news_ns = news["published_at"].array.asi8
    impacts = news["impact"].to_numpy(dtype=float)
    qualities = news["quality"].to_numpy(dtype=float)
    hour_ns = float(pd.Timedelta(hours=1).value)
    lookback_ns = int(pd.Timedelta(hours=lookback_hours).value)
    day_ns = int(pd.Timedelta(hours=24).value)

    rows: list[tuple[float, float, float, float, float]] = []
    for timestamp in market_ns:
        end = int(np.searchsorted(news_ns, timestamp, side="right"))
        start = int(np.searchsorted(news_ns, timestamp - lookback_ns, side="left"))
        day_start = int(np.searchsorted(news_ns, timestamp - day_ns, side="left"))
        if start == end:
            rows.append((0.0, 0.0, 0.0, 0.0, float(end - day_start)))
            continue

        ages = (timestamp - news_ns[start:end]) / hour_ns
        decay = np.power(0.5, ages / half_life_hours)
        weights = decay * qualities[start:end]
        weight_sum = float(weights.sum())
        if weight_sum <= np.finfo(float).eps:
            rows.append((0.0, 0.0, 0.0, 0.0, float(end - day_start)))
            continue

        recent_impacts = impacts[start:end]
        mean_impact = float(np.sum(weights * recent_impacts) / weight_sum)
        disagreement = float(
            np.sqrt(np.sum(weights * (recent_impacts - mean_impact) ** 2) / weight_sum)
        )
        bullish = float(np.sum(weights * np.clip(recent_impacts, 0.0, None)))
        bearish = float(np.sum(weights * np.clip(-recent_impacts, 0.0, None)))
        rows.append(
            (mean_impact, bullish, bearish, disagreement, float(end - day_start))
        )

    return pd.DataFrame(
        rows,
        columns=[
            "news_impact",
            "news_bullish_attention",
            "news_bearish_attention",
            "news_disagreement",
            "news_count_24h",
        ],
        index=market_dates.index,
    )


def build_features(
    df: pd.DataFrame,
    date_col: str,
    close_col: str,
    news_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create causal features; every row uses only that candle and earlier data."""
    close = df[close_col].astype(float)
    log_close = np.log(close)
    returns = log_close.diff()

    features = pd.DataFrame(index=df.index)
    features["return_1"] = returns
    features["return_3"] = log_close.diff(3)
    features["return_10"] = log_close.diff(10)
    features["volatility_10"] = returns.rolling(10, min_periods=2).std()
    features["volatility_30"] = returns.rolling(30, min_periods=2).std()

    if "Open" in df:
        open_price = df["Open"].where(df["Open"] > 0)
        features["candle_body"] = np.log(close / open_price)
    if {"High", "Low"}.issubset(df.columns):
        price_range = (df["High"] - df["Low"]).clip(lower=0)
        features["high_low_range"] = price_range / close
        features["close_position"] = (
            (close - df["Low"]) / price_range.replace(0, np.nan) - 0.5
        )
    if "Volume" in df:
        volume = df["Volume"].clip(lower=0)
        features["log_volume"] = np.log1p(volume)
        features["volume_change"] = np.log1p(volume).diff()

    dates = df[date_col]
    minutes = dates.dt.hour * 60 + dates.dt.minute
    features["time_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    features["time_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    features["weekday_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7.0)
    features["weekday_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7.0)

    if news_features is not None:
        features = features.join(news_features)

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    window: int,
    horizon: int,
    target_start: int,
    target_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build samples for target indexes in ``[target_start, target_stop)``."""
    starts = range(max(window, target_start), target_stop)
    x_values = [features[i - window : i] for i in starts]
    y_values = [targets[i : i + horizon, 0] for i in starts]
    if not x_values:
        raise ValueError("Not enough rows to create sequences with this configuration.")
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def build_model(
    kind: ModelKind,
    window: int,
    feature_count: int,
    horizon: int,
    units: int,
    dropout: float,
    learning_rate: float,
) -> tf.keras.Model:
    recurrent_layer = GRU if kind == "GRU" else LSTM
    second_units = max(16, units // 2)
    model = Sequential(
        [
            recurrent_layer(
                units,
                return_sequences=True,
                input_shape=(window, feature_count),
            ),
            LayerNormalization(),
            Dropout(dropout),
            recurrent_layer(second_units),
            LayerNormalization(),
            Dropout(dropout),
            Dense(
                second_units,
                activation="swish",
                kernel_regularizer=regularizers.l2(1e-5),
            ),
            Dense(horizon),
        ],
        name=f"improved_{kind.lower()}",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    nonzero = np.abs(actual) > np.finfo(float).eps
    return {
        "mse": float(mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_percent": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100),
    }


def calibrate_return_scale(
    actual_close: np.ndarray,
    previous_close: np.ndarray,
    predicted_returns: np.ndarray,
) -> float:
    """Select validation-only shrinkage toward the last-price baseline.

    Short-horizon financial returns are noisy. This calibration prevents a weak
    model from making unnecessarily large bets while retaining useful signals
    when the validation data supports them.
    """
    candidates = np.linspace(0.0, 1.0, 51)
    errors = [
        mean_absolute_error(
            actual_close,
            previous_close * np.exp(predicted_returns * candidate),
        )
        for candidate in candidates
    ]
    return float(candidates[int(np.argmin(errors))])


def infer_interval(dates: pd.Series) -> pd.Timedelta:
    differences = dates.sort_values().diff().dropna()
    positive = differences[differences > pd.Timedelta(0)]
    if positive.empty:
        raise ValueError("Cannot infer the candle interval from the timestamp column.")
    return positive.median()


def extract_asset_symbol(coin_name: str) -> str:
    """Extract asset symbol (e.g., BTC-USD) from full coin name.
    
    Examples:
        "BTC-USD_1Hour_BID_2026-08-01_to_2026-09-01_Asia_Tehran" -> "BTC-USD"
        "EUR-USD_Minute_2025-10-22_Asia_Tehran" -> "EUR-USD"
    """
    return coin_name.split("_")[0]


def save_outputs(
    kind: ModelKind,
    coin_name: str,
    model: tf.keras.Model,
    feature_scaler: RobustScaler,
    target_scaler: StandardScaler,
    return_scale: float,
    config: ForecastConfig,
    feature_names: list[str],
    news_metadata: dict[str, Any] | None,
    news_features: pd.DataFrame | None,
    metrics: dict[str, Any],
    dates: pd.Series,
    close: np.ndarray,
    split_index: int,
    test_predictions: np.ndarray,
    future_dates: pd.DatetimeIndex,
    future_predictions: np.ndarray,
    show_plot: bool,
) -> None:
    model_dir = PROJECT_ROOT / "models"
    asset_symbol = extract_asset_symbol(coin_name)
    stamp = datetime.now().strftime("%Y-%m-%d")
    plot_dir = PROJECT_ROOT / "plots" / asset_symbol / stamp
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{kind.lower()}_{coin_name}.h5"
    preprocessing_path = model_dir / f"{kind.lower()}_{coin_name}_preprocessing.pkl"
    metrics_path = model_dir / f"{kind.lower()}_{coin_name}_metrics.json"
    model.save(model_path)
    with preprocessing_path.open("wb") as output:
        pickle.dump(
            {
                "feature_scaler": feature_scaler,
                "target_scaler": target_scaler,
                "return_scale": return_scale,
                "feature_names": feature_names,
                "news_metadata": news_metadata,
                "config": asdict(config),
            },
            output,
        )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if news_features is not None:
        fig, (ax, news_ax) = plt.subplots(
            2,
            1,
            figsize=(16, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(16, 7))
        news_ax = None
    context_start = max(0, split_index - config.window)
    model_label = f"{kind} + news" if news_features is not None else kind
    ax.plot(
        dates.iloc[context_start:],
        close[context_start:],
        color="seagreen",
        linewidth=1.3,
        label=f"Actual {coin_name}",
    )
    ax.plot(
        dates.iloc[split_index:],
        test_predictions,
        color="royalblue",
        linewidth=1.2,
        label=f"{model_label} one-step test forecast",
    )
    ax.plot(
        future_dates,
        future_predictions,
        color="darkorange",
        marker="o",
        markersize=3,
        label=f"{model_label} future forecast (no new news assumed)",
    )
    ax.axvline(dates.iloc[split_index], color="gray", linestyle="--", alpha=0.7)
    ax.set_title(f"{coin_name} price forecast using {model_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend()
    if news_ax is not None and news_features is not None:
        sentiment = news_features["news_impact"].iloc[context_start:]
        sentiment_dates = dates.iloc[context_start:]
        news_ax.plot(sentiment_dates, sentiment, color="slateblue", linewidth=1.0)
        news_ax.fill_between(
            sentiment_dates,
            0,
            sentiment,
            where=sentiment >= 0,
            color="seagreen",
            alpha=0.3,
            label="Bullish asset impact",
        )
        news_ax.fill_between(
            sentiment_dates,
            0,
            sentiment,
            where=sentiment < 0,
            color="firebrick",
            alpha=0.3,
            label="Bearish asset impact",
        )
        news_ax.axhline(0, color="gray", linewidth=0.8)
        news_ax.set_ylabel("News impact")
        news_ax.set_ylim(-1.05, 1.05)
        news_ax.grid(alpha=0.2)
        news_ax.legend(loc="upper left", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()

    png_path = plot_dir / f"{stamp}_{kind}.png"
    pickle_path = plot_dir / f"{stamp}_{kind}.pkl"
    fig.savefig(png_path, dpi=150)
    with pickle_path.open("wb") as output:
        pickle.dump(fig, output)
    if show_plot:
        plt.show()
    plt.close(fig)

    print(f"Model saved to: {model_path}")
    print(f"Preprocessing saved to: {preprocessing_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Plot saved to: {png_path}")


def choose_csv_file(title: str, *, required: bool = True) -> Path | None:
    try:
        from easygui import fileopenbox
    except ImportError as exc:
        raise ValueError("Pass a CSV path on the command line (easygui is unavailable).") from exc
    selected = fileopenbox(title=title, filetypes=["*.csv"])
    if not selected:
        if required:
            raise ValueError("No CSV file was selected.")
        return None
    return Path(selected)


def train_and_forecast(
    kind: ModelKind,
    csv_path: Path,
    date_col: str,
    close_col: str,
    config: ForecastConfig,
    news_csv_path: Path | None = None,
    news_date_col: str | None = None,
    news_impact_col: str | None = None,
    interval_minutes: float | None = None,
    show_plot: bool = False,
) -> dict[str, Any]:
    if not 0.5 <= config.split_ratio < 1:
        raise ValueError("split_ratio must be in [0.5, 1.0).")
    if not 0 < config.validation_ratio < 0.5:
        raise ValueError("validation_ratio must be in (0, 0.5).")
    if min(config.window, config.forecast_steps, config.batch_size, config.epochs) < 1:
        raise ValueError("window, forecast_steps, batch_size, and epochs must be positive.")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    if config.news_half_life_hours <= 0 or config.news_lookback_hours <= 0:
        raise ValueError("News half-life and lookback must be positive.")

    set_reproducible_seed(config.seed)
    df = load_market_data(csv_path, date_col, close_col)
    news_metadata: dict[str, Any] | None = None
    news_features: pd.DataFrame | None = None
    if news_csv_path is not None:
        news, news_metadata = load_news_data(
            path=news_csv_path,
            market_dates=df[date_col],
            asset_name=csv_path.stem,
            date_col=news_date_col,
            impact_col=news_impact_col,
        )
        news_features = build_news_features(
            market_dates=df[date_col],
            news=news,
            half_life_hours=config.news_half_life_hours,
            lookback_hours=config.news_lookback_hours,
        )
        news_metadata["feature_coverage_percent"] = float(
            np.mean(news_features.abs().sum(axis=1) > 0) * 100
        )
        if news_metadata["feature_coverage_percent"] == 0:
            raise ValueError(
                "No news falls within the market data period and configured news "
                "lookback. Check the two files' dates and timezones."
            )
        print(
            f"Loaded {len(news)} news rows using "
            f"{news_metadata['scoring_method']} scoring"
        )
    features = build_features(df, date_col, close_col, news_features)
    close = df[close_col].to_numpy(dtype=np.float64)
    log_returns = np.diff(np.log(close), prepend=np.log(close[0])).reshape(-1, 1)

    split_index = int(len(df) * config.split_ratio)
    validation_start = int(split_index * (1 - config.validation_ratio))
    minimum_validation_start = config.window + config.forecast_steps + 1
    if validation_start < minimum_validation_start:
        raise ValueError(
            "Dataset is too small. Use a shorter window/horizon or provide more rows "
            f"(received {len(df)} rows)."
        )
    if split_index + 1 >= len(df):
        raise ValueError("The test partition must contain at least two rows.")
    if news_features is not None and news_metadata is not None:
        training_news_coverage = float(
            np.mean(
                news_features.iloc[:validation_start].abs().sum(axis=1) > 0
            )
            * 100
        )
        news_metadata["training_feature_coverage_percent"] = training_news_coverage
        if training_news_coverage == 0:
            raise ValueError(
                "News does not overlap the training partition. Provide historical "
                "news covering the price-training period."
            )

    # Fit preprocessing only on the sub-training partition. Validation and test
    # information therefore cannot influence model selection.
    feature_scaler = RobustScaler(quantile_range=(10.0, 90.0))
    target_scaler = StandardScaler()
    feature_scaler.fit(features.iloc[:validation_start])
    target_scaler.fit(log_returns[1:validation_start])
    scaled_features = feature_scaler.transform(features)
    scaled_targets = target_scaler.transform(log_returns)

    x_train, y_train = make_sequences(
        scaled_features,
        scaled_targets,
        config.window,
        config.forecast_steps,
        config.window,
        validation_start - config.forecast_steps + 1,
    )
    x_validation, y_validation = make_sequences(
        scaled_features,
        scaled_targets,
        config.window,
        config.forecast_steps,
        validation_start,
        split_index - config.forecast_steps + 1,
    )

    model = build_model(
        kind,
        config.window,
        x_train.shape[-1],
        config.forecast_steps,
        config.units,
        config.dropout,
        config.learning_rate,
    )
    print(
        f"Training {kind}: {len(x_train)} train samples, "
        f"{len(x_validation)} validation samples, {x_train.shape[-1]} features"
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=config.epochs,
        batch_size=min(config.batch_size, len(x_train)),
        shuffle=True,
        callbacks=[
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                min_delta=1e-5,
                restore_best_weights=True,
                verbose=1,
            ),
        ],
        verbose=1,
    )

    validation_scaled = model.predict(
        x_validation, batch_size=config.batch_size, verbose=0
    )
    validation_returns = target_scaler.inverse_transform(
        validation_scaled[:, :1]
    ).ravel()
    validation_indexes = np.arange(
        validation_start, validation_start + len(x_validation)
    )
    return_scale = calibrate_return_scale(
        actual_close=close[validation_indexes],
        previous_close=close[validation_indexes - 1],
        predicted_returns=validation_returns,
    )
    print(f"Validation-selected return scale: {return_scale:.2f}")

    # Walk-forward evaluation: every prediction uses actual candles observed
    # before its timestamp, never the candle it is predicting.
    x_test = np.asarray(
        [scaled_features[i - config.window : i] for i in range(split_index, len(df))],
        dtype=np.float32,
    )
    predicted_scaled = model.predict(x_test, batch_size=config.batch_size, verbose=0)
    raw_predicted_returns = target_scaler.inverse_transform(
        predicted_scaled[:, :1]
    ).ravel()
    predicted_returns = raw_predicted_returns * return_scale
    previous_close = close[split_index - 1 : len(df) - 1]
    test_predictions = previous_close * np.exp(predicted_returns)
    actual_test = close[split_index:]
    baseline_predictions = previous_close

    model_metrics = calculate_metrics(actual_test, test_predictions)
    raw_model_metrics = calculate_metrics(
        actual_test, previous_close * np.exp(raw_predicted_returns)
    )
    baseline_metrics = calculate_metrics(actual_test, baseline_predictions)
    actual_direction = np.sign(actual_test - previous_close)
    predicted_direction = np.sign(test_predictions - previous_close)
    model_metrics["directional_accuracy_percent"] = float(
        np.mean(actual_direction == predicted_direction) * 100
    )
    baseline_mae = baseline_metrics["mae"]
    model_metrics["mae_skill_vs_naive_percent"] = float(
        (1 - model_metrics["mae"] / baseline_mae) * 100
        if baseline_mae > 0
        else 0.0
    )
    metrics: dict[str, Any] = {
        "model": kind,
        "rows": len(df),
        "train_samples": len(x_train),
        "validation_samples": len(x_validation),
        "test_samples": len(actual_test),
        "news": news_metadata,
        "validation_selected_return_scale": return_scale,
        "model_metrics": model_metrics,
        "raw_unscaled_model_metrics": raw_model_metrics,
        "naive_last_price_baseline": baseline_metrics,
    }
    print(json.dumps(metrics, indent=2))

    final_window = scaled_features[-config.window :][None, ...].astype(np.float32)
    future_scaled = model.predict(final_window, verbose=0)[0].reshape(-1, 1)
    future_returns = target_scaler.inverse_transform(future_scaled).ravel() * return_scale
    future_predictions = close[-1] * np.exp(np.cumsum(future_returns))
    interval = (
        pd.Timedelta(minutes=interval_minutes)
        if interval_minutes is not None
        else infer_interval(df[date_col])
    )
    future_dates = pd.date_range(
        start=df[date_col].iloc[-1] + interval,
        periods=config.forecast_steps,
        freq=interval,
    )

    save_outputs(
        kind=kind,
        coin_name=csv_path.stem,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        return_scale=return_scale,
        config=config,
        feature_names=list(features.columns),
        news_metadata=news_metadata,
        news_features=news_features,
        metrics=metrics,
        dates=df[date_col],
        close=close,
        split_index=split_index,
        test_predictions=test_predictions,
        future_dates=future_dates,
        future_predictions=future_predictions,
        show_plot=show_plot,
    )
    return metrics


def build_argument_parser(kind: ModelKind) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train an improved {kind} market forecaster")
    parser.add_argument(
        "price_csv", nargs="?", type=Path, help="Market price CSV; opens a dialog if omitted"
    )
    parser.add_argument(
        "news_csv",
        nargs="?",
        type=Path,
        help="Optional asset news CSV; cancel the second dialog to train without news",
    )
    parser.add_argument("--date-col", default="Asia/Tehran")
    parser.add_argument("--close-col", default="Close")
    parser.add_argument(
        "--news-date-col",
        help="News publication-time column; automatically detected when omitted",
    )
    parser.add_argument(
        "--news-impact-col",
        help="Optional -1 to +1 asset-impact column; automatically detected when omitted",
    )
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--forecast-steps", type=int, default=24)
    parser.add_argument("--split-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--news-half-life-hours", type=float, default=24.0)
    parser.add_argument("--news-lookback-hours", type=float, default=168.0)
    parser.add_argument(
        "--interval-minutes",
        type=float,
        help="Future candle interval; inferred from timestamps when omitted",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Display the plot after saving it (default)",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Save the plot without displaying it",
    )
    parser.set_defaults(show=True)
    return parser


def main(kind: ModelKind) -> None:
    args = build_argument_parser(kind).parse_args()
    csv_path = args.price_csv or choose_csv_file(
        "Choose the CSV file containing market prices"
    )
    news_csv_path = args.news_csv or choose_csv_file(
        "Choose an optional CSV file containing timestamped asset news",
        required=False,
    )
    assert csv_path is not None
    if not csv_path.is_file():
        raise FileNotFoundError(f"Price CSV file does not exist: {csv_path}")
    if news_csv_path is not None and not news_csv_path.is_file():
        raise FileNotFoundError(f"News CSV file does not exist: {news_csv_path}")
    config = ForecastConfig(
        window=args.window,
        forecast_steps=args.forecast_steps,
        split_ratio=args.split_ratio,
        validation_ratio=args.validation_ratio,
        units=args.units,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        news_half_life_hours=args.news_half_life_hours,
        news_lookback_hours=args.news_lookback_hours,
    )
    train_and_forecast(
        kind=kind,
        csv_path=csv_path.resolve(),
        date_col=args.date_col,
        close_col=args.close_col,
        config=config,
        news_csv_path=news_csv_path.resolve() if news_csv_path is not None else None,
        news_date_col=args.news_date_col,
        news_impact_col=args.news_impact_col,
        interval_minutes=args.interval_minutes,
        show_plot=args.show,
    )
