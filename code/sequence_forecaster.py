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
from matplotlib.widgets import Button
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
NEWS_EVENT_TYPES = (
    "earnings",
    "macro",
    "monetary_policy",
    "geopolitical",
    "regulation",
    "supply_demand",
    "technical",
    "company",
    "other",
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
    # dateutil treats ``UTC+03:00`` as a POSIX timezone (whose sign is
    # intentionally reversed). Broker exports mean an ordinary positive UTC
    # offset, so normalize that spelling before handing it to pandas.
    normalized = values.astype("string").str.replace(
        r"\s+UTC([+-]\d{2}:?\d{2})$", r"\1", regex=True
    )
    try:
        return pd.to_datetime(
            normalized, format="mixed", dayfirst=True, errors="coerce"
        )
    except (TypeError, ValueError):
        return pd.to_datetime(normalized, dayfirst=True, errors="coerce")


def load_market_data(path: Path, date_col: str, close_col: str) -> pd.DataFrame:
    # Keeping fields as strings initially preserves leading zeroes in the
    # fractional half of malformed decimal-comma volumes such as ``15,007``.
    # Sniff the delimiter because many broker exports keep the .csv extension
    # while separating fields with tabs or semicolons.
    df = pd.read_csv(path, dtype="string", sep=None, engine="python")
    missing = {date_col, close_col}.difference(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing {sorted(missing)}. Available columns: {list(df.columns)}"
        )

    df = df.copy()
    # Some broker exports leave a decimal-comma volume unquoted. A value such
    # as ``15,761`` is parsed as Volume=15 and an unnamed column=761. Restore
    # it as 15.761 before feature engineering.
    unnamed_columns = [
        column for column in df.columns if str(column).startswith("Unnamed:")
    ]
    if "Volume" in df and len(unnamed_columns) == 1:
        fractional_column = unnamed_columns[0]
        whole = pd.to_numeric(df["Volume"], errors="coerce")
        fractional_text = df[fractional_column].astype("string").str.strip()
        fractional = pd.to_numeric(fractional_text, errors="coerce")
        split_volume = whole.notna() & fractional.notna() & fractional.ge(0)
        if split_volume.all():
            denominators = fractional_text.str.len().map(
                lambda width: 10 ** int(width)
            )
            df["Volume"] = whole + fractional / denominators
            df = df.drop(columns=[fractional_column])
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

    quality_components: list[pd.Series] = []
    quality_columns: list[str] = []
    for column in (
        "relevance",
        "importance",
        "source_quality",
        "confidence",
        "novelty",
    ):
        if column in raw.columns:
            values = pd.to_numeric(raw[column], errors="coerce").fillna(1.0)
            quality_components.append(values.clip(0.0, 1.0))
            quality_columns.append(column)
    if quality_components:
        # Avoid shrinking five moderate scores almost to zero as a raw product.
        component_frame = pd.concat(quality_components, axis=1)
        quality = np.exp(np.log(component_frame.clip(lower=1e-6)).mean(axis=1))
        quality[(component_frame == 0).any(axis=1)] = 0.0
    else:
        quality = pd.Series(1.0, index=raw.index)

    impact_horizons = pd.to_numeric(
        raw.get("impact_horizon_hours", pd.Series(np.nan, index=raw.index)),
        errors="coerce",
    ).clip(1.0, 720.0)
    event_types = raw.get("event_type", pd.Series("other", index=raw.index))
    event_types = event_types.fillna("other").astype(str).str.lower()
    event_types = event_types.where(event_types.isin(NEWS_EVENT_TYPES), "other")

    news = pd.DataFrame(
        {
            "published_at": parsed_dates,
            "impact": impacts.clip(-1.0, 1.0),
            "quality": quality.clip(0.0, 1.0),
            "impact_horizon_hours": impact_horizons,
            "event_type": event_types,
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
        "event_type_rows": int(
            raw.get("event_type", pd.Series(dtype=str)).notna().sum()
        ),
        "impact_horizon_rows": int(impact_horizons.notna().sum()),
    }
    return news[
        ["published_at", "impact", "quality", "impact_horizon_hours", "event_type"]
    ], metadata


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
    event_types = news.get(
        "event_type", pd.Series("other", index=news.index)
    ).to_numpy()
    impact_horizons = pd.to_numeric(
        news.get("impact_horizon_hours", pd.Series(np.nan, index=news.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    hour_ns = float(pd.Timedelta(hours=1).value)
    lookback_ns = int(pd.Timedelta(hours=lookback_hours).value)
    day_ns = int(pd.Timedelta(hours=24).value)
    empty_row = (0.0,) * (6 + len(NEWS_EVENT_TYPES))

    rows: list[tuple[float, ...]] = []
    for timestamp in market_ns:
        end = int(np.searchsorted(news_ns, timestamp, side="right"))
        start = int(np.searchsorted(news_ns, timestamp - lookback_ns, side="left"))
        day_start = int(np.searchsorted(news_ns, timestamp - day_ns, side="left"))
        count_24h = float(end - day_start)
        if start == end:
            rows.append((*empty_row, count_24h))
            continue

        ages = (timestamp - news_ns[start:end]) / hour_ns
        row_horizons = impact_horizons[start:end]
        # Treat the expected impact horizon as approximately two half-lives.
        event_half_lives = np.where(
            np.isfinite(row_horizons),
            np.maximum(1.0, row_horizons / 2.0),
            half_life_hours,
        )
        decay = np.power(0.5, ages / event_half_lives)
        weights = decay * qualities[start:end]
        weight_sum = float(weights.sum())
        if weight_sum <= np.finfo(float).eps:
            rows.append((*empty_row, count_24h))
            continue

        recent_impacts = impacts[start:end]
        mean_impact = float(np.sum(weights * recent_impacts) / weight_sum)
        disagreement = float(
            np.sqrt(np.sum(weights * (recent_impacts - mean_impact) ** 2) / weight_sum)
        )
        bullish = float(np.sum(weights * np.clip(recent_impacts, 0.0, None)))
        bearish = float(np.sum(weights * np.clip(-recent_impacts, 0.0, None)))
        current_types = event_types[start:end]
        event_shares = tuple(
            float(np.sum(weights[current_types == event_type]) / weight_sum)
            for event_type in NEWS_EVENT_TYPES
        )
        rows.append(
            (
                mean_impact,
                bullish,
                bearish,
                disagreement,
                weight_sum,
                float(np.max(decay)),
                *event_shares,
                count_24h,
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "news_impact",
            "news_bullish_attention",
            "news_bearish_attention",
            "news_disagreement",
            "news_attention_weight",
            "news_freshness",
            *[f"news_event_{event_type}" for event_type in NEWS_EVENT_TYPES],
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
    features["return_5"] = log_close.diff(5)
    features["return_10"] = log_close.diff(10)
    features["return_20"] = log_close.diff(20)
    features["return_60"] = log_close.diff(60)
    features["mean_return_5"] = returns.rolling(5, min_periods=2).mean()
    features["mean_return_20"] = returns.rolling(20, min_periods=2).mean()
    features["volatility_10"] = returns.rolling(10, min_periods=2).std()
    features["volatility_30"] = returns.rolling(30, min_periods=2).std()
    features["ewm_return_10"] = returns.ewm(span=10, adjust=False).mean()
    features["ewm_return_30"] = returns.ewm(span=30, adjust=False).mean()
    features["trend_10"] = log_close - log_close.ewm(span=10, adjust=False).mean()
    features["trend_30"] = log_close - log_close.ewm(span=30, adjust=False).mean()

    gains = returns.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    losses = (-returns.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    directional_movement = gains + losses
    features["rsi_14"] = (
        gains / directional_movement.replace(0, np.nan) - 0.5
    )

    if "Open" in df:
        open_price = df["Open"].where(df["Open"] > 0)
        features["candle_body"] = np.log(close / open_price)
        features["opening_gap"] = np.log(open_price / close.shift(1))
    if {"High", "Low"}.issubset(df.columns):
        high = df["High"].where(df["High"] > 0)
        low = df["Low"].where(df["Low"] > 0)
        price_range = (high - low).where(high >= low)
        features["high_low_range"] = price_range / close
        features["close_position"] = (
            (close - low) / price_range.replace(0, np.nan) - 0.5
        ).clip(-0.5, 0.5)
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                price_range,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        features["atr_14"] = true_range.rolling(14, min_periods=2).mean() / close
    if "Volume" in df:
        volume = df["Volume"].clip(lower=0)
        log_volume = np.log1p(volume)
        features["log_volume"] = log_volume
        features["volume_change"] = log_volume.diff()
        rolling_volume = log_volume.rolling(30, min_periods=5)
        features["volume_zscore_30"] = (
            (log_volume - rolling_volume.mean())
            / rolling_volume.std().replace(0, np.nan)
        )

    dates = df[date_col]
    typical_interval = infer_interval(dates)
    features["elapsed_intervals"] = (
        dates.diff().dt.total_seconds() / typical_interval.total_seconds()
    ).clip(0, 10)
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


def estimate_return_bounds(training_returns: np.ndarray) -> tuple[float, float]:
    """Return conservative data-derived bounds for one-candle predictions."""
    values = np.asarray(training_returns, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite returns are required.")
    lower, upper = np.quantile(values, [0.005, 0.995])
    if lower == upper:
        padding = max(abs(float(lower)) * 0.1, np.finfo(float).eps)
        lower, upper = lower - padding, upper + padding
    return float(lower), float(upper)


def clip_predicted_returns(
    predicted_returns: np.ndarray,
    return_bounds: tuple[float, float],
) -> np.ndarray:
    """Prevent an unconstrained model output from implying an unseen price jump."""
    return np.clip(
        np.asarray(predicted_returns, dtype=float),
        return_bounds[0],
        return_bounds[1],
    )


def calibrate_return_scales(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
) -> np.ndarray:
    """Select an independent validation shrinkage for each forecast step."""
    actual = np.asarray(actual_returns, dtype=float)
    predicted = np.asarray(predicted_returns, dtype=float)
    if actual.ndim == 1:
        actual = actual[:, None]
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted validation returns must have matching shapes."
        )

    candidates = np.linspace(0.0, 1.0, 51)
    scales = []
    for horizon in range(actual.shape[1]):
        errors = [
            mean_absolute_error(
                actual[:, horizon], predicted[:, horizon] * candidate
            )
            for candidate in candidates
        ]
        scales.append(candidates[int(np.argmin(errors))])
    return np.asarray(scales, dtype=float)


def estimate_direction_probabilities(
    future_returns: np.ndarray,
    validation_actual_returns: np.ndarray,
    validation_predicted_returns: np.ndarray,
    seed: int,
    simulations: int = 10_000,
) -> dict[str, Any]:
    """Estimate the chance of finishing above or below the latest price.

    The network is deterministic, so its output alone is not a probability.
    We approximate forecast uncertainty by bootstrapping out-of-sample
    validation residuals and applying them across the predicted horizon.
    """
    actual = np.asarray(validation_actual_returns, dtype=float)
    predicted = np.asarray(validation_predicted_returns, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted validation returns must match.")
    residuals = actual - predicted
    if residuals.ndim == 1:
        residuals = residuals[:, None]
    residuals = residuals[np.all(np.isfinite(residuals), axis=1)]
    if len(residuals) < 2:
        raise ValueError(
            "At least two validation residuals are required to estimate "
            "direction probabilities."
        )

    horizon = len(future_returns)
    if residuals.shape[1] == 1:
        # Backward-compatible behavior for callers with only one-step errors.
        residual_rows = None
    elif residuals.shape[1] != horizon:
        raise ValueError(
            "Validation residual horizons must match the future forecast horizon."
        )
    else:
        residual_rows = residuals

    rng = np.random.default_rng(seed)
    if residual_rows is None:
        simulated_residuals = rng.choice(
            residuals[:, 0], size=(simulations, horizon), replace=True
        )
        method = "one-step validation-residual bootstrap"
    else:
        # Sample complete direct-forecast error vectors. This keeps both the
        # horizon-specific error distribution and dependence between steps.
        sampled_rows = rng.integers(0, len(residual_rows), size=simulations)
        simulated_residuals = residual_rows[sampled_rows]
        method = "horizon-vector validation-residual bootstrap"
    simulated_paths = np.cumsum(
        np.asarray(future_returns, dtype=float)[None, :] + simulated_residuals,
        axis=1,
    )
    terminal_log_returns = simulated_paths[:, -1]
    bullish_percent = float(np.mean(terminal_log_returns > 0) * 100)
    quantiles = np.quantile(simulated_paths, [0.1, 0.5, 0.9], axis=0)
    return {
        "bullish_percent": bullish_percent,
        "bearish_percent": 100.0 - bullish_percent,
        "horizon_steps": horizon,
        "simulations": simulations,
        "method": method,
        "cumulative_log_return_quantiles": {
            "p10": quantiles[0].tolist(),
            "p50": quantiles[1].tolist(),
            "p90": quantiles[2].tolist(),
        },
    }


def infer_interval(dates: pd.Series) -> pd.Timedelta:
    differences = dates.sort_values().diff().dropna()
    positive = differences[differences > pd.Timedelta(0)]
    if positive.empty:
        raise ValueError("Cannot infer the candle interval from the timestamp column.")
    return positive.median()


def make_future_dates(
    dates: pd.Series,
    periods: int,
    interval: pd.Timedelta,
) -> pd.DatetimeIndex:
    """Continue the observed candle schedule without inventing closed sessions."""
    if periods < 1:
        raise ValueError("periods must be positive.")
    if interval <= pd.Timedelta(0):
        raise ValueError("The candle interval must be positive.")

    # For intraday/daily series, observed weekday/time slots are a practical
    # exchange-calendar fallback. It skips forex/gold closures while retaining
    # continuous weekend candles for crypto, without hard-coding asset names.
    minutes = dates.dt.hour * 60 + dates.dt.minute
    observed_slots = set(zip(dates.dt.dayofweek.astype(int), minutes.astype(int)))
    candidate = pd.Timestamp(dates.iloc[-1])
    future: list[pd.Timestamp] = []
    maximum_attempts = periods * max(32, int(pd.Timedelta(days=14) / interval))
    attempts = 0
    while len(future) < periods and attempts < maximum_attempts:
        candidate += interval
        attempts += 1
        slot = (candidate.dayofweek, candidate.hour * 60 + candidate.minute)
        if slot in observed_slots:
            future.append(candidate)
    if len(future) != periods:
        # Very short or irregular datasets may not reveal a reusable schedule.
        return pd.date_range(dates.iloc[-1] + interval, periods=periods, freq=interval)
    return pd.DatetimeIndex(future)


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
    return_scales: np.ndarray,
    return_bounds: tuple[float, float],
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
    direction_probabilities: dict[str, Any],
    show_plot: bool,
    save_plots: bool = False,
) -> None:
    model_dir = PROJECT_ROOT / "models"
    asset_symbol = extract_asset_symbol(coin_name)
    stamp = datetime.now().strftime("%Y-%m-%d")
    plot_dir = PROJECT_ROOT / "plots" / asset_symbol / stamp
    model_dir.mkdir(parents=True, exist_ok=True)

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
                "return_scales": return_scales.tolist(),
                "return_bounds": return_bounds,
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
    uncertainty = direction_probabilities["cumulative_log_return_quantiles"]
    ax.fill_between(
        future_dates,
        close[-1] * np.exp(np.asarray(uncertainty["p10"])),
        close[-1] * np.exp(np.asarray(uncertainty["p90"])),
        color="darkorange",
        alpha=0.15,
        label="80% validation-residual interval",
    )
    ax.axvline(dates.iloc[split_index], color="gray", linestyle="--", alpha=0.7)
    ax.set_title(f"{coin_name} price forecast using {model_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend()
    bullish = direction_probabilities["bullish_percent"]
    bearish = direction_probabilities["bearish_percent"]
    ax.text(
        0.015,
        0.97,
        f"Forecast direction ({config.forecast_steps} steps)\n"
        f"Bullish: {bullish:.1f}%   Bearish: {bearish:.1f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.85},
    )
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
    # Leave room for an interactive save button when the figure is displayed.
    fig.tight_layout(rect=(0, 0.07 if show_plot else 0, 1, 1))

    png_path = plot_dir / f"{stamp}_{kind}.png"
    pickle_path = plot_dir / f"{stamp}_{kind}.pkl"
    save_button: Button | None = None
    button_axes = None

    def save_plot(_event: Any = None) -> None:
        """Save both the rendered image and the reusable Matplotlib figure."""
        plot_dir.mkdir(parents=True, exist_ok=True)
        if button_axes is not None:
            button_axes.set_visible(False)
        try:
            fig.savefig(png_path, dpi=150)
            with pickle_path.open("wb") as output:
                pickle.dump(fig, output)
        finally:
            if button_axes is not None:
                button_axes.set_visible(True)
                fig.canvas.draw_idle()
        print(f"Plot saved to: {png_path}")

    if save_plots:
        save_plot()
    if show_plot:
        button_axes = fig.add_axes((0.44, 0.015, 0.12, 0.04))
        save_button = Button(button_axes, "Save plot")
        save_button.on_clicked(save_plot)
        plt.show()
        # Keep a reference until the GUI window closes; Matplotlib widgets stop
        # responding when they are garbage-collected.
        _ = save_button
    plt.close(fig)

    print(f"Model saved to: {model_path}")
    print(f"Preprocessing saved to: {preprocessing_path}")
    print(f"Metrics saved to: {metrics_path}")


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
    save_plots: bool = False,
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
    if interval_minutes is not None and interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive when provided.")

    set_reproducible_seed(config.seed)
    df = load_market_data(csv_path, date_col, close_col)
    dates = df[date_col]
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
    return_bounds = estimate_return_bounds(log_returns[1:validation_start])
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

    tf.keras.backend.clear_session()
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
    history = model.fit(
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
    validation_returns_by_horizon = clip_predicted_returns(
        target_scaler.inverse_transform(
            validation_scaled.reshape(-1, 1)
        ).reshape(validation_scaled.shape),
        return_bounds,
    )
    validation_actual_returns_by_horizon = target_scaler.inverse_transform(
        y_validation.reshape(-1, 1)
    ).reshape(y_validation.shape)
    return_scales = calibrate_return_scales(
        validation_actual_returns_by_horizon,
        validation_returns_by_horizon,
    )
    return_scale = float(return_scales[0])
    print(f"Validation-selected return scales: {return_scales.tolist()}")

    # Walk-forward evaluation: every prediction uses actual candles observed
    # before its timestamp, never the candle it is predicting.
    x_test = np.asarray(
        [scaled_features[i - config.window : i] for i in range(split_index, len(df))],
        dtype=np.float32,
    )
    predicted_scaled = model.predict(x_test, batch_size=config.batch_size, verbose=0)
    raw_predicted_returns = clip_predicted_returns(
        target_scaler.inverse_transform(predicted_scaled[:, :1]).ravel(),
        return_bounds,
    )
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
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "prediction_return_bounds": list(return_bounds),
        "validation_selected_return_scale": return_scale,
        "validation_selected_return_scales": return_scales.tolist(),
        "model_metrics": model_metrics,
        "raw_unscaled_model_metrics": raw_model_metrics,
        "naive_last_price_baseline": baseline_metrics,
    }
    final_window = scaled_features[-config.window :][None, ...].astype(np.float32)
    future_scaled = model.predict(final_window, verbose=0)[0].reshape(-1, 1)
    future_returns = (
        clip_predicted_returns(
            target_scaler.inverse_transform(future_scaled).ravel(), return_bounds
        )
        * return_scales
    )
    future_predictions = close[-1] * np.exp(np.cumsum(future_returns))
    direction_probabilities = estimate_direction_probabilities(
        future_returns=future_returns,
        validation_actual_returns=validation_actual_returns_by_horizon,
        validation_predicted_returns=(
            validation_returns_by_horizon * return_scales[None, :]
        ),
        seed=config.seed,
    )
    metrics["forecast_direction_probability"] = direction_probabilities
    interval = (
        pd.Timedelta(minutes=interval_minutes)
        if interval_minutes is not None
        else infer_interval(df[date_col])
    )
    future_dates = make_future_dates(dates, config.forecast_steps, interval)
    uncertainty = direction_probabilities["cumulative_log_return_quantiles"]

    context_start = max(0, split_index - config.window)
    metrics["chart"] = {
        "title": f"{extract_asset_symbol(csv_path.stem)} price forecast ({kind})",
        "actual_dates": [
            pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
            for value in dates.iloc[context_start:]
        ],
        "actual_prices": close[context_start:].tolist(),
        "test_dates": [
            pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
            for value in dates.iloc[split_index:]
        ],
        "test_predictions": test_predictions.tolist(),
        "future_dates": [
            pd.Timestamp(value).strftime("%Y-%m-%d %H:%M") for value in future_dates
        ],
        "future_predictions": future_predictions.tolist(),
        "future_lower": (
            close[-1] * np.exp(np.asarray(uncertainty["p10"]))
        ).tolist(),
        "future_median": (
            close[-1] * np.exp(np.asarray(uncertainty["p50"]))
        ).tolist(),
        "future_upper": (
            close[-1] * np.exp(np.asarray(uncertainty["p90"]))
        ).tolist(),
        "forecast_interval_percent": 80,
    }
    print(json.dumps(metrics, indent=2))

    save_outputs(
        kind=kind,
        coin_name=csv_path.stem,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        return_scale=return_scale,
        return_scales=return_scales,
        return_bounds=return_bounds,
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
        direction_probabilities=direction_probabilities,
        show_plot=show_plot,
        save_plots=save_plots,
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
        help="Display the plot with an interactive Save plot button (default)",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Do not display the plot (combine with --save-plots to save it)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Automatically save plot PNG and pickle files; otherwise use the Save plot button",
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
        save_plots=args.save_plots,
    )
