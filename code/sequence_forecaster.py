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


def build_features(df: pd.DataFrame, date_col: str, close_col: str) -> pd.DataFrame:
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


def save_outputs(
    kind: ModelKind,
    coin_name: str,
    model: tf.keras.Model,
    feature_scaler: RobustScaler,
    target_scaler: StandardScaler,
    return_scale: float,
    config: ForecastConfig,
    feature_names: list[str],
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
    plot_dir = PROJECT_ROOT / "plots" / coin_name
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
                "config": asdict(config),
            },
            output,
        )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(16, 7))
    context_start = max(0, split_index - config.window)
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
        label=f"{kind} one-step test forecast",
    )
    ax.plot(
        future_dates,
        future_predictions,
        color="darkorange",
        marker="o",
        markersize=3,
        label=f"{kind} future forecast",
    )
    ax.axvline(dates.iloc[split_index], color="gray", linestyle="--", alpha=0.7)
    ax.set_title(f"{coin_name} price forecast using {kind}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    stamp = datetime.now().strftime("%Y-%m-%d")
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


def choose_csv_file() -> Path:
    try:
        from easygui import fileopenbox
    except ImportError as exc:
        raise ValueError("Pass a CSV path on the command line (easygui is unavailable).") from exc
    selected = fileopenbox(title="Choose a CSV file containing market data")
    if not selected:
        raise ValueError("No CSV file was selected.")
    return Path(selected)


def train_and_forecast(
    kind: ModelKind,
    csv_path: Path,
    date_col: str,
    close_col: str,
    config: ForecastConfig,
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

    set_reproducible_seed(config.seed)
    df = load_market_data(csv_path, date_col, close_col)
    features = build_features(df, date_col, close_col)
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
    parser.add_argument("csv", nargs="?", type=Path, help="Market data CSV; opens a dialog if omitted")
    parser.add_argument("--date-col", default="Asia/Tehran")
    parser.add_argument("--close-col", default="Close")
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
    parser.add_argument(
        "--interval-minutes",
        type=float,
        help="Future candle interval; inferred from timestamps when omitted",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot after saving it")
    return parser


def main(kind: ModelKind) -> None:
    args = build_argument_parser(kind).parse_args()
    csv_path = args.csv or choose_csv_file()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
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
    )
    train_and_forecast(
        kind=kind,
        csv_path=csv_path.resolve(),
        date_col=args.date_col,
        close_col=args.close_col,
        config=config,
        interval_minutes=args.interval_minutes,
        show_plot=args.show,
    )
