"""Leakage-aware XGBoost market forecasting pipeline."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from sequence_forecaster import (
    ForecastConfig,
    build_features,
    build_news_features,
    calculate_metrics,
    calibrate_return_scales,
    estimate_direction_probabilities,
    extract_asset_symbol,
    infer_interval,
    load_market_data,
    load_news_data,
    make_sequences,
    set_reproducible_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def first_horizon(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return values if values.ndim == 1 else values[:, 0]


def train_and_forecast_xgboost(
    csv_path: Path,
    date_col: str,
    close_col: str = "Close",
    forecast_steps: int = 24,
    news_csv_path: Path | None = None,
    *,
    window: int = 60,
    split_ratio: float = 0.80,
    validation_ratio: float = 0.20,
    seed: int = 42,
    n_estimators: int = 300,
    max_depth: int = 3,
    learning_rate: float = 0.03,
    interval_minutes: float | None = None,
    save_model: bool = True,
) -> dict[str, Any]:
    """Train XGBoost on rolling causal features and forecast a direct horizon."""
    if min(window, forecast_steps, n_estimators, max_depth) < 1:
        raise ValueError("window, forecast_steps, n_estimators, and max_depth must be positive.")
    if not 0.5 <= split_ratio < 1 or not 0 < validation_ratio < 0.5:
        raise ValueError("Invalid chronological split configuration.")

    set_reproducible_seed(seed)
    df = load_market_data(csv_path, date_col, close_col)
    dates = df[date_col]
    news_metadata = None
    news_features = None
    if news_csv_path is not None:
        news, news_metadata = load_news_data(
            news_csv_path, dates, csv_path.stem
        )
        news_features = build_news_features(dates, news, 24.0, 168.0)
        coverage = float(np.mean(news_features.abs().sum(axis=1) > 0) * 100)
        news_metadata["feature_coverage_percent"] = coverage
        if coverage == 0:
            raise ValueError(
                "No news overlaps the selected historical price period. "
                "Use matching historical news or train with price data only."
            )

    features = build_features(df, date_col, close_col, news_features)
    feature_values = features.to_numpy(dtype=np.float32)
    close = df[close_col].to_numpy(dtype=np.float64)
    log_returns = np.diff(np.log(close), prepend=np.log(close[0])).reshape(-1, 1)
    split_index = int(len(df) * split_ratio)
    validation_start = int(split_index * (1 - validation_ratio))
    if validation_start < window + forecast_steps + 1:
        raise ValueError(
            "Dataset is too small. Use a shorter forecast horizon or provide more rows "
            f"(received {len(df)} rows)."
        )

    x_train, y_train = make_sequences(
        feature_values,
        log_returns,
        window,
        forecast_steps,
        window,
        validation_start - forecast_steps + 1,
    )
    x_validation, y_validation = make_sequences(
        feature_values,
        log_returns,
        window,
        forecast_steps,
        validation_start,
        split_index - forecast_steps + 1,
    )
    x_train = x_train.reshape(len(x_train), -1)
    x_validation = x_validation.reshape(len(x_validation), -1)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(x_train, y_train)

    validation_predictions = np.asarray(model.predict(x_validation))
    if validation_predictions.ndim == 1:
        validation_predictions = validation_predictions[:, None]
    return_scales = calibrate_return_scales(y_validation, validation_predictions)
    validation_returns = validation_predictions[:, 0]
    validation_indexes = np.arange(
        validation_start, validation_start + len(x_validation)
    )
    return_scale = float(return_scales[0])

    x_test = np.asarray(
        [feature_values[i - window : i].reshape(-1) for i in range(split_index, len(df))],
        dtype=np.float32,
    )
    raw_test_returns = first_horizon(model.predict(x_test))
    predicted_returns = raw_test_returns * return_scale
    previous_close = close[split_index - 1 : len(df) - 1]
    test_predictions = previous_close * np.exp(predicted_returns)
    actual_test = close[split_index:]
    baseline_metrics = calculate_metrics(actual_test, previous_close)
    model_metrics = calculate_metrics(actual_test, test_predictions)
    model_metrics["directional_accuracy_percent"] = float(
        np.mean(np.sign(actual_test - previous_close) == np.sign(test_predictions - previous_close))
        * 100
    )
    model_metrics["mae_skill_vs_naive_percent"] = float(
        (1 - model_metrics["mae"] / baseline_metrics["mae"]) * 100
        if baseline_metrics["mae"] > 0
        else 0.0
    )

    final_window = feature_values[-window:].reshape(1, -1)
    future_returns = (
        np.asarray(model.predict(final_window)).reshape(-1) * return_scales
    )
    future_predictions = close[-1] * np.exp(np.cumsum(future_returns))
    probabilities = estimate_direction_probabilities(
        future_returns,
        log_returns[validation_indexes, 0],
        validation_returns * return_scale,
        seed,
    )
    interval = (
        pd.Timedelta(minutes=interval_minutes)
        if interval_minutes is not None
        else infer_interval(dates)
    )
    future_dates = pd.date_range(
        dates.iloc[-1] + interval, periods=forecast_steps, freq=interval
    )
    context_start = max(0, split_index - window)
    metrics: dict[str, Any] = {
        "model": "XGBOOST",
        "rows": len(df),
        "train_samples": len(x_train),
        "validation_samples": len(x_validation),
        "test_samples": len(actual_test),
        "news": news_metadata,
        "validation_selected_return_scale": return_scale,
        "validation_selected_return_scales": return_scales.tolist(),
        "model_metrics": model_metrics,
        "naive_last_price_baseline": baseline_metrics,
        "forecast_direction_probability": probabilities,
        "chart": {
            "title": f"{extract_asset_symbol(csv_path.stem)} price forecast (XGBoost)",
            "actual_dates": [value.strftime("%Y-%m-%d %H:%M") for value in dates.iloc[context_start:]],
            "actual_prices": close[context_start:].tolist(),
            "test_dates": [value.strftime("%Y-%m-%d %H:%M") for value in dates.iloc[split_index:]],
            "test_predictions": test_predictions.tolist(),
            "future_dates": [value.strftime("%Y-%m-%d %H:%M") for value in future_dates],
            "future_predictions": future_predictions.tolist(),
        },
    }

    if save_model:
        model_dir = PROJECT_ROOT / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        stem = f"xgboost_{csv_path.stem}"
        model.save_model(str(model_dir / f"{stem}.json"))
        with (model_dir / f"{stem}_preprocessing.pkl").open("wb") as output:
            pickle.dump(
                {
                    "window": window,
                    "feature_names": list(features.columns),
                    "return_scale": return_scale,
                    "return_scales": return_scales.tolist(),
                    "forecast_steps": forecast_steps,
                },
                output,
            )
        (model_dir / f"{stem}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("price_csv", type=Path)
    parser.add_argument("news_csv", nargs="?", type=Path)
    parser.add_argument("--date-col", default="Asia/Tehran")
    parser.add_argument("--close-col", default="Close")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--forecast-steps", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    metrics = train_and_forecast_xgboost(
        args.price_csv,
        args.date_col,
        args.close_col,
        args.forecast_steps,
        args.news_csv,
        window=args.window,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
