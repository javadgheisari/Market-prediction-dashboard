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
    build_features,
    build_news_features,
    calculate_metrics,
    calibrate_return_scales,
    clip_predicted_returns,
    estimate_return_bounds,
    estimate_direction_probabilities,
    extract_asset_symbol,
    infer_interval,
    load_market_data,
    load_news_data,
    make_future_dates,
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
    early_stopping_rounds: int = 30,
    n_jobs: int = 1,
    news_half_life_hours: float = 24.0,
    news_lookback_hours: float = 168.0,
    news_date_col: str | None = None,
    news_impact_col: str | None = None,
    interval_minutes: float | None = None,
    save_model: bool = True,
) -> dict[str, Any]:
    """Train XGBoost on rolling causal features and forecast a direct horizon."""
    if min(
        window,
        forecast_steps,
        n_estimators,
        max_depth,
        early_stopping_rounds,
        n_jobs,
    ) < 1:
        raise ValueError(
            "window, forecast_steps, n_estimators, max_depth, "
            "early_stopping_rounds, and n_jobs must be positive."
        )
    if not 0.5 <= split_ratio < 1 or not 0 < validation_ratio < 0.5:
        raise ValueError("Invalid chronological split configuration.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if news_half_life_hours <= 0 or news_lookback_hours <= 0:
        raise ValueError("News half-life and lookback must be positive.")
    if interval_minutes is not None and interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive when provided.")

    set_reproducible_seed(seed)
    df = load_market_data(csv_path, date_col, close_col)
    dates = df[date_col]
    news_metadata = None
    news_features = None
    if news_csv_path is not None:
        news, news_metadata = load_news_data(
            news_csv_path,
            dates,
            csv_path.stem,
            date_col=news_date_col,
            impact_col=news_impact_col,
        )
        news_features = build_news_features(
            dates, news, news_half_life_hours, news_lookback_hours
        )
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
    if split_index + 1 >= len(df):
        raise ValueError("The test partition must contain at least two rows.")
    if news_features is not None and news_metadata is not None:
        training_coverage = float(
            np.mean(news_features.iloc[:validation_start].abs().sum(axis=1) > 0)
            * 100
        )
        news_metadata["training_feature_coverage_percent"] = training_coverage
        if training_coverage == 0:
            raise ValueError(
                "News does not overlap the training partition. Provide historical "
                "news covering the price-training period."
            )

    return_bounds = estimate_return_bounds(log_returns[1:validation_start])

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
        n_jobs=n_jobs,
        tree_method="hist",
        eval_metric="mae",
        early_stopping_rounds=min(early_stopping_rounds, n_estimators),
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )

    validation_predictions = clip_predicted_returns(
        model.predict(x_validation), return_bounds
    )
    if validation_predictions.ndim == 1:
        validation_predictions = validation_predictions[:, None]
    return_scales = calibrate_return_scales(y_validation, validation_predictions)
    return_scale = float(return_scales[0])

    x_test = np.asarray(
        [
            feature_values[i - window : i].reshape(-1)
            for i in range(split_index, len(df))
        ],
        dtype=np.float32,
    )
    raw_test_returns = clip_predicted_returns(
        first_horizon(model.predict(x_test)), return_bounds
    )
    predicted_returns = raw_test_returns * return_scale
    previous_close = close[split_index - 1 : len(df) - 1]
    test_predictions = previous_close * np.exp(predicted_returns)
    actual_test = close[split_index:]
    baseline_metrics = calculate_metrics(actual_test, previous_close)
    model_metrics = calculate_metrics(actual_test, test_predictions)
    raw_model_metrics = calculate_metrics(
        actual_test, previous_close * np.exp(raw_test_returns)
    )
    model_metrics["directional_accuracy_percent"] = float(
        np.mean(
            np.sign(actual_test - previous_close)
            == np.sign(test_predictions - previous_close)
        )
        * 100
    )
    model_metrics["mae_skill_vs_naive_percent"] = float(
        (1 - model_metrics["mae"] / baseline_metrics["mae"]) * 100
        if baseline_metrics["mae"] > 0
        else 0.0
    )

    final_window = feature_values[-window:].reshape(1, -1)
    future_returns = (
        clip_predicted_returns(
            np.asarray(model.predict(final_window)).reshape(-1), return_bounds
        )
        * return_scales
    )
    future_predictions = close[-1] * np.exp(np.cumsum(future_returns))
    probabilities = estimate_direction_probabilities(
        future_returns,
        y_validation,
        validation_predictions * return_scales[None, :],
        seed,
    )
    interval = (
        pd.Timedelta(minutes=interval_minutes)
        if interval_minutes is not None
        else infer_interval(dates)
    )
    future_dates = make_future_dates(dates, forecast_steps, interval)
    uncertainty = probabilities["cumulative_log_return_quantiles"]
    context_start = max(0, split_index - window)
    metrics: dict[str, Any] = {
        "model": "XGBOOST",
        "rows": len(df),
        "train_samples": len(x_train),
        "validation_samples": len(x_validation),
        "test_samples": len(actual_test),
        "news": news_metadata,
        "best_iteration": int(model.best_iteration),
        "selected_trees": int(model.best_iteration + 1),
        "prediction_return_bounds": list(return_bounds),
        "validation_selected_return_scale": return_scale,
        "validation_selected_return_scales": return_scales.tolist(),
        "model_metrics": model_metrics,
        "raw_unscaled_model_metrics": raw_model_metrics,
        "naive_last_price_baseline": baseline_metrics,
        "forecast_direction_probability": probabilities,
        "chart": {
            "title": f"{extract_asset_symbol(csv_path.stem)} price forecast (XGBoost)",
            "actual_dates": [
                value.strftime("%Y-%m-%d %H:%M")
                for value in dates.iloc[context_start:]
            ],
            "actual_prices": close[context_start:].tolist(),
            "test_dates": [
                value.strftime("%Y-%m-%d %H:%M")
                for value in dates.iloc[split_index:]
            ],
            "test_predictions": test_predictions.tolist(),
            "future_dates": [
                value.strftime("%Y-%m-%d %H:%M") for value in future_dates
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
                    "return_bounds": return_bounds,
                    "forecast_steps": forecast_steps,
                    "best_iteration": int(model.best_iteration),
                    "n_estimators": int(model.best_iteration + 1),
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
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--news-date-col")
    parser.add_argument("--news-impact-col")
    parser.add_argument("--news-half-life-hours", type=float, default=24.0)
    parser.add_argument("--news-lookback-hours", type=float, default=168.0)
    parser.add_argument("--interval-minutes", type=float)
    args = parser.parse_args()
    metrics = train_and_forecast_xgboost(
        args.price_csv,
        args.date_col,
        args.close_col,
        args.forecast_steps,
        args.news_csv,
        window=args.window,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        n_jobs=args.n_jobs,
        news_date_col=args.news_date_col,
        news_impact_col=args.news_impact_col,
        news_half_life_hours=args.news_half_life_hours,
        news_lookback_hours=args.news_lookback_hours,
        interval_minutes=args.interval_minutes,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
