"""Leakage-aware XGBoost portfolio for direct market forecasting."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
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
    load_context_features,
    load_news_data,
    make_future_dates,
    make_sequences,
    set_reproducible_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[:, None] if values.ndim == 1 else values


def _recency_weights(sample_count: int, half_life_fraction: float) -> np.ndarray:
    """Return smoothly increasing weights without discarding older regimes."""
    half_life = max(1.0, sample_count * half_life_fraction)
    age = np.arange(sample_count - 1, -1, -1, dtype=float)
    weights = np.power(0.5, age / half_life)
    return weights / np.mean(weights)


def _portfolio_specs(
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
    portfolio_size: int,
) -> list[dict[str, Any]]:
    candidates = [
        {
            "name": "balanced",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": 2.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 1.5,
            "random_state": seed,
        },
        {
            "name": "conservative",
            "n_estimators": max(100, int(n_estimators * 1.15)),
            "max_depth": max(2, max_depth - 1),
            "learning_rate": learning_rate * 0.8,
            "min_child_weight": 5.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.15,
            "reg_lambda": 3.0,
            "random_state": seed + 1,
        },
        {
            "name": "adaptive",
            "n_estimators": max(100, int(n_estimators * 1.25)),
            "max_depth": min(6, max_depth + 1),
            "learning_rate": learning_rate * 0.7,
            "min_child_weight": 3.0,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "random_state": seed + 2,
        },
    ]
    return candidates[:portfolio_size]


def _select_portfolio_weights(
    actual: np.ndarray, calibrated_predictions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Choose per-horizon members/blends using validation MAE only.

    The zero-return naive forecast is an explicit candidate. This prevents a
    weak model from being forced into the final forecast merely because it was
    trained successfully.
    """
    actual = _as_2d(actual)
    members = np.asarray(calibrated_predictions, dtype=float)
    member_count, sample_count, horizon_count = members.shape
    if actual.shape != (sample_count, horizon_count):
        raise ValueError("Portfolio predictions must match validation targets.")

    weights = np.zeros((horizon_count, member_count), dtype=float)
    combined = np.zeros_like(actual)
    diagnostics: list[dict[str, Any]] = []
    for horizon in range(horizon_count):
        target = actual[:, horizon]
        member_mae = np.asarray(
            [mean_absolute_error(target, member[:, horizon]) for member in members]
        )
        inverse = 1.0 / np.maximum(member_mae, np.finfo(float).eps)
        blend_weights = inverse / inverse.sum()
        choices = [np.zeros(member_count)]
        choices.extend(np.eye(member_count))
        choices.append(blend_weights)
        choice_predictions = [
            sum(weight[index] * members[index, :, horizon] for index in range(member_count))
            for weight in choices
        ]
        errors = [mean_absolute_error(target, prediction) for prediction in choice_predictions]
        selected = int(np.argmin(errors))
        weights[horizon] = choices[selected]
        combined[:, horizon] = choice_predictions[selected]
        labels = ["naive_zero", *[f"member_{i}" for i in range(member_count)], "blend"]
        diagnostics.append(
            {
                "horizon": horizon + 1,
                "selected": labels[selected],
                "validation_mae": float(errors[selected]),
                "naive_mae": float(errors[0]),
                "member_mae": member_mae.tolist(),
                "weights": weights[horizon].tolist(),
            }
        )
    return weights, combined, diagnostics


def train_and_forecast_xgboost(
    csv_path: Path,
    date_col: str,
    close_col: str = "Close",
    forecast_steps: int = 24,
    news_csv_path: Path | None = None,
    context_csv_path: Path | None = None,
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
    portfolio_size: int = 3,
    recency_half_life: float = 0.5,
    news_half_life_hours: float = 24.0,
    news_lookback_hours: float = 168.0,
    news_date_col: str | None = None,
    news_impact_col: str | None = None,
    interval_minutes: float | None = None,
    save_model: bool = True,
) -> dict[str, Any]:
    """Train a validation-selected portfolio and forecast a direct horizon."""
    if min(
        window, forecast_steps, n_estimators, max_depth,
        early_stopping_rounds, n_jobs, portfolio_size,
    ) < 1:
        raise ValueError("Training counts, dimensions, and portfolio_size must be positive.")
    if portfolio_size > 3:
        raise ValueError("portfolio_size cannot exceed the three configured members.")
    if not 0.5 <= split_ratio < 1 or not 0 < validation_ratio < 0.5:
        raise ValueError("Invalid chronological split configuration.")
    if learning_rate <= 0 or recency_half_life <= 0:
        raise ValueError("learning_rate and recency_half_life must be positive.")
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
            news_csv_path, dates, csv_path.stem,
            date_col=news_date_col, impact_col=news_impact_col,
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

    context_metadata = None
    context_features = None
    if context_csv_path is not None:
        context_features, context_metadata = load_context_features(
            context_csv_path, dates
        )
        if context_metadata["overall_coverage_percent"] == 0:
            raise ValueError(
                "No market intelligence overlaps the selected price history."
            )

    features = build_features(
        df, date_col, close_col, news_features, context_features
    )
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
            np.mean(news_features.iloc[:validation_start].abs().sum(axis=1) > 0) * 100
        )
        news_metadata["training_feature_coverage_percent"] = training_coverage
        if training_coverage == 0:
            raise ValueError(
                "News does not overlap the training partition. Provide historical "
                "news covering the price-training period."
            )
    if context_features is not None and context_metadata is not None:
        training_coverage = float(
            np.mean(
                context_features.iloc[:validation_start].abs().sum(axis=1) > 0
            )
            * 100
        )
        context_metadata["training_feature_coverage_percent"] = training_coverage
        if training_coverage == 0:
            raise ValueError(
                "Market intelligence does not overlap the training partition. "
                "Collect a longer history or train without market intelligence."
            )

    return_bounds = estimate_return_bounds(log_returns[1:validation_start])
    x_train, y_train = make_sequences(
        feature_values, log_returns, window, forecast_steps,
        window, validation_start - forecast_steps + 1,
    )
    x_validation, y_validation = make_sequences(
        feature_values, log_returns, window, forecast_steps,
        validation_start, split_index - forecast_steps + 1,
    )
    x_train = x_train.reshape(len(x_train), -1)
    x_validation = x_validation.reshape(len(x_validation), -1)
    sample_weight = _recency_weights(len(x_train), recency_half_life)

    models: list[XGBRegressor] = []
    member_scales: list[np.ndarray] = []
    member_validation: list[np.ndarray] = []
    member_metadata: list[dict[str, Any]] = []
    specs = _portfolio_specs(
        n_estimators, max_depth, learning_rate, seed, portfolio_size
    )
    for spec in specs:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=spec["n_estimators"],
            max_depth=spec["max_depth"],
            learning_rate=spec["learning_rate"],
            min_child_weight=spec["min_child_weight"],
            subsample=spec["subsample"],
            colsample_bytree=spec["colsample_bytree"],
            reg_alpha=spec["reg_alpha"],
            reg_lambda=spec["reg_lambda"],
            random_state=spec["random_state"],
            n_jobs=n_jobs,
            tree_method="hist",
            eval_metric="mae",
            early_stopping_rounds=min(early_stopping_rounds, spec["n_estimators"]),
        )
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_validation, y_validation)],
            verbose=False,
        )
        raw_validation = clip_predicted_returns(
            _as_2d(model.predict(x_validation)), return_bounds
        )
        scales = calibrate_return_scales(y_validation, raw_validation)
        calibrated = raw_validation * scales[None, :]
        models.append(model)
        member_scales.append(scales)
        member_validation.append(calibrated)
        member_metadata.append(
            {
                **spec,
                "best_iteration": int(model.best_iteration),
                "selected_trees": int(model.best_iteration + 1),
                "validation_mae": float(mean_absolute_error(y_validation, calibrated)),
                "return_scales": scales.tolist(),
            }
        )

    validation_members = np.asarray(member_validation)
    portfolio_weights, validation_predictions, horizon_selection = (
        _select_portfolio_weights(y_validation, validation_members)
    )
    return_scales = np.sum(
        portfolio_weights * np.asarray(member_scales).T, axis=1
    )
    return_scale = float(return_scales[0])

    x_test = np.asarray(
        [feature_values[i - window : i].reshape(-1) for i in range(split_index, len(df))],
        dtype=np.float32,
    )
    raw_test_member_returns = np.asarray(
        [
            clip_predicted_returns(_as_2d(model.predict(x_test))[:, 0], return_bounds)
            for model in models
        ]
    )
    test_member_returns = raw_test_member_returns * np.asarray(
        [scales[0] for scales in member_scales]
    )[:, None]
    predicted_returns = np.sum(
        portfolio_weights[0, :, None] * test_member_returns, axis=0
    )
    raw_predicted_returns = np.sum(
        portfolio_weights[0, :, None] * raw_test_member_returns, axis=0
    )
    previous_close = close[split_index - 1 : len(df) - 1]
    test_predictions = previous_close * np.exp(predicted_returns)
    actual_test = close[split_index:]
    baseline_metrics = calculate_metrics(actual_test, previous_close)
    model_metrics = calculate_metrics(actual_test, test_predictions)
    model_metrics["directional_accuracy_percent"] = float(
        np.mean(
            np.sign(actual_test - previous_close)
            == np.sign(test_predictions - previous_close)
        ) * 100
    )
    model_metrics["mae_skill_vs_naive_percent"] = float(
        (1 - model_metrics["mae"] / baseline_metrics["mae"]) * 100
        if baseline_metrics["mae"] > 0 else 0.0
    )

    final_window = feature_values[-window:].reshape(1, -1)
    future_member_returns = np.asarray(
        [
            clip_predicted_returns(
                _as_2d(model.predict(final_window))[0], return_bounds
            ) * member_scales[index]
            for index, model in enumerate(models)
        ]
    )
    future_returns = np.sum(portfolio_weights.T * future_member_returns, axis=0)
    future_predictions = close[-1] * np.exp(np.cumsum(future_returns))
    probabilities = estimate_direction_probabilities(
        future_returns,
        y_validation,
        validation_predictions,
        seed,
    )
    interval = (
        pd.Timedelta(minutes=interval_minutes)
        if interval_minutes is not None else infer_interval(dates)
    )
    future_dates = make_future_dates(dates, forecast_steps, interval)
    uncertainty = probabilities["cumulative_log_return_quantiles"]
    context_start = max(0, split_index - window)
    metrics: dict[str, Any] = {
        "model": "XGBOOST_PORTFOLIO" if len(models) > 1 else "XGBOOST",
        "rows": len(df),
        "train_samples": len(x_train),
        "validation_samples": len(x_validation),
        "test_samples": len(actual_test),
        "news": news_metadata,
        "context": context_metadata,
        "portfolio": {
            "members": member_metadata,
            "horizon_selection": horizon_selection,
            "recency_half_life_fraction": recency_half_life,
            "sample_weight_min": float(sample_weight.min()),
            "sample_weight_max": float(sample_weight.max()),
        },
        "best_iteration": int(models[0].best_iteration),
        "selected_trees": int(models[0].best_iteration + 1),
        "prediction_return_bounds": list(return_bounds),
        "validation_selected_return_scale": return_scale,
        "validation_selected_return_scales": return_scales.tolist(),
        "validation_portfolio_mae": float(
            mean_absolute_error(y_validation, validation_predictions)
        ),
        "model_metrics": model_metrics,
        "raw_unscaled_model_metrics": calculate_metrics(
            actual_test, previous_close * np.exp(raw_predicted_returns)
        ),
        "naive_last_price_baseline": baseline_metrics,
        "forecast_direction_probability": probabilities,
        "chart": {
            "title": f"{extract_asset_symbol(csv_path.stem)} price forecast (XGBoost portfolio)",
            "actual_dates": [value.strftime("%Y-%m-%d %H:%M") for value in dates.iloc[context_start:]],
            "actual_prices": close[context_start:].tolist(),
            "test_dates": [value.strftime("%Y-%m-%d %H:%M") for value in dates.iloc[split_index:]],
            "test_predictions": test_predictions.tolist(),
            "future_dates": [value.strftime("%Y-%m-%d %H:%M") for value in future_dates],
            "future_predictions": future_predictions.tolist(),
            "future_lower": (close[-1] * np.exp(np.asarray(uncertainty["p10"]))).tolist(),
            "future_median": (close[-1] * np.exp(np.asarray(uncertainty["p50"]))).tolist(),
            "future_upper": (close[-1] * np.exp(np.asarray(uncertainty["p90"]))).tolist(),
            "forecast_interval_percent": 80,
        },
    }

    if save_model:
        model_dir = PROJECT_ROOT / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        stem = f"xgboost_{csv_path.stem}"
        primary_index = int(np.argmax(portfolio_weights[0]))
        models[primary_index].save_model(str(model_dir / f"{stem}.json"))
        for index, model in enumerate(models):
            model.save_model(str(model_dir / f"{stem}_member_{index}.json"))
        with (model_dir / f"{stem}_preprocessing.pkl").open("wb") as output:
            pickle.dump(
                {
                    "window": window,
                    "feature_names": list(features.columns),
                    "return_scale": return_scale,
                    "return_scales": return_scales.tolist(),
                    "member_return_scales": [value.tolist() for value in member_scales],
                    "portfolio_weights": portfolio_weights.tolist(),
                    "portfolio_members": member_metadata,
                    "return_bounds": return_bounds,
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
    parser.add_argument("--context-csv", type=Path)
    parser.add_argument("--date-col", default="Asia/Tehran")
    parser.add_argument("--close-col", default="Close")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--forecast-steps", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--portfolio-size", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--recency-half-life", type=float, default=0.5)
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
        args.context_csv,
        window=args.window,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        n_jobs=args.n_jobs,
        portfolio_size=args.portfolio_size,
        recency_half_life=args.recency_half_life,
        news_date_col=args.news_date_col,
        news_impact_col=args.news_impact_col,
        news_half_life_hours=args.news_half_life_hours,
        news_lookback_hours=args.news_lookback_hours,
        interval_minutes=args.interval_minutes,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
