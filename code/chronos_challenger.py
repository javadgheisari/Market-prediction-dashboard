"""Chronos-2 zero-shot forecasting and held-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sequence_forecaster import (
    build_news_features,
    calculate_metrics,
    extract_asset_symbol,
    infer_interval,
    load_market_data,
    load_context_features,
    load_news_data,
    make_future_dates,
)


DEFAULT_MODEL_ID = "amazon/chronos-2"
QUANTILES = [0.1, 0.5, 0.9]


def _known_calendar_features(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    values = pd.DatetimeIndex(dates)
    minutes = values.hour * 60 + values.minute
    return pd.DataFrame(
        {
            "time_sin": np.sin(2 * np.pi * minutes / 1440.0),
            "time_cos": np.cos(2 * np.pi * minutes / 1440.0),
            "weekday_sin": np.sin(2 * np.pi * values.dayofweek / 7.0),
            "weekday_cos": np.cos(2 * np.pi * values.dayofweek / 7.0),
        }
    )


def _chronos_frame(
    dates: pd.Series | pd.DatetimeIndex,
    returns: np.ndarray | None,
    item_id: str,
    covariates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame({"id": item_id, "timestamp": pd.DatetimeIndex(dates)})
    if returns is not None:
        frame["target"] = np.asarray(returns, dtype=float)
    pieces = [frame.reset_index(drop=True), _known_calendar_features(dates)]
    if covariates is not None:
        pieces.append(covariates.reset_index(drop=True).astype(float))
    return pd.concat(pieces, axis=1)


def _load_pipeline(model_id: str, device_map: str):
    try:
        from chronos import Chronos2Pipeline
    except ImportError as error:
        raise RuntimeError(
            "Chronos-2 is not installed. Run: pip install chronos-forecasting"
        ) from error
    return Chronos2Pipeline.from_pretrained(model_id, device_map=device_map)


def _predict(
    pipeline: Any,
    context: pd.DataFrame,
    future: pd.DataFrame,
    prediction_length: int,
) -> pd.DataFrame:
    prediction = pipeline.predict_df(
        context,
        future_df=future,
        prediction_length=prediction_length,
        quantile_levels=QUANTILES,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )
    required = {"predictions", "0.1", "0.5", "0.9"}
    missing = required.difference(prediction.columns)
    if missing:
        raise RuntimeError(f"Chronos response is missing columns: {sorted(missing)}")
    return prediction.sort_values("timestamp").reset_index(drop=True)


def _price_paths(last_close: float, prediction: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        column: last_close
        * np.exp(np.cumsum(prediction[column].to_numpy(dtype=float)))
        for column in ("predictions", "0.1", "0.5", "0.9")
    }


def _direction_probabilities(prediction: pd.DataFrame) -> dict[str, Any]:
    cumulative = {
        column: np.cumsum(prediction[column].to_numpy(dtype=float))
        for column in ("0.1", "0.5", "0.9")
    }
    lower = float(cumulative["0.1"][-1])
    upper = float(cumulative["0.9"][-1])
    if upper <= lower:
        bullish = 100.0 if cumulative["0.5"][-1] > 0 else 0.0
    elif 0 <= lower:
        bullish = 90.0
    elif 0 >= upper:
        bullish = 10.0
    else:
        bearish = 10.0 + 80.0 * (0.0 - lower) / (upper - lower)
        bullish = 100.0 - bearish
    bullish = float(np.clip(bullish, 0.0, 100.0))
    return {
        "bullish_percent": bullish,
        "bearish_percent": 100.0 - bullish,
        "horizon_steps": len(prediction),
        "method": "Chronos cumulative p10/p50/p90 interpolation",
        "cumulative_log_return_quantiles": {
            "p10": cumulative["0.1"].tolist(),
            "p50": cumulative["0.5"].tolist(),
            "p90": cumulative["0.9"].tolist(),
        },
    }


def train_and_forecast_chronos(
    csv_path: Path,
    date_col: str,
    close_col: str = "Close",
    forecast_steps: int = 24,
    news_csv_path: Path | None = None,
    context_csv_path: Path | None = None,
    *,
    split_ratio: float = 0.8,
    context_length: int = 512,
    model_id: str = DEFAULT_MODEL_ID,
    device_map: str = "cpu",
    news_half_life_hours: float = 24.0,
    news_lookback_hours: float = 168.0,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    """Run a held-out Chronos evaluation and a forecast from all known data."""
    if forecast_steps < 1 or context_length < 24:
        raise ValueError("forecast_steps must be positive and context_length at least 24.")
    if not 0.5 <= split_ratio < 1:
        raise ValueError("split_ratio must be in [0.5, 1).")
    if device_map not in {"cpu", "cuda", "auto"}:
        raise ValueError("device_map must be cpu, cuda, or auto.")

    market = load_market_data(csv_path, date_col, close_col)
    dates = market[date_col]
    close = market[close_col].to_numpy(dtype=float)
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    split_index = int(len(market) * split_ratio)
    evaluation_steps = min(max(forecast_steps, 48), len(market) - split_index)
    if split_index < 24 or evaluation_steps < 2:
        raise ValueError("Dataset is too short for the requested Chronos holdout.")

    news_metadata = None
    news_features = None
    if news_csv_path is not None:
        news, news_metadata = load_news_data(
            news_csv_path, dates, csv_path.stem
        )
        news_features = build_news_features(
            dates, news, news_half_life_hours, news_lookback_hours
        )
        news_metadata["feature_coverage_percent"] = float(
            np.mean(news_features.abs().sum(axis=1) > 0) * 100
        )

    context_metadata = None
    context_features = None
    if context_csv_path is not None:
        context_features, context_metadata = load_context_features(
            context_csv_path, dates
        )
    all_covariates = None
    if news_features is not None or context_features is not None:
        all_covariates = pd.concat(
            [
                frame
                for frame in (news_features, context_features)
                if frame is not None
            ],
            axis=1,
        )

    loaded_pipeline = pipeline or _load_pipeline(model_id, device_map)
    item_id = csv_path.stem
    context_start = max(0, split_index - context_length)
    context_covariates = (
        all_covariates.iloc[context_start:split_index]
        if all_covariates is not None
        else None
    )
    test_context = _chronos_frame(
        dates.iloc[context_start:split_index],
        returns[context_start:split_index],
        item_id,
        context_covariates,
    )
    test_dates = dates.iloc[split_index : split_index + evaluation_steps]
    # News published after the forecast origin is not knowable, so test and
    # future news covariates are zero. This keeps the evaluation causal.
    test_covariates = (
        pd.DataFrame(
            0.0, index=range(evaluation_steps), columns=all_covariates.columns
        )
        if all_covariates is not None
        else None
    )
    test_future = _chronos_frame(test_dates, None, item_id, test_covariates)
    test_prediction = _predict(
        loaded_pipeline, test_context, test_future, evaluation_steps
    )
    test_paths = _price_paths(close[split_index - 1], test_prediction)
    actual_test = close[split_index : split_index + evaluation_steps]
    baseline_test = np.repeat(close[split_index - 1], evaluation_steps)
    model_metrics = calculate_metrics(actual_test, test_paths["predictions"])
    baseline_metrics = calculate_metrics(actual_test, baseline_test)
    actual_direction = np.sign(np.diff(actual_test, prepend=close[split_index - 1]))
    predicted_direction = np.sign(
        np.diff(test_paths["predictions"], prepend=close[split_index - 1])
    )
    model_metrics["directional_accuracy_percent"] = float(
        np.mean(actual_direction == predicted_direction) * 100
    )
    model_metrics["mae_skill_vs_naive_percent"] = float(
        (1 - model_metrics["mae"] / baseline_metrics["mae"]) * 100
        if baseline_metrics["mae"]
        else 0.0
    )

    interval = infer_interval(dates)
    future_dates = make_future_dates(dates, forecast_steps, interval)
    full_context_start = max(0, len(market) - context_length)
    full_covariates = (
        all_covariates.iloc[full_context_start:]
        if all_covariates is not None else None
    )
    full_context = _chronos_frame(
        dates.iloc[full_context_start:],
        returns[full_context_start:],
        item_id,
        full_covariates,
    )
    future_covariates = (
        pd.DataFrame(0.0, index=range(forecast_steps), columns=all_covariates.columns)
        if all_covariates is not None
        else None
    )
    future_frame = _chronos_frame(
        future_dates, None, item_id, future_covariates
    )
    future_prediction = _predict(
        loaded_pipeline, full_context, future_frame, forecast_steps
    )
    future_paths = _price_paths(close[-1], future_prediction)
    probabilities = _direction_probabilities(future_prediction)
    actual_context_start = max(0, split_index - 60)
    return {
        "model": "CHRONOS_2",
        "model_id": model_id,
        "device_map": device_map,
        "rows": len(market),
        "test_samples": evaluation_steps,
        "context_length": context_length,
        "news": news_metadata,
        "context": context_metadata,
        "validation_selected_return_scale": 1.0,
        "validation_selected_return_scales": [1.0] * forecast_steps,
        "model_metrics": model_metrics,
        "naive_last_price_baseline": baseline_metrics,
        "eligible_for_ensemble": model_metrics["mae"] < baseline_metrics["mae"],
        "forecast_direction_probability": probabilities,
        "chart": {
            "title": f"{extract_asset_symbol(csv_path.stem)} price forecast (Chronos-2)",
            "actual_dates": [
                value.strftime("%Y-%m-%d %H:%M")
                for value in dates.iloc[actual_context_start:]
            ],
            "actual_prices": close[actual_context_start:].tolist(),
            "test_dates": [value.strftime("%Y-%m-%d %H:%M") for value in test_dates],
            "test_predictions": test_paths["predictions"].tolist(),
            "future_dates": [
                value.strftime("%Y-%m-%d %H:%M") for value in future_dates
            ],
            "future_predictions": future_paths["predictions"].tolist(),
            "future_lower": future_paths["0.1"].tolist(),
            "future_median": future_paths["0.5"].tolist(),
            "future_upper": future_paths["0.9"].tolist(),
            "forecast_interval_percent": 80,
        },
    }


def evaluate_chronos(
    csv_path: Path,
    date_col: str,
    close_col: str = "Close",
    *,
    forecast_steps: int = 24,
    split_ratio: float = 0.8,
    model_id: str = DEFAULT_MODEL_ID,
    device_map: str = "cpu",
) -> dict[str, Any]:
    result = train_and_forecast_chronos(
        csv_path,
        date_col,
        close_col,
        forecast_steps,
        split_ratio=split_ratio,
        model_id=model_id,
        device_map=device_map,
    )
    return {
        key: result[key]
        for key in (
            "model_id",
            "device_map",
            "test_samples",
            "model_metrics",
            "naive_last_price_baseline",
            "eligible_for_ensemble",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("price_csv", type=Path)
    parser.add_argument("--news-csv", type=Path)
    parser.add_argument("--context-csv", type=Path)
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--close-col", default="Close")
    parser.add_argument("--forecast-steps", type=int, default=24)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device-map", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    result = train_and_forecast_chronos(
        args.price_csv,
        args.date_col,
        args.close_col,
        args.forecast_steps,
        news_csv_path=args.news_csv,
        context_csv_path=args.context_csv,
        split_ratio=args.split_ratio,
        context_length=args.context_length,
        model_id=args.model_id,
        device_map=args.device_map,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
