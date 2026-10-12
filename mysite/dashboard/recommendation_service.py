import math
from typing import Any

from dashboard import forecaster, forexsb_service, news_service
from dashboard.data_paths import has_bundled_data
from dashboard.forms import MODEL_CHOICES


DEFAULT_ADVANCED_CONFIG = {
    "window": 60,
    "split_ratio": 0.8,
    "validation_ratio": 0.2,
    "portfolio_size": 3,
    "recency_half_life": 0.5,
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "epochs": 50,
    "chronos_model_id": "amazon/chronos-2",
    "chronos_context_length": 512,
    "chronos_device": "cpu",
}


def _normalized_symbols(symbols) -> list[str]:
    return list(
        dict.fromkeys(
            str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
        )
    )


def _price_source(symbol: str) -> str:
    if has_bundled_data(symbol):
        return "bundled"
    if forexsb_service.supports_symbol(symbol):
        return "forexsb"
    raise FileNotFoundError(
        f"No automatic price history is available for {symbol}. "
        "Run this asset from its forecast page with an uploaded CSV."
    )


def _bounded_percent(value: object, default: float = 50.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return max(0.0, min(100.0, number))


def _forecast_recommendation(
    symbol: str,
    model_name: str,
    predict_steps: int,
    use_news: bool,
) -> dict[str, Any]:
    price_source = _price_source(symbol)
    chart = forecaster.run_dashboard_forecast(
        symbol=symbol,
        model_name=model_name,
        predict_steps=predict_steps,
        price_csv=None,
        price_source=price_source,
        use_news=use_news,
        advanced_config=DEFAULT_ADVANCED_CONFIG.copy(),
    )
    actual_prices = chart.get("actual_prices") or []
    future_prices = chart.get("future_predictions") or []
    if not actual_prices or not future_prices:
        raise ValueError("The model returned no prices that can be ranked.")

    current_price = float(actual_prices[-1])
    target_price = float(future_prices[-1])
    if (
        not math.isfinite(current_price)
        or not math.isfinite(target_price)
        or current_price <= 0
        or target_price <= 0
    ):
        raise ValueError("The model returned an invalid price target.")

    metrics = chart.get("metrics") or {}
    expected_return = ((target_price / current_price) - 1.0) * 100.0
    bullish_percent = _bounded_percent(metrics.get("bullish_percent"))
    directional_accuracy = _bounded_percent(
        metrics.get("directional_accuracy_percent")
    )
    confidence = (0.65 * bullish_percent) + (0.35 * directional_accuracy)
    forecast_is_naive = bool(metrics.get("forecast_is_naive", False))
    should_buy = (
        not forecast_is_naive
        and expected_return > 0
        and bullish_percent >= 55.0
        and confidence >= 55.0
    )

    return {
        "decision": "BUY" if should_buy else "DO NOT BUY",
        "should_buy": should_buy,
        "ranking_score": round(expected_return * confidence / 100.0, 4),
        "expected_return_percent": round(expected_return, 4),
        "confidence_percent": round(confidence, 1),
        "bullish_percent": round(bullish_percent, 1),
        "directional_accuracy_percent": round(directional_accuracy, 1),
        "current_price": current_price,
        "target_price": target_price,
        "forecast_is_naive": forecast_is_naive,
        "price_source": price_source,
        "news_enabled": bool(metrics.get("news_enabled", use_news)),
    }


def _saved_news_recommendations(symbols: list[str]) -> dict[str, object]:
    recommendations = []
    for symbol in symbols:
        item: dict[str, object] = {"symbol": symbol, "fetched": 0, "saved": 0}
        try:
            decision = news_service.predict_news_buy_decision(symbol)
        except Exception as exc:
            item["error"] = str(exc)
        else:
            item.update(decision)
            item["ranking_score"] = round(
                float(decision["signal_score"])
                * (float(decision["confidence_percent"]) / 100.0),
                4,
            )
        recommendations.append(item)
    return _rank_result(recommendations, len(symbols), 0)


def _rank_result(
    recommendations: list[dict[str, object]],
    asset_count: int,
    refreshed_count: int,
) -> dict[str, object]:
    ranked = sorted(
        (item for item in recommendations if "ranking_score" in item),
        key=lambda item: (-float(item["ranking_score"]), str(item["symbol"])),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    failed = sorted(
        (item for item in recommendations if "ranking_score" not in item),
        key=lambda item: str(item["symbol"]),
    )
    return {
        "recommendations": ranked + failed,
        "asset_count": asset_count,
        "refreshed_count": refreshed_count,
        "analyzed_count": len(ranked),
        "buy_count": sum(bool(item.get("should_buy")) for item in ranked),
        "error_count": len(failed),
    }


def rank_buy_recommendations(
    symbols,
    *,
    model_name: str,
    predict_steps: int = 24,
    use_news: bool = True,
    refresh_news: bool = True,
) -> dict[str, object]:
    """Rank selected assets using either news alone or a forecast model."""
    normalized = _normalized_symbols(symbols)
    if not normalized:
        raise ValueError("Choose at least one asset to rank.")
    model_labels = dict(MODEL_CHOICES)
    if model_name not in model_labels:
        raise ValueError(f"Unsupported recommendation model '{model_name}'.")
    if model_name == "news_buy":
        use_news = True
        if refresh_news:
            result = news_service.refresh_market_news(normalized)
            result["refreshed_count"] = result.pop("cleared_count", 0)
        else:
            result = _saved_news_recommendations(normalized)
        return {
            **result,
            "mode": "news_only",
            "model_name": model_name,
            "model_label": model_labels[model_name],
            "use_news": True,
            "predict_steps": None,
        }

    refresh_results: dict[str, dict[str, object]] = {
        symbol: {"fetched": 0, "saved": 0} for symbol in normalized
    }
    refreshed_count = 0
    if use_news and refresh_news:
        for symbol in normalized:
            try:
                news_service.clear_asset_news(symbol)
            except OSError as exc:
                refresh_results[symbol]["error"] = (
                    f"Could not clear saved news: {exc}"
                )
            else:
                refreshed_count += 1
        for symbol in normalized:
            if refresh_results[symbol].get("error"):
                continue
            try:
                crawl = news_service.crawl_asset_news(symbol)
            except Exception as exc:
                refresh_results[symbol]["error"] = str(exc)
            else:
                refresh_results[symbol].update(
                    fetched=int(crawl.get("fetched", 0)),
                    saved=int(crawl.get("saved", 0)),
                )

    recommendations = []
    for symbol in normalized:
        item: dict[str, object] = {
            "symbol": symbol,
            **refresh_results[symbol],
        }
        if item.get("error"):
            recommendations.append(item)
            continue
        try:
            item.update(
                _forecast_recommendation(
                    symbol,
                    model_name=model_name,
                    predict_steps=predict_steps,
                    use_news=use_news,
                )
            )
        except Exception as exc:
            item["error"] = str(exc)
        recommendations.append(item)

    result = _rank_result(recommendations, len(normalized), refreshed_count)
    return {
        **result,
        "mode": "forecast",
        "model_name": model_name,
        "model_label": model_labels[model_name],
        "use_news": use_news,
        "predict_steps": predict_steps,
    }
