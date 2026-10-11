import sys
import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from django.core.paginator import Paginator

from dashboard.data_paths import DATA_DIR


CODE_DIR = Path(__file__).resolve().parents[2] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def asset_news_path(symbol: str) -> Path:
    return DATA_DIR / "news" / f"{symbol.upper()}_news.csv"


def crawl_asset_news(symbol: str) -> dict[str, object]:
    """Run the project's RSS crawler for one dashboard asset."""
    from news_crawler import crawl_asset_news as crawl

    return crawl(symbol, days=7, max_per_asset=100, timeout=20.0)


def clear_asset_news(symbol: str) -> bool:
    """Delete locally saved headlines for one asset, if they exist."""
    path = asset_news_path(symbol)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def refresh_market_news(symbols: list[str] | tuple[str, ...]) -> dict[str, object]:
    """Replace saved news and rank every asset in one dashboard market.

    Clearing is completed as a separate first phase so a successful crawl can
    never be merged with stale rows.  An error for one asset is retained in the
    result instead of preventing recommendations for the rest of the market.
    """
    normalized_symbols = list(
        dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol.strip())
    )
    if not normalized_symbols:
        raise ValueError("The selected market does not contain any assets.")

    results: dict[str, dict[str, object]] = {
        symbol: {
            "symbol": symbol,
            "cleared": False,
            "fetched": 0,
            "saved": 0,
        }
        for symbol in normalized_symbols
    }

    # Clear the whole selected market before fetching any replacement data.
    for symbol in normalized_symbols:
        try:
            clear_asset_news(symbol)
        except OSError as exc:
            results[symbol]["error"] = f"Could not clear saved news: {exc}"
        else:
            results[symbol]["cleared"] = True

    for symbol in normalized_symbols:
        item = results[symbol]
        if item.get("error"):
            continue
        try:
            crawl_result = crawl_asset_news(symbol)
            item.update(
                {
                    "fetched": int(crawl_result.get("fetched", 0)),
                    "saved": int(crawl_result.get("saved", 0)),
                }
            )
            decision = predict_news_buy_decision(symbol)
        except Exception as exc:
            # Feed/network/scorer failures are asset-specific; keep ranking the
            # other symbols and make this failure visible on the result page.
            item["error"] = str(exc)
            continue

        item.update(decision)
        item["ranking_score"] = round(
            float(decision["signal_score"])
            * (float(decision["confidence_percent"]) / 100.0),
            4,
        )

    ranked = sorted(
        (item for item in results.values() if "ranking_score" in item),
        key=lambda item: (
            -float(item["ranking_score"]),
            -float(item["signal_score"]),
            str(item["symbol"]),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    failed = sorted(
        (item for item in results.values() if "ranking_score" not in item),
        key=lambda item: str(item["symbol"]),
    )
    recommendations = ranked + failed
    return {
        "recommendations": recommendations,
        "asset_count": len(normalized_symbols),
        "cleared_count": sum(bool(item["cleared"]) for item in results.values()),
        "analyzed_count": len(ranked),
        "buy_count": sum(bool(item.get("should_buy")) for item in ranked),
        "error_count": len(failed),
    }


def _bounded_number(
    row: dict[str, str], name: str, default: float, lower: float, upper: float
) -> float:
    try:
        value = float(row.get(name, default) or default)
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))


def predict_news_buy_decision(
    symbol: str, *, now: datetime | None = None
) -> dict[str, object]:
    """Return a buy/no-buy decision using saved news signals only."""
    path = asset_news_path(symbol)
    if not path.is_file():
        raise FileNotFoundError(
            f"No crawled news file exists for {symbol.upper()}. Crawl news first."
        )

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    signals = []
    for row in rows:
        try:
            published_at = datetime.fromisoformat(
                row.get("published_at", "").replace("Z", "+00:00")
            )
        except (AttributeError, TypeError, ValueError):
            continue
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        else:
            published_at = published_at.astimezone(timezone.utc)
        age_hours = (current_time - published_at).total_seconds() / 3600
        if age_hours < 0:
            continue

        horizon = _bounded_number(
            row, "impact_horizon_hours", 48.0, 1.0, 720.0
        )
        if age_hours > horizon:
            continue

        impact = _bounded_number(row, "asset_impact", 0.0, -1.0, 1.0)
        quality_values = (
            _bounded_number(row, "relevance", 0.5, 0.0, 1.0),
            _bounded_number(row, "importance", 0.5, 0.0, 1.0),
            _bounded_number(row, "source_quality", 0.5, 0.0, 1.0),
            _bounded_number(row, "confidence", 0.35, 0.0, 1.0),
            _bounded_number(row, "novelty", 0.5, 0.0, 1.0),
        )
        quality = math.prod(quality_values) ** (1 / len(quality_values))
        recency = 0.5 ** (age_hours / max(1.0, horizon / 2))
        weight = quality * recency
        if weight > 0:
            signals.append(
                {
                    "impact": impact,
                    "weight": weight,
                    "headline": row.get("headline", ""),
                    "published_at": published_at,
                }
            )

    if not signals:
        raise ValueError(
            f"No active news signals are available for {symbol.upper()}. "
            "Refresh the news and try again."
        )

    total_weight = sum(signal["weight"] for signal in signals)
    score = sum(
        signal["impact"] * signal["weight"] for signal in signals
    ) / total_weight
    disagreement = sum(
        abs(signal["impact"] - score) * signal["weight"] for signal in signals
    ) / total_weight
    agreement = max(0.0, 1.0 - disagreement / 2)
    evidence = 1.0 - math.exp(-total_weight / 1.5)
    confidence = agreement * evidence
    should_buy = score >= 0.15 and confidence >= 0.20

    if should_buy:
        explanation = (
            "The active, quality-weighted news signal is sufficiently bullish."
        )
    elif score < 0:
        explanation = (
            "The active news signal is bearish, so the model does not support a buy."
        )
    else:
        explanation = (
            "The bullish news evidence is not strong or consistent enough to "
            "support a buy."
        )

    strongest = max(
        signals, key=lambda signal: abs(signal["impact"] * signal["weight"])
    )
    return {
        "decision": "BUY" if should_buy else "DO NOT BUY",
        "should_buy": should_buy,
        "signal_score": round(score, 4),
        "confidence_percent": round(confidence * 100, 1),
        "articles_analyzed": len(signals),
        "articles_saved": len(rows),
        "strongest_headline": strongest["headline"],
        "explanation": explanation,
    }


def load_asset_news(
    symbol: str, page: int | str = 1, per_page: int = 25
) -> dict[str, object]:
    """Load one page of locally saved headlines for dashboard display."""
    path = asset_news_path(symbol)
    if not path.is_file():
        return {
            "rows": [],
            "total": 0,
            "path": path,
            "page": 1,
            "num_pages": 1,
            "start_index": 0,
            "end_index": 0,
            "has_previous": False,
            "has_next": False,
            "page_range": [1],
        }

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    rows.sort(key=lambda row: row.get("published_at", ""), reverse=True)
    paginator = Paginator(rows, per_page)
    page_obj = paginator.get_page(page)
    display_rows = []
    for row in page_obj.object_list:
        raw_url = row.get("url", "")
        parsed_url = urlparse(raw_url)
        safe_url = raw_url if parsed_url.scheme in {"http", "https"} else ""
        raw_date = row.get("published_at", "")
        try:
            published_at = datetime.fromisoformat(
                raw_date.replace("Z", "+00:00")
            ).strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            published_at = raw_date
        try:
            impact = float(row.get("asset_impact", 0) or 0)
        except ValueError:
            impact = 0.0
        display_rows.append(
            {
                "published_iso": raw_date,
                "published_at": published_at,
                "headline": row.get("headline", ""),
                "source": row.get("source", ""),
                "url": safe_url,
                "impact": impact,
            }
        )
    return {
        "rows": display_rows,
        "total": len(rows),
        "path": path,
        "page": page_obj.number,
        "num_pages": paginator.num_pages,
        "start_index": page_obj.start_index(),
        "end_index": page_obj.end_index(),
        "has_previous": page_obj.has_previous(),
        "has_next": page_obj.has_next(),
        "previous_page": page_obj.previous_page_number()
        if page_obj.has_previous()
        else None,
        "next_page": page_obj.next_page_number() if page_obj.has_next() else None,
        "page_range": list(paginator.page_range),
    }
