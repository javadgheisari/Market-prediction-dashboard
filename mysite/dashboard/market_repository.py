"""Import legacy CSV observations and export news for file-based trainers."""

import csv
import hashlib
import math
from datetime import timezone
from pathlib import Path

from django.db import transaction
from django.utils.dateparse import parse_datetime

from dashboard.models import ContextObservation, NewsArticle


NEWS_FIELDS = (
    "published_at", "headline", "asset_impact", "relevance", "importance",
    "source_quality", "confidence", "novelty", "impact_horizon_hours",
    "event_type", "scoring_method", "source", "url", "article_id",
)
CONTEXT_FIELDS = ("available_at", "feature", "value", "source", "max_age_hours")
NEWS_DEFAULTS = {
    "asset_impact": 0, "relevance": 0.5, "importance": 0.5,
    "source_quality": 0.5, "confidence": 0.35, "novelty": 0.5,
    "impact_horizon_hours": 48,
}


def _timestamp(value: str, source: str):
    parsed = parse_datetime(value.strip())
    if parsed is None:
        raise ValueError(f"Invalid timestamp {value!r} in {source}")
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _number(value: str | None, default: float, source: str) -> float:
    try:
        number = float(value) if value not in (None, "") else default
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number {value!r} in {source}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite number {value!r} in {source}")
    return number


def news_row(symbol: str, row: dict[str, str], source: str) -> NewsArticle:
    identifier = row.get("article_id") or hashlib.sha256(
        (row.get("url") or row.get("headline") or "").encode("utf-8")
    ).hexdigest()[:20]
    if not row.get("headline"):
        raise ValueError(f"Missing headline in {source}")
    return NewsArticle(
        symbol=symbol.upper(),
        article_id=identifier,
        published_at=_timestamp(row.get("published_at", ""), source),
        headline=row["headline"],
        **{field: _number(row.get(field), default, source) for field, default in NEWS_DEFAULTS.items()},
        **{field: row.get(field, "") or "" for field in ("event_type", "scoring_method", "source", "url")},
    )


def context_row(symbol: str, row: dict[str, str], source: str) -> ContextObservation:
    if not row.get("feature") or not row.get("source"):
        raise ValueError(f"Missing context feature or source in {source}")
    return ContextObservation(
        symbol=symbol.upper(),
        available_at=_timestamp(row.get("available_at", ""), source),
        feature=row["feature"],
        value=_number(row.get("value"), 0, source),
        source=row["source"],
        max_age_hours=_number(row.get("max_age_hours"), 0, source),
    )


def _read_rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
        return list(reader)


def import_news_file(path: Path) -> int:
    symbol = path.name.removesuffix("_news.csv").upper()
    rows = _read_rows(path, NEWS_FIELDS)
    return save_news_rows(symbol, rows, source=str(path))


def save_news_rows(symbol: str, rows: list[dict[str, str]], *, source: str = "crawler") -> int:
    unique = {item.article_id: item for item in (news_row(symbol, row, source) for row in rows)}
    with transaction.atomic():
        NewsArticle.objects.bulk_create(
            list(unique.values()), batch_size=500, update_conflicts=True,
            update_fields=[field.name for field in NewsArticle._meta.fields if field.name not in {"id", "symbol", "article_id"}],
            unique_fields=["symbol", "article_id"],
        )
    return len(unique)


def import_context_file(path: Path) -> int:
    symbol = path.name.removesuffix("_context.csv").upper()
    rows = _read_rows(path, CONTEXT_FIELDS)
    unique = {
        (item.available_at, item.feature, item.source): item
        for item in (context_row(symbol, row, str(path)) for row in rows)
    }
    with transaction.atomic():
        ContextObservation.objects.bulk_create(
            list(unique.values()), batch_size=500, update_conflicts=True,
            update_fields=["value", "max_age_hours"],
            unique_fields=["symbol", "available_at", "feature", "source"],
        )
    return len(unique)


def news_as_rows(symbol: str) -> list[dict[str, str]]:
    return [
        {
            "published_at": article.published_at.isoformat(timespec="seconds"),
            "headline": article.headline,
            **{field: str(getattr(article, field)) for field in NEWS_DEFAULTS},
            **{field: getattr(article, field) for field in ("event_type", "scoring_method", "source", "url", "article_id")},
        }
        for article in NewsArticle.objects.filter(symbol=symbol.upper()).order_by("published_at", "pk")
    ]


def write_news_csv(symbol: str, path: Path) -> bool:
    rows = news_as_rows(symbol)
    if not rows:
        return False
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NEWS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return True
