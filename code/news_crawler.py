"""Collect asset-related headlines from RSS search feeds into per-asset CSVs.

The default provider is Google News' public RSS search endpoint.  Only feed
metadata is stored; this script does not scrape article bodies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "news_assets.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "news"
DEFAULT_FEED_TEMPLATE = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)
CSV_FIELDS = (
    "published_at",
    "headline",
    "asset_impact",
    "relevance",
    "importance",
    "source_quality",
    "confidence",
    "source",
    "url",
    "article_id",
)

POSITIVE_WORDS = {
    "beat", "beats", "bullish", "gain", "gains", "growth", "higher",
    "jump", "jumps", "outperform", "profit", "rally", "record", "rise",
    "rises", "surge", "surges", "upgrade", "upgraded",
}
NEGATIVE_WORDS = {
    "bearish", "crash", "cut", "cuts", "decline", "declines", "downgrade",
    "downgraded", "drop", "drops", "fall", "falls", "fraud", "hack",
    "lawsuit", "loss", "miss", "misses", "plunge", "recall", "slump",
}
HIGH_IMPORTANCE_WORDS = {
    "acquisition", "bankruptcy", "earnings", "etf", "fed", "federal reserve",
    "inflation", "merger", "regulation", "sec", "tariff",
}


@dataclass(frozen=True)
class Article:
    published_at: datetime
    headline: str
    source: str
    url: str


def clean_text(value: str | None) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    return " ".join(text.split())


def child_text(element: ET.Element, name: str) -> str:
    """Read a child by local name, allowing RSS and Atom namespaces."""
    for child in element:
        if child.tag.rsplit("}", 1)[-1].lower() == name.lower():
            return clean_text(child.text)
    return ""


def parse_date(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_feed(payload: bytes) -> list[Article]:
    root = ET.fromstring(payload)
    articles: list[Article] = []
    for element in root.iter():
        kind = element.tag.rsplit("}", 1)[-1].lower()
        if kind not in {"item", "entry"}:
            continue
        headline = child_text(element, "title")
        published = child_text(element, "pubDate") or child_text(element, "published")
        published = published or child_text(element, "updated")
        source = child_text(element, "source")
        url = child_text(element, "link")
        if not url:
            for child in element:
                if child.tag.rsplit("}", 1)[-1].lower() == "link":
                    url = child.attrib.get("href", "")
                    if url:
                        break
        timestamp = parse_date(published)
        if headline and url and timestamp:
            articles.append(Article(timestamp, headline, source, url.strip()))
    return articles


def fetch_feed(session: requests.Session, url: str, timeout: float) -> list[Article]:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return parse_feed(response.content)


def term_matches(text: str, term: str) -> bool:
    normalized_term = term.casefold()
    if normalized_term.isalnum() and len(normalized_term) <= 5:
        return bool(re.search(rf"\b{re.escape(normalized_term)}\b", text.casefold()))
    return normalized_term in text.casefold()


def is_relevant(headline: str, terms: Iterable[str]) -> bool:
    return any(term_matches(headline, term) for term in terms)


def score_headline(headline: str, terms: Iterable[str]) -> tuple[float, float, float, float]:
    """Return transparent, deliberately low-confidence lexical estimates."""
    normalized = headline.casefold()
    words = set(re.findall(r"[a-z0-9]+", normalized))
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)
    impact = max(-1.0, min(1.0, (positive - negative) / 3.0))
    matched = sum(1 for term in terms if term_matches(normalized, term))
    relevance = min(1.0, 0.55 + 0.15 * matched)
    importance = 0.8 if any(word in normalized for word in HIGH_IMPORTANCE_WORDS) else 0.5
    confidence = min(0.65, 0.35 + 0.1 * (positive + negative))
    return impact, relevance, importance, confidence


def article_row(article: Article, terms: list[str]) -> dict[str, str]:
    impact, relevance, importance, confidence = score_headline(article.headline, terms)
    article_id = hashlib.sha256(article.url.encode("utf-8")).hexdigest()[:20]
    return {
        "published_at": article.published_at.isoformat(timespec="seconds"),
        "headline": article.headline,
        "asset_impact": f"{impact:.4f}",
        "relevance": f"{relevance:.2f}",
        "importance": f"{importance:.2f}",
        "source_quality": "0.60",
        "confidence": f"{confidence:.2f}",
        "source": article.source,
        "url": article.url,
        "article_id": article_id,
    }


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{field: row.get(field, "") for field in CSV_FIELDS} for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def merge_rows(existing: list[dict[str, str]], incoming: list[dict[str, str]]) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for row in existing + incoming:
        identifier = row.get("article_id") or hashlib.sha256(
            (row.get("url") or row.get("headline", "")).encode("utf-8")
        ).hexdigest()[:20]
        row["article_id"] = identifier
        by_id[identifier] = row
    return sorted(by_id.values(), key=lambda row: row.get("published_at", ""))


def load_assets(path: Path) -> dict[str, dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    assets = raw.get("assets", raw)
    if not isinstance(assets, dict):
        raise ValueError("Config must contain an 'assets' object.")
    return {str(symbol).upper(): value for symbol, value in assets.items()}


def crawl_asset_news(
    symbol: str,
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    days: int = 7,
    max_per_asset: int = 100,
    timeout: float = 20.0,
    feed_template: str = DEFAULT_FEED_TEMPLATE,
    session: requests.Session | None = None,
) -> dict[str, object]:
    """Fetch, score, merge, and save recent news for one configured asset."""
    if days < 1 or max_per_asset < 1 or timeout <= 0:
        raise ValueError("days, max_per_asset, and timeout must be positive")

    normalized_symbol = symbol.strip().upper()
    assets = load_assets(config_path)
    if normalized_symbol not in assets:
        raise ValueError(f"Unknown asset: {normalized_symbol}")

    definition = assets[normalized_symbol]
    query = str(definition["query"])
    terms = [str(term) for term in definition.get("terms", [normalized_symbol])]
    feed_url = feed_template.format(query=quote_plus(f"{query} when:{days}d"))
    client = session or requests.Session()
    client.headers.setdefault(
        "User-Agent", "MarketPredictionDashboard/1.0 (RSS news collector)"
    )

    articles = fetch_feed(client, feed_url, timeout)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    articles = [
        article
        for article in articles
        if article.published_at >= cutoff and is_relevant(article.headline, terms)
    ]
    articles = sorted(
        articles, key=lambda item: item.published_at, reverse=True
    )[:max_per_asset]
    path = output_dir / f"{normalized_symbol}_news.csv"
    merged = merge_rows(
        read_existing(path),
        [article_row(article, terms) for article in articles],
    )
    write_rows(path, merged)
    return {
        "symbol": normalized_symbol,
        "fetched": len(articles),
        "saved": len(merged),
        "path": path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", help="Symbols to collect; default: every configured asset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--days", type=int, default=7, help="Keep fetched articles this many days old")
    parser.add_argument("--max-per-asset", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between feed requests")
    parser.add_argument("--feed-template", default=DEFAULT_FEED_TEMPLATE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.days < 1 or args.max_per_asset < 1 or args.timeout <= 0 or args.delay < 0:
        raise SystemExit("days/max-per-asset/timeout must be positive and delay cannot be negative")
    assets = load_assets(args.config)
    selected = [symbol.upper() for symbol in (args.assets or assets)]
    unknown = sorted(set(selected) - set(assets))
    if unknown:
        raise SystemExit(f"Unknown assets: {', '.join(unknown)}")

    session = requests.Session()
    session.headers["User-Agent"] = "MarketPredictionDashboard/1.0 (RSS news collector)"
    failures = 0
    for index, symbol in enumerate(selected):
        try:
            result = crawl_asset_news(
                symbol,
                config_path=args.config,
                output_dir=args.output_dir,
                days=args.days,
                max_per_asset=args.max_per_asset,
                timeout=args.timeout,
                feed_template=args.feed_template,
                session=session,
            )
            print(
                f"{symbol}: fetched {result['fetched']}, "
                f"saved {result['saved']} -> {result['path']}"
            )
        except (requests.RequestException, ET.ParseError, OSError, ValueError) as error:
            failures += 1
            print(f"{symbol}: failed: {error}")
        if index + 1 < len(selected) and args.delay:
            time.sleep(args.delay)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
