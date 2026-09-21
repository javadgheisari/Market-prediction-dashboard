import sys
import csv
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from django.core.paginator import Paginator

from dashboard.data_paths import DATA_DIR


CODE_DIR = Path(__file__).resolve().parents[2] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def crawl_asset_news(symbol: str) -> dict[str, object]:
    """Run the project's RSS crawler for one dashboard asset."""
    from news_crawler import crawl_asset_news as crawl

    return crawl(symbol, days=7, max_per_asset=100, timeout=20.0)


def load_asset_news(
    symbol: str, page: int | str = 1, per_page: int = 25
) -> dict[str, object]:
    """Load one page of locally saved headlines for dashboard display."""
    path = DATA_DIR / "news" / f"{symbol.upper()}_news.csv"
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
