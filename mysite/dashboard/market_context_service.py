import csv
import sys
from pathlib import Path

from dashboard.data_paths import DATA_DIR


CODE_DIR = Path(__file__).resolve().parents[2] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def asset_context_path(symbol: str) -> Path:
    return DATA_DIR / "context" / f"{symbol.strip().upper()}_context.csv"


def collect_asset_context(symbol: str) -> dict[str, object]:
    from market_context import collect_asset_context as collect

    return collect(symbol)


def load_context_status(symbol: str) -> dict[str, object]:
    path = asset_context_path(symbol)
    if not path.is_file():
        return {"available": False, "rows": 0, "sources": [], "path": path}
    row_count = 0
    sources = set()
    latest_at = ""
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            if row.get("source"):
                sources.add(row["source"])
            latest_at = max(latest_at, row.get("available_at", ""))
    return {
        "available": row_count > 0,
        "rows": row_count,
        "sources": sorted(sources),
        "latest_at": latest_at,
        "path": path,
    }
