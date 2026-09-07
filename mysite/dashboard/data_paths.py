from pathlib import Path
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATE_COLUMNS = ("Date", "Asia/Tehran", "UTC", "datetime", "timestamp")

SYMBOL_DIRS = {
    "EUR": ("forex",),
    "GBP": ("forex",),
    "BTC": ("crypto/btc",),
    "ETH": ("crypto/eth",),
    "XAU": ("gold", "crypto/xau"),
    "XAG": ("gold", "crypto/xau"),
}

CATEGORY_NAMES = {
    "forex": "Forex",
    "crypto": "Cryptocurrency",
    "gold": "Precious metals",
}


def list_available_assets() -> dict[str, list[str]]:
    """Return only assets which have a market CSV under the project data dir."""
    grouped: dict[str, set[str]] = {}
    if not DATA_DIR.exists():
        return {}

    for path in DATA_DIR.rglob("*.csv"):
        relative = path.relative_to(DATA_DIR)
        if not relative.parts or relative.parts[0] == "news":
            continue
        match = re.match(r"([A-Za-z0-9]+)(?:-[A-Za-z0-9]+)?", path.stem)
        if not match:
            continue
        symbol = match.group(1).upper()
        if symbol in {"XAU", "XAG"}:
            category = "Precious metals"
        else:
            category = CATEGORY_NAMES.get(relative.parts[0], "Other")
        grouped.setdefault(category, set()).add(symbol)

    return {
        category: sorted(symbols)
        for category, symbols in sorted(grouped.items())
    }


def resolve_data_csv(symbol: str) -> Path:
    """Find the best matching market CSV for a dashboard symbol under /data."""
    symbol = symbol.upper()
    configured_dirs = SYMBOL_DIRS.get(symbol)
    if configured_dirs:
        search_roots = [DATA_DIR / directory for directory in configured_dirs]
    else:
        search_roots = [
            path for path in DATA_DIR.iterdir() if path.is_dir() and path.name != "news"
        ]

    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            name = path.name.upper()
            if symbol in name or f"{symbol}-USD" in name:
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No market CSV found for '{symbol}' under {DATA_DIR}. "
            f"Add a file whose name contains '{symbol}' or '{symbol}-USD'."
        )

    def rank(path: Path) -> tuple:
        # Filename conventions vary (XAUUSD, XAU-USD, and XAU), so punctuation
        # must not make a short export outrank a substantially longer history.
        # Size is a cheap, deterministic proxy for the number of candles.
        data_size = path.stat().st_size
        category_boost = (
            1 if symbol in {"XAU", "XAG"} and "gold" in path.parts else 0
        )
        return (category_boost, data_size, path.stat().st_mtime)

    return max(candidates, key=rank)


def get_model_dir(model_name: str, symbol: str) -> Path:
    return MODELS_DIR / model_name / symbol.upper()


def load_price_data(path: Path) -> pd.DataFrame:
    """Load and normalize a price CSV into Date/Close columns used by the dashboard."""
    df = pd.read_csv(path)
    date_col = next((column for column in DATE_COLUMNS if column in df.columns), None)
    if date_col is None:
        raise ValueError(
            f"No date column found in {path}. Expected one of {DATE_COLUMNS}."
        )
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    if "Close" not in df.columns:
        raise ValueError(f"No Close column found in {path}.")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Date"] = pd.to_datetime(
        df["Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    df = (
        df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M")
    return df
