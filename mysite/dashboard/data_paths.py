from pathlib import Path
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATE_COLUMNS = ("Date", "Time", "Asia/Tehran", "UTC", "datetime", "timestamp")

SYMBOL_DIRS = {
    "EUR": ("assets/eur", "forex"),
    "GBP": ("assets/gbp", "forex"),
    "BTC": ("assets/btc", "crypto/btc"),
    "ETH": ("assets/eth", "crypto/eth"),
    "XAU": ("assets/xau", "gold", "crypto/xau"),
    "XAG": ("assets/xag", "gold", "crypto/xau"),
}

# Assets shown in the dashboard even when the user needs to upload the price
# history. BTC and ETH retain their bundled datasets; the rest share the same
# forecasting and news-crawling workflow once a CSV is supplied.
SUPPORTED_CRYPTO_ASSETS = (
    "AAVE",
    "ADA",
    "ATOM",
    "AVAX",
    "BCH",
    "BNB",
    "BTC",
    "DOGE",
    "DOT",
    "ETC",
    "ETH",
    "FIL",
    "ICP",
    "LINK",
    "LTC",
    "NEAR",
    "SOL",
    "TRX",
    "UNI",
    "XLM",
    "XRP",
)

# Keep the dashboard's existing short-symbol convention: each FX symbol maps
# to its conventional USD major pair (JPY means USDJPY, while EUR means
# EURUSD). The provider-specific pair mapping lives in the data services.
SUPPORTED_FX_MAJOR_ASSETS = (
    "AUD",
    "CAD",
    "CHF",
    "EUR",
    "GBP",
    "JPY",
    "NZD",
)

SUPPORTED_COMMODITY_ASSETS = (
    "BRENT",
    "USOIL",
    "XAG",
    "XAU",
)

CATEGORY_NAMES = {
    "forex": "FX majors",
    "crypto": "Cryptocurrency",
    "gold": "Commodities",
    "commodities": "Commodities",
}


def list_available_assets() -> dict[str, list[str]]:
    """Return all supported markets plus assets discovered from bundled CSVs."""
    grouped: dict[str, set[str]] = {}

    for path in DATA_DIR.rglob("*.csv"):
        relative = path.relative_to(DATA_DIR)
        if not relative.parts or relative.parts[0] == "news":
            continue
        match = re.match(r"([A-Za-z0-9]+)(?:-[A-Za-z0-9]+)?", path.stem)
        if not match:
            continue
        symbol = match.group(1).upper()
        # Compact pair names such as XAUUSD and EURUSD represent the base asset.
        if symbol.endswith("USD") and len(symbol) > 3:
            symbol = symbol[:-3]
        if symbol in SUPPORTED_COMMODITY_ASSETS:
            category = "Commodities"
        else:
            category = CATEGORY_NAMES.get(relative.parts[0], "Other")
        grouped.setdefault(category, set()).add(symbol)

    grouped.setdefault("Cryptocurrency", set()).update(SUPPORTED_CRYPTO_ASSETS)
    grouped.setdefault("FX majors", set()).update(SUPPORTED_FX_MAJOR_ASSETS)
    grouped.setdefault("Commodities", set()).update(SUPPORTED_COMMODITY_ASSETS)
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


def has_bundled_data(symbol: str) -> bool:
    """Return whether a project price CSV is available for a symbol."""
    try:
        resolve_data_csv(symbol)
    except FileNotFoundError:
        return False
    return True


def get_model_dir(model_name: str, symbol: str) -> Path:
    return MODELS_DIR / model_name / symbol.upper()


def load_price_data(path: Path) -> pd.DataFrame:
    """Load and normalize a price CSV into Date/Close columns used by the dashboard."""
    # Broker exports commonly use tabs or semicolons despite retaining a .csv
    # extension.  Let pandas sniff the delimiter instead of assuming commas.
    df = pd.read_csv(path, sep=None, engine="python")
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
