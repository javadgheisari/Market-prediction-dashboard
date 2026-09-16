"""Fetch recent BiQuote candles and merge them with dashboard CSV data."""

from __future__ import annotations

import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from dashboard.data_paths import DATE_COLUMNS


API_URL = "https://biquote.io/api/{symbol}/ohlc"
UTC = ZoneInfo("UTC")
REQUEST_TIMEOUT_SECONDS = 15
MAX_BARS = 1000

SYMBOLS = {
    "AUD": "AUDUSD",
    "BTC": "BTCUSD",
    "BRENT": "UKOIL",
    "CAD": "USDCAD",
    "CHF": "USDCHF",
    "ETH": "ETHUSD",
    "EUR": "EURUSD",
    "GBP": "GBPUSD",
    "JPY": "USDJPY",
    "NZD": "NZDUSD",
    "USOIL": "USOIL",
    "XAU": "XAUUSD",
    "XAG": "XAGUSD",
}

INTERVALS = {
    1: "1m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    240: "4h",
    1440: "1d",
}


def api_symbol(symbol: str) -> str:
    """Translate a dashboard base symbol to the corresponding BiQuote pair."""
    normalized = symbol.strip().upper().replace("-", "")
    fallback = normalized if normalized.endswith("USD") else f"{normalized}USD"
    return SYMBOLS.get(normalized, fallback)


def _date_column(frame: pd.DataFrame, path: Path) -> str:
    column = next((name for name in DATE_COLUMNS if name in frame.columns), None)
    if column is None:
        raise ValueError(
            f"No date column found in {path}. Expected one of {DATE_COLUMNS}."
        )
    return column


def _localized_timestamp(value: object, default_timezone: ZoneInfo) -> pd.Timestamp:
    if isinstance(value, str):
        # Dukascopy-style values include a literal ``UTC`` before their offset,
        # which pandas does not parse consistently across versions.
        value = re.sub(r"\s+UTC(?=[+-]\d{2}:?\d{2}$)", "", value.strip())
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(default_timezone)
    return timestamp.tz_convert(default_timezone)


def _interval(frame: pd.DataFrame, date_column: str, timezone: ZoneInfo) -> str:
    parsed = []
    for value in frame[date_column].dropna().tail(500):
        try:
            parsed.append(_localized_timestamp(value, timezone).tz_convert(UTC))
        except (TypeError, ValueError):
            continue
    if len(parsed) < 2:
        return "1h"
    minutes = (
        pd.Series(parsed).sort_values().diff().dropna().dt.total_seconds() / 60
    )
    positive = minutes[minutes > 0]
    if positive.empty:
        return "1h"
    observed = float(positive.median())
    closest = min(INTERVALS, key=lambda candidate: abs(candidate - observed))
    return INTERVALS[closest]


def fetch_closed_bars(
    symbol: str,
    interval: str,
    after_utc: pd.Timestamp,
    session=requests,
) -> list[dict]:
    """Return closed BiQuote bars newer than ``after_utc`` (newest-first API)."""
    response = session.get(
        API_URL.format(symbol=api_symbol(symbol)),
        params={
            "interval": interval,
            "limit": MAX_BARS,
            "from": after_utc.isoformat().replace("+00:00", "Z"),
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    bars = payload.get("bars")
    if not isinstance(bars, list):
        raise RuntimeError("BiQuote returned an invalid OHLC response.")
    return sorted(
        (bar for bar in bars if not bar.get("isOpen", False)),
        key=lambda bar: bar["openTime"],
    )


def download_h1_history(symbol: str, session=requests) -> pd.DataFrame:
    """Download BiQuote's recent closed H1 candles as a model-ready frame."""
    response = session.get(
        API_URL.format(symbol=api_symbol(symbol)),
        params={"interval": "1h", "limit": MAX_BARS},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    bars = payload.get("bars")
    if not isinstance(bars, list):
        raise RuntimeError("BiQuote returned an invalid OHLC response.")

    rows = []
    for bar in bars:
        if bar.get("isOpen", False):
            continue
        try:
            timestamp = pd.Timestamp(bar["openTime"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(UTC)
            else:
                timestamp = timestamp.tz_convert(UTC)
            rows.append(
                {
                    "Date": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "Open": bar["open"],
                    "High": bar["high"],
                    "Low": bar["low"],
                    "Close": bar["close"],
                    "Volume": bar.get("tickVolume", bar.get("volume", 0)),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("BiQuote returned an invalid OHLC bar.") from exc
    if not rows:
        raise ValueError(f"BiQuote returned no closed H1 candles for {api_symbol(symbol)}.")

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset="Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )


def merge_latest_prices(path: Path, symbol: str, session=requests) -> pd.DataFrame:
    """Treat CSV times as UTC and append newer completed BiQuote candles."""
    frame = pd.read_csv(path, dtype="string", sep=None, engine="python")
    date_column = _date_column(frame, path)

    # Repair unquoted decimal-comma volumes (for example ``2,500``) which are
    # parsed as Volume=2 plus a trailing unnamed column=500.
    unnamed = [column for column in frame if str(column).startswith("Unnamed:")]
    if "Volume" in frame and len(unnamed) == 1:
        fractional = frame[unnamed[0]].str.strip()
        valid = frame["Volume"].str.strip().str.fullmatch(
            r"[+-]?\d+"
        ) & fractional.str.fullmatch(r"\d+")
        if valid.fillna(False).all():
            frame["Volume"] = frame["Volume"].str.strip() + "." + fractional
            frame = frame.drop(columns=unnamed)

    dated_rows = []
    for index, value in frame[date_column].items():
        try:
            dated_rows.append((index, _localized_timestamp(value, UTC)))
        except (TypeError, ValueError):
            continue
    if not dated_rows:
        raise ValueError(f"No valid timestamps were found in {path}.")

    # Normalize the existing history too. Naive CSV values are interpreted as
    # UTC; explicitly offset values are converted to their equivalent UTC time.
    for index, timestamp in dated_rows:
        frame.at[index, date_column] = timestamp.tz_convert(UTC).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    last_utc = max(timestamp.tz_convert(UTC) for _, timestamp in dated_rows)
    try:
        bars = fetch_closed_bars(
            symbol,
            _interval(frame, date_column, UTC),
            last_utc,
            session,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not refresh {api_symbol(symbol)} prices from BiQuote: {exc}"
        ) from exc
    additions = []
    for bar in bars:
        timestamp = pd.Timestamp(bar["openTime"]).tz_convert(UTC)
        if timestamp <= last_utc:
            continue
        row = {column: pd.NA for column in frame.columns}
        row[date_column] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        for column, api_key in (
            ("Open", "open"),
            ("High", "high"),
            ("Low", "low"),
            ("Close", "close"),
            ("Volume", "tickVolume"),
        ):
            if column in frame.columns:
                row[column] = bar.get(api_key, 0)
        additions.append(row)

    if additions:
        frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
    return frame


@contextmanager
def refreshed_price_csv(path: Path, symbol: str):
    """Yield a temporary UTC CSV completed with the latest closed candles."""
    frame = merge_latest_prices(path, symbol)
    with tempfile.TemporaryDirectory(prefix="market-refreshed-") as directory:
        refreshed = Path(directory) / path.name
        frame.to_csv(refreshed, index=False)
        yield refreshed


@contextmanager
def utc_price_csv(path: Path):
    """Yield a temporary copy whose price timestamps are normalized to UTC."""
    frame = pd.read_csv(path, dtype="string", sep=None, engine="python")
    date_column = _date_column(frame, path)
    valid_timestamps = 0
    for index, value in frame[date_column].items():
        try:
            timestamp = _localized_timestamp(value, UTC).tz_convert(UTC)
        except (TypeError, ValueError):
            continue
        frame.at[index, date_column] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        valid_timestamps += 1
    if not valid_timestamps:
        raise ValueError(f"No valid timestamps were found in {path}.")

    with tempfile.TemporaryDirectory(prefix="market-utc-") as directory:
        normalized = Path(directory) / path.name
        frame.to_csv(normalized, index=False)
        yield normalized
