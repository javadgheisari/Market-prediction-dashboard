"""Download and convert ForexSB's public historical data feed."""

from __future__ import annotations

import gzip
import json
import struct
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


INFO_URL = "https://data.forexsb.com/datafeed/info/premium.json.gz"
DUKASCOPY_URL = "https://data.forexsb.com/datafeed/data/dukascopy/{symbol}30.lb.gz"
BINANCE_URL = "https://data.forexsb.com/datafeed/data/binance/{symbol}30.lb.gz"
REQUEST_TIMEOUT_SECONDS = 30
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_H1_BARS = 10_000
MILLENNIUM_UTC = pd.Timestamp("2000-01-01", tz="UTC")

# Symbols from the application's published feed metadata. Assets outside this
# set still work through CSV upload.
SUPPORTED_CRYPTO = {
    "ADA", "ATOM", "AVAX", "BNB", "BTC", "DOGE", "DOT", "ETC", "ETH",
    "FIL", "ICP", "LINK", "LTC", "SOL", "TRX", "UNI", "XLM", "XRP",
}


def supports_symbol(symbol: str) -> bool:
    symbol = symbol.upper()
    return symbol in SUPPORTED_CRYPTO or symbol in {"EUR", "GBP", "XAU", "XAG"}


def _decode_gzip(payload: bytes) -> bytes:
    return gzip.decompress(payload) if payload.startswith(b"\x1f\x8b") else payload


def _get(session, url: str) -> bytes:
    response = session.get(
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": "MarketPredictionDashboard/1.0"},
    )
    response.raise_for_status()
    payload = response.content
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError("ForexSB response exceeded the safe download limit.")
    decoded = _decode_gzip(payload)
    if len(decoded) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError("ForexSB response exceeded the safe decoded-data limit.")
    return decoded


def _feed_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol in SUPPORTED_CRYPTO:
        return f"{symbol}USDT"
    return symbol if symbol.endswith("USD") else f"{symbol}USD"


def _record_size(payload: bytes) -> int:
    if len(payload) % 28 == 0:
        previous = -1
        valid = True
        for offset in range(0, len(payload), 28):
            timestamp = struct.unpack_from("<i", payload, offset)[0]
            if timestamp < previous:
                valid = False
                break
            previous = timestamp
        if valid:
            return 28
    if len(payload) % 24 == 0:
        return 24
    raise ValueError("ForexSB returned an unsupported bar-data format.")


def download_h1_history(symbol: str, session=requests) -> pd.DataFrame:
    """Download ForexSB M30 bars and aggregate them into closed UTC H1 bars."""
    feed_symbol = _feed_symbol(symbol)
    try:
        metadata = json.loads(_get(session, INFO_URL).decode("utf-8"))
        info = metadata[feed_symbol]
        endpoint = BINANCE_URL if feed_symbol.endswith("USDT") else DUKASCOPY_URL
        payload = _get(session, endpoint.format(symbol=feed_symbol))
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download {feed_symbol} history from ForexSB: {exc}") from exc
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"ForexSB has no historical feed for {feed_symbol}.") from exc

    size = _record_size(payload)
    values = []
    format_string = "<7i" if size == 28 else "<6i"
    price_scale = float(info["priceScale"])
    volume_scale = float(info.get("volumeScale", 1) or 1)
    for offset in range(0, len(payload), size):
        record = struct.unpack_from(format_string, payload, offset)
        values.append(
            {
                "Date": MILLENNIUM_UTC + pd.to_timedelta(record[0], unit="m"),
                "Open": record[1] / price_scale,
                "High": record[2] / price_scale,
                "Low": record[3] / price_scale,
                "Close": record[4] / price_scale,
                "Volume": record[5] / volume_scale,
            }
        )
    if not values:
        raise ValueError(f"ForexSB returned no historical bars for {feed_symbol}.")

    frame = pd.DataFrame(values).sort_values("Date")
    frame["Hour"] = frame["Date"].dt.floor("h")
    hourly = frame.groupby("Hour", as_index=False).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    ).rename(columns={"Hour": "Date"})
    current_hour = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
    hourly = hourly.loc[hourly["Date"] < current_hour].copy()
    # Full crypto feeds can contain nearly a decade of hourly candles. Keep a
    # substantial recent window so interactive model training remains practical.
    hourly = hourly.tail(MAX_H1_BARS).reset_index(drop=True)
    hourly["Date"] = hourly["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return hourly


@contextmanager
def forexsb_h1_csv(symbol: str):
    """Yield downloaded H1 history as a temporary model-ready CSV."""
    frame = download_h1_history(symbol)
    with tempfile.TemporaryDirectory(prefix="market-forexsb-") as directory:
        path = Path(directory) / f"{symbol.upper()}_H1.csv"
        frame.to_csv(path, index=False)
        yield path
