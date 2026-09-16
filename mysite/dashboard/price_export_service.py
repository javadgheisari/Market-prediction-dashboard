"""Build downloadable, up-to-date price histories for dashboard assets."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

from dashboard import forexsb_service, market_data_service


@contextmanager
def latest_price_csv(symbol: str):
    """Yield an H1 CSV using ForexSB history plus the latest BiQuote bars."""
    symbol = symbol.strip().upper()
    if forexsb_service.supports_symbol(symbol):
        with forexsb_service.forexsb_h1_csv(symbol) as historical_csv:
            with market_data_service.refreshed_price_csv(
                historical_csv, symbol
            ) as refreshed_csv:
                yield refreshed_csv
        return

    # Some dashboard assets are absent from ForexSB. BiQuote still supplies a
    # recent H1 history, so they remain downloadable without a seed CSV.
    frame = market_data_service.download_h1_history(symbol)
    with tempfile.TemporaryDirectory(prefix="market-download-") as directory:
        path = Path(directory) / f"{symbol}_H1.csv"
        frame.to_csv(path, index=False)
        yield path
