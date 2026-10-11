"""Collect causal, timestamped market context for dashboard assets.

The output uses a provider-neutral long schema.  ``available_at`` is the time a
value could have been known, rather than merely the period it describes.  This
lets the forecasting pipeline perform backward-only as-of joins without using
future information.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "market_context.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "context"
COINGECKO_API = "https://api.coingecko.com/api/v3"
BINANCE_FUTURES_API = "https://fapi.binance.com"
BINANCE_SPOT_API = "https://api.binance.com"
DEFILLAMA_API = "https://api.llama.fi"
FRED_API = "https://api.stlouisfed.org/fred"
EIA_API = "https://api.eia.gov/v2"
GLASSNODE_API = "https://api.glassnode.com"
CFTC_API = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"
SEC_DATA_API = "https://data.sec.gov"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CSV_FIELDS = ("available_at", "feature", "value", "source", "max_age_hours")
SAFE_FEATURE = re.compile(r"^[a-z][a-z0-9_]{1,79}$")


@dataclass(frozen=True)
class Observation:
    available_at: datetime
    feature: str
    value: float
    source: str
    max_age_hours: float

    def row(self) -> dict[str, str]:
        return {
            "available_at": self.available_at.astimezone(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "feature": self.feature,
            "value": f"{self.value:.12g}",
            "source": self.source,
            "max_age_hours": f"{self.max_age_hours:.4g}",
        }


def _utc_from_milliseconds(value: int | float | str) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)


def _utc_from_seconds(value: int | float | str) -> datetime:
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _request_json(session, url: str, *, timeout: float, **kwargs):
    response = session.get(url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config.get("assets"), dict):
        raise ValueError("Market context config must contain an assets object.")
    return config


def fetch_coingecko(
    coin_id: str, session, *, timeout: float, days: str = "max"
) -> list[Observation]:
    payload = _request_json(
        session,
        f"{COINGECKO_API}/coins/{coin_id}/market_chart",
        timeout=timeout,
        params={"vs_currency": "usd", "days": days},
    )
    observations: list[Observation] = []
    series = {
        "coingecko_price_usd": payload.get("prices", []),
        "coingecko_market_cap_usd": payload.get("market_caps", []),
        "coingecko_volume_usd": payload.get("total_volumes", []),
    }
    for feature, values in series.items():
        for timestamp, raw_value in values:
            value = _number(raw_value)
            if value is not None:
                observations.append(
                    Observation(
                        _utc_from_milliseconds(timestamp), feature, value,
                        "coingecko", 48.0,
                    )
                )
    market_caps = {timestamp: _number(value) for timestamp, value in payload.get("market_caps", [])}
    volumes = {timestamp: _number(value) for timestamp, value in payload.get("total_volumes", [])}
    for timestamp in market_caps.keys() & volumes.keys():
        market_cap = market_caps[timestamp]
        volume = volumes[timestamp]
        if market_cap is not None and market_cap > 0 and volume is not None:
            observations.append(
                Observation(
                    _utc_from_milliseconds(timestamp),
                    "coingecko_volume_to_market_cap",
                    volume / market_cap,
                    "coingecko",
                    48.0,
                )
            )
    return observations


def fetch_coingecko_global(session, *, timeout: float) -> list[Observation]:
    payload = _request_json(
        session, f"{COINGECKO_API}/global", timeout=timeout
    ).get("data", {})
    timestamp = datetime.now(timezone.utc)
    candidates = {
        "crypto_active_count": payload.get("active_cryptocurrencies"),
        "crypto_total_market_cap_usd": payload.get("total_market_cap", {}).get("usd"),
        "crypto_total_volume_usd": payload.get("total_volume", {}).get("usd"),
        "crypto_btc_dominance": payload.get("market_cap_percentage", {}).get("btc"),
        "crypto_eth_dominance": payload.get("market_cap_percentage", {}).get("eth"),
        "crypto_market_cap_change_24h": payload.get(
            "market_cap_change_percentage_24h_usd"
        ),
    }
    total_market_cap = _number(candidates["crypto_total_market_cap_usd"])
    total_volume = _number(candidates["crypto_total_volume_usd"])
    if total_market_cap is not None and total_market_cap > 0 and total_volume is not None:
        candidates["crypto_volume_to_market_cap"] = total_volume / total_market_cap
    return [
        Observation(timestamp, feature, value, "coingecko", 12.0)
        for feature, raw_value in candidates.items()
        if (value := _number(raw_value)) is not None
    ]


def fetch_coingecko_breadth(session, *, timeout: float) -> list[Observation]:
    """Summarize participation and concentration across large crypto assets."""
    payload = _request_json(
        session,
        f"{COINGECKO_API}/coins/markets",
        timeout=timeout,
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h,7d,30d",
        },
    )
    assets = [item for item in payload if isinstance(item, dict)]
    market_caps = [_number(item.get("market_cap")) for item in assets]
    total_cap = sum(value for value in market_caps if value is not None and value > 0)
    candidates: dict[str, float] = {}
    if total_cap > 0:
        top_ten_cap = sum(
            value for value in market_caps[:10] if value is not None and value > 0
        )
        candidates["crypto_top10_market_cap_share"] = top_ten_cap / total_cap

    periods = {
        "24h": "price_change_percentage_24h",
        "7d": "price_change_percentage_7d_in_currency",
        "30d": "price_change_percentage_30d_in_currency",
    }
    for label, field in periods.items():
        changes = [
            value
            for item in assets
            if (value := _number(item.get(field))) is not None
        ]
        if not changes:
            continue
        ordered = sorted(changes)
        midpoint = len(ordered) // 2
        median = (
            ordered[midpoint]
            if len(ordered) % 2
            else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
        )
        candidates[f"crypto_advance_share_{label}"] = (
            sum(value > 0 for value in changes) / len(changes)
        )
        candidates[f"crypto_median_return_{label}"] = median
        weighted = [
            (change, market_cap)
            for item in assets
            if (change := _number(item.get(field))) is not None
            and (market_cap := _number(item.get("market_cap"))) is not None
            and market_cap > 0
        ]
        weighted_cap = sum(market_cap for _, market_cap in weighted)
        if weighted_cap > 0:
            candidates[f"crypto_cap_weighted_return_{label}"] = sum(
                change * market_cap for change, market_cap in weighted
            ) / weighted_cap

    timestamp = datetime.now(timezone.utc)
    return [
        Observation(timestamp, feature, value, "coingecko", 12.0)
        for feature, value in candidates.items()
    ]


def fetch_coingecko_snapshot(
    coin_id: str, session, *, timeout: float
) -> list[Observation]:
    payload = _request_json(
        session,
        f"{COINGECKO_API}/coins/{coin_id}",
        timeout=timeout,
        params={
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        },
    )
    market = payload.get("market_data", {})
    community = payload.get("community_data", {})
    developer = payload.get("developer_data", {})
    candidates = {
        "coingecko_market_cap_rank": payload.get("market_cap_rank"),
        "coingecko_circulating_supply": market.get("circulating_supply"),
        "coingecko_total_supply": market.get("total_supply"),
        "coingecko_max_supply": market.get("max_supply"),
        "coingecko_fully_diluted_value_usd": (
            market.get("fully_diluted_valuation") or {}
        ).get("usd"),
        "coingecko_reddit_subscribers": community.get("reddit_subscribers"),
        "coingecko_github_stars": developer.get("stars"),
        "coingecko_github_forks": developer.get("forks"),
        "coingecko_github_subscribers": developer.get("subscribers"),
        "coingecko_commits_4w": developer.get("commit_count_4_weeks"),
    }
    current_market_cap = _number((market.get("market_cap") or {}).get("usd"))
    current_volume = _number((market.get("total_volume") or {}).get("usd"))
    fully_diluted = _number(
        (market.get("fully_diluted_valuation") or {}).get("usd")
    )
    circulating_supply = _number(market.get("circulating_supply"))
    total_supply = _number(market.get("total_supply"))
    if current_market_cap is not None and current_market_cap > 0:
        if current_volume is not None:
            candidates["coingecko_volume_to_market_cap"] = (
                current_volume / current_market_cap
            )
        if fully_diluted is not None:
            candidates["coingecko_fdv_to_market_cap"] = (
                fully_diluted / current_market_cap
            )
    if total_supply is not None and total_supply > 0 and circulating_supply is not None:
        candidates["coingecko_circulating_supply_ratio"] = (
            circulating_supply / total_supply
        )
    timestamp = datetime.now(timezone.utc)
    return [
        Observation(timestamp, feature, value, "coingecko", 168.0)
        for feature, raw_value in candidates.items()
        if (value := _number(raw_value)) is not None
    ]


def fetch_coingecko_trending(
    coin_id: str, session, *, timeout: float
) -> list[Observation]:
    payload = _request_json(
        session, f"{COINGECKO_API}/search/trending", timeout=timeout
    )
    for index, entry in enumerate(payload.get("coins", []), start=1):
        item = entry.get("item", {})
        if item.get("id") == coin_id:
            score = _number(item.get("score"))
            value = score if score is not None else float(index)
            return [
                Observation(
                    datetime.now(timezone.utc), "coingecko_trending_rank",
                    value, "coingecko", 12.0,
                )
            ]
    return []


def fetch_bitcoin_benchmark(session, *, timeout: float) -> list[Observation]:
    observations = fetch_coingecko("bitcoin", session, timeout=timeout)
    return [
        Observation(
            item.available_at,
            item.feature.replace("coingecko_", "benchmark_btc_"),
            item.value,
            item.source,
            item.max_age_hours,
        )
        for item in observations
    ]


def _binance_series(
    payload: Iterable[dict[str, Any]], feature: str, value_key: str,
    timestamp_key: str,
) -> list[Observation]:
    observations = []
    for item in payload:
        value = _number(item.get(value_key))
        timestamp = item.get(timestamp_key)
        if value is not None and timestamp is not None:
            observations.append(
                Observation(
                    _utc_from_milliseconds(timestamp), feature, value,
                    "binance_futures", 12.0,
                )
            )
    return observations


def fetch_binance_derivatives(
    symbol: str, session, *, timeout: float
) -> list[Observation]:
    requests_to_make = (
        (
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": 1000},
            "funding_rate",
            "fundingRate",
            "fundingTime",
        ),
        (
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": "1h", "limit": 500},
            "open_interest_usd",
            "sumOpenInterestValue",
            "timestamp",
        ),
        (
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": "1h", "limit": 500},
            "global_long_short_ratio",
            "longShortRatio",
            "timestamp",
        ),
        (
            "/futures/data/topLongShortPositionRatio",
            {"symbol": symbol, "period": "1h", "limit": 500},
            "top_trader_long_short_ratio",
            "longShortRatio",
            "timestamp",
        ),
        (
            "/futures/data/takerlongshortRatio",
            {"symbol": symbol, "period": "1h", "limit": 500},
            "taker_buy_sell_ratio",
            "buySellRatio",
            "timestamp",
        ),
    )
    observations: list[Observation] = []
    for path, params, feature, value_key, timestamp_key in requests_to_make:
        payload = _request_json(
            session,
            f"{BINANCE_FUTURES_API}{path}",
            timeout=timeout,
            params=params,
        )
        observations.extend(
            _binance_series(payload, feature, value_key, timestamp_key)
        )
    premium_klines = _request_json(
        session,
        f"{BINANCE_FUTURES_API}/fapi/v1/premiumIndexKlines",
        timeout=timeout,
        params={"symbol": symbol, "interval": "1h", "limit": 1000},
    )
    for row in premium_klines:
        if len(row) > 6 and (value := _number(row[4])) is not None:
            observations.append(
                Observation(
                    _utc_from_milliseconds(row[6]), "perpetual_basis", value,
                    "binance_futures", 12.0,
                )
            )
    return observations


def fetch_binance_spot(
    symbol: str, session, *, timeout: float
) -> list[Observation]:
    """Fetch historical trade-flow bars and a current order-book snapshot."""
    observations = []
    klines = _request_json(
        session,
        f"{BINANCE_SPOT_API}/api/v3/klines",
        timeout=timeout,
        params={"symbol": symbol, "interval": "1h", "limit": 1000},
    )
    for row in klines:
        if len(row) < 11:
            continue
        quote_volume = _number(row[7])
        trade_count = _number(row[8])
        taker_buy_quote = _number(row[10])
        if row[6] is None:
            continue
        available_at = _utc_from_milliseconds(row[6])
        values = {
            "spot_quote_volume": quote_volume,
            "spot_trade_count": trade_count,
            "spot_taker_buy_share": (
                taker_buy_quote / quote_volume
                if taker_buy_quote is not None
                and quote_volume is not None
                and quote_volume > 0
                else None
            ),
        }
        for feature, value in values.items():
            if value is not None:
                observations.append(
                    Observation(
                        available_at, feature, value, "binance_spot", 6.0
                    )
                )

    depth = _request_json(
        session,
        f"{BINANCE_SPOT_API}/api/v3/depth",
        timeout=timeout,
        params={"symbol": symbol, "limit": 1000},
    )
    bids = [(_number(price), _number(quantity)) for price, quantity in depth.get("bids", [])]
    asks = [(_number(price), _number(quantity)) for price, quantity in depth.get("asks", [])]
    bids = [(price, quantity) for price, quantity in bids if price and quantity]
    asks = [(price, quantity) for price, quantity in asks if price and quantity]
    if bids and asks:
        timestamp = datetime.now(timezone.utc)
        best_bid, best_ask = bids[0][0], asks[0][0]
        midpoint = (best_bid + best_ask) / 2.0
        current = {
            "spot_spread_bps": (best_ask - best_bid) / midpoint * 10_000,
        }
        for distance in (0.001, 0.005, 0.01):
            suffix = str(int(distance * 1000))
            bid_depth = sum(
                price * quantity
                for price, quantity in bids
                if price >= midpoint * (1 - distance)
            )
            ask_depth = sum(
                price * quantity
                for price, quantity in asks
                if price <= midpoint * (1 + distance)
            )
            total_depth = bid_depth + ask_depth
            current[f"spot_depth_{suffix}permille_usd"] = total_depth
            current[f"spot_depth_imbalance_{suffix}permille"] = (
                (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0
            )
        observations.extend(
            Observation(timestamp, feature, value, "binance_spot", 2.0)
            for feature, value in current.items()
        )
    return observations


def fetch_defillama(
    protocol: str, session, *, timeout: float
) -> list[Observation]:
    payload = _request_json(
        session, f"{DEFILLAMA_API}/protocol/{protocol}", timeout=timeout
    )
    observations = []
    for item in payload.get("tvl", []):
        value = _number(item.get("totalLiquidityUSD"))
        timestamp = item.get("date")
        if value is not None and timestamp is not None:
            # Daily protocol aggregates are conservatively made available at
            # the beginning of the following UTC day.
            available_at = _utc_from_seconds(timestamp) + timedelta(days=1)
            observations.append(
                Observation(
                    available_at, "defi_tvl_usd", value, "defillama", 96.0
                )
            )
    return observations


def fetch_defillama_chain(
    chain: str, session, *, timeout: float
) -> list[Observation]:
    payload = _request_json(
        session,
        f"{DEFILLAMA_API}/v2/historicalChainTvl/{chain}",
        timeout=timeout,
    )
    observations = []
    for item in payload:
        value = _number(item.get("tvl"))
        timestamp = item.get("date")
        if value is not None and timestamp is not None:
            observations.append(
                Observation(
                    _utc_from_seconds(timestamp) + timedelta(days=1),
                    "defi_chain_tvl_usd",
                    value,
                    "defillama",
                    96.0,
                )
            )
    return observations


def fetch_stablecoin_supply(session, *, timeout: float) -> list[Observation]:
    payload = _request_json(
        session,
        "https://stablecoins.llama.fi/stablecoincharts/all",
        timeout=timeout,
    )
    observations = []
    for item in payload:
        total = item.get("totalCirculatingUSD") or item.get("totalCirculating") or {}
        value = _number(total.get("peggedUSD"))
        timestamp = item.get("date")
        if value is not None and timestamp is not None:
            observations.append(
                Observation(
                    _utc_from_seconds(timestamp) + timedelta(days=1),
                    "stablecoin_supply_usd",
                    value,
                    "defillama",
                    96.0,
                )
            )
    return observations


def fetch_fred(
    series: dict[str, str], session, *, api_key: str, timeout: float
) -> list[Observation]:
    observations = []
    for feature, series_id in series.items():
        if not SAFE_FEATURE.fullmatch(feature):
            continue
        payload = _request_json(
            session,
            f"{FRED_API}/series/observations",
            timeout=timeout,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "realtime_start": "1776-07-04",
                "realtime_end": "9999-12-31",
                "output_type": 4,
                "sort_order": "asc",
            },
        )
        for item in payload.get("observations", []):
            value = _number(item.get("value"))
            try:
                release_date = datetime.strptime(
                    item["realtime_start"], "%Y-%m-%d"
                )
            except (KeyError, TypeError, ValueError):
                continue
            if value is not None:
                # output_type=4 returns the initial release rather than today's
                # revised historical value. The ALFRED real-time start is the
                # first date that value was available to a forecaster.
                available_at = release_date.replace(tzinfo=timezone.utc)
                observations.append(
                    Observation(
                        available_at, f"macro_{feature}", value, "fred", 240.0
                    )
                )
    return observations


def fetch_glassnode(
    asset: str,
    metrics: dict[str, str],
    session,
    *,
    api_key: str,
    timeout: float,
    days: int = 365,
) -> list[Observation]:
    observations = []
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    for feature, path in metrics.items():
        if not SAFE_FEATURE.fullmatch(feature) or not str(path).startswith("/v1/"):
            continue
        payload = _request_json(
            session,
            f"{GLASSNODE_API}{path}",
            timeout=timeout,
            params={"a": asset, "s": since, "i": "1h", "api_key": api_key},
        )
        for item in payload:
            value = _number(item.get("v"))
            timestamp = item.get("t")
            if value is not None and timestamp is not None:
                observations.append(
                    Observation(
                        _utc_from_seconds(timestamp), f"onchain_{feature}", value,
                        "glassnode", 72.0,
                    )
                )
    return observations


def fetch_eia(
    series: dict[str, str], session, *, api_key: str, timeout: float
) -> list[Observation]:
    """Fetch weekly petroleum fundamentals through EIA's v2 series adapter."""
    observations = []
    for feature, series_id in series.items():
        if not SAFE_FEATURE.fullmatch(feature):
            continue
        payload = _request_json(
            session,
            f"{EIA_API}/seriesid/{series_id}",
            timeout=timeout,
            params={"api_key": api_key},
        )
        for item in payload.get("response", {}).get("data", []):
            value = _number(item.get("value"))
            try:
                period = datetime.strptime(item["period"], "%Y-%m-%d")
            except (KeyError, TypeError, ValueError):
                continue
            if value is not None:
                # Weekly petroleum periods end Friday and are normally released
                # the following week. Six days is a conservative availability
                # delay that avoids leaking the unpublished inventory value.
                available_at = period.replace(tzinfo=timezone.utc) + timedelta(days=6)
                observations.append(
                    Observation(
                        available_at, f"energy_{feature}", value, "eia", 336.0
                    )
                )
    return observations


def fetch_cftc(
    market_name: str, session, *, timeout: float
) -> list[Observation]:
    """Fetch weekly legacy COT positioning for a configured futures market."""
    fields = (
        "market_and_exchange_names,report_date_as_yyyy_mm_dd,open_interest_all,"
        "noncomm_positions_long_all,noncomm_positions_short_all,"
        "comm_positions_long_all,comm_positions_short_all"
    )
    safe_market_name = market_name.replace("'", "''")
    payload = _request_json(
        session,
        CFTC_API,
        timeout=timeout,
        params={
            "$select": fields,
            "$where": (
                "upper(market_and_exchange_names) like "
                f"'%{safe_market_name.upper()}%'"
            ),
            "$order": "report_date_as_yyyy_mm_dd asc",
            "$limit": 5000,
        },
    )
    observations = []
    for item in payload:
        try:
            report_date = datetime.fromisoformat(
                str(item["report_date_as_yyyy_mm_dd"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError):
            continue
        if report_date.tzinfo is None:
            report_date = report_date.replace(tzinfo=timezone.utc)
        available_at = report_date + timedelta(days=4)
        noncommercial_long = _number(item.get("noncomm_positions_long_all"))
        noncommercial_short = _number(item.get("noncomm_positions_short_all"))
        commercial_long = _number(item.get("comm_positions_long_all"))
        commercial_short = _number(item.get("comm_positions_short_all"))
        values = {
            "cot_open_interest": _number(item.get("open_interest_all")),
            "cot_noncommercial_net": noncommercial_long - noncommercial_short
            if noncommercial_long is not None and noncommercial_short is not None
            else None,
            "cot_commercial_net": commercial_long - commercial_short
            if commercial_long is not None and commercial_short is not None
            else None,
        }
        for feature, value in values.items():
            if value is not None:
                observations.append(
                    Observation(available_at, feature, value, "cftc", 336.0)
                )
    return observations


def fetch_sec_events(symbol: str, session, *, timeout: float) -> list[Observation]:
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise ValueError(
            "Set SEC_USER_AGENT to an application name and contact email before "
            "collecting SEC data."
        )
    headers = {"User-Agent": user_agent}
    tickers = _request_json(
        session, SEC_TICKERS_URL, timeout=timeout, headers=headers
    )
    company = next(
        (
            item for item in tickers.values()
            if str(item.get("ticker", "")).upper() == symbol.upper()
        ),
        None,
    )
    if company is None:
        raise ValueError(f"SEC has no ticker mapping for {symbol}.")
    cik = str(company["cik_str"]).zfill(10)
    payload = _request_json(
        session,
        f"{SEC_DATA_API}/submissions/CIK{cik}.json",
        timeout=timeout,
        headers=headers,
    )
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accepted = recent.get("acceptanceDateTime", [])
    filing_dates = recent.get("filingDate", [])
    tracked = {"10-K", "10-Q", "8-K", "3", "4", "5", "144"}
    observations = []
    for index, form in enumerate(forms):
        if form not in tracked:
            continue
        raw_timestamp = accepted[index] if index < len(accepted) else ""
        try:
            available_at = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            if index >= len(filing_dates):
                continue
            try:
                available_at = datetime.strptime(
                    filing_dates[index], "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc) + timedelta(days=1)
            except (TypeError, ValueError):
                continue
        if available_at.tzinfo is None:
            available_at = available_at.replace(tzinfo=timezone.utc)
        feature = f"sec_form_{form.lower().replace('-', '_')}"
        observations.append(
            Observation(available_at, feature, 1.0, "sec", 72.0)
        )

    company_facts = _request_json(
        session,
        f"{SEC_DATA_API}/api/xbrl/companyfacts/CIK{cik}.json",
        timeout=timeout,
        headers=headers,
    )
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    concepts = {
        "revenue": (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
        ),
        "net_income": ("NetIncomeLoss",),
        "operating_income": ("OperatingIncomeLoss",),
        "diluted_eps": ("EarningsPerShareDiluted",),
        "assets": ("Assets",),
        "liabilities": ("Liabilities",),
        "cash": ("CashAndCashEquivalentsAtCarryingValue",),
        "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    }
    for feature, candidate_concepts in concepts.items():
        fact = next(
            (us_gaap[name] for name in candidate_concepts if name in us_gaap),
            None,
        )
        if fact is None:
            continue
        facts_by_filing: dict[datetime, tuple[tuple[Any, ...], float]] = {}
        for unit_values in fact.get("units", {}).values():
            for item in unit_values:
                form = item.get("form")
                if form not in {"10-K", "10-Q"}:
                    continue
                value = _number(item.get("val"))
                try:
                    filed_at = datetime.strptime(
                        item["filed"], "%Y-%m-%d"
                    ).replace(tzinfo=timezone.utc) + timedelta(days=1)
                except (KeyError, TypeError, ValueError):
                    continue
                if value is not None:
                    try:
                        end = datetime.strptime(item.get("end", ""), "%Y-%m-%d")
                    except (TypeError, ValueError):
                        end = datetime.min
                    try:
                        start = datetime.strptime(
                            item.get("start", ""), "%Y-%m-%d"
                        )
                        duration_days = (end - start).days
                    except (TypeError, ValueError):
                        duration_days = 0
                    target_days = 365 if form == "10-K" else 91
                    # A filing often repeats comparison periods. Prefer the
                    # latest period end and, for duration facts, the period
                    # closest to the form's annual/quarterly reporting span.
                    rank = (
                        end,
                        -abs(duration_days - target_days) if duration_days else 0,
                    )
                    previous = facts_by_filing.get(filed_at)
                    if previous is None or rank > previous[0]:
                        facts_by_filing[filed_at] = (rank, value)
        observations.extend(
            Observation(
                filed_at,
                f"sec_fundamental_{feature}",
                ranked_value[1],
                "sec",
                24.0 * 400,
            )
            for filed_at, ranked_value in facts_by_filing.items()
        )
    return observations


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {field: row.get(field, "") for field in CSV_FIELDS}
            for row in csv.DictReader(handle)
        ]


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


def merge_rows(
    existing: list[dict[str, str]], incoming: Iterable[Observation]
) -> list[dict[str, str]]:
    by_key = {
        (row["available_at"], row["feature"], row["source"]): row
        for row in existing
        if row.get("available_at") and row.get("feature")
    }
    for observation in incoming:
        row = observation.row()
        by_key[(row["available_at"], row["feature"], row["source"])] = row
    return sorted(
        by_key.values(),
        key=lambda row: (row["available_at"], row["feature"], row["source"]),
    )


def collect_asset_context(
    symbol: str,
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout: float = 20.0,
    session=None,
) -> dict[str, Any]:
    """Collect every configured source, retaining useful partial results."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    normalized = symbol.strip().upper()
    config = load_config(config_path)
    definition = config["assets"].get(normalized, {})
    client = session or requests.Session()
    client.headers.setdefault("User-Agent", "MarketPredictionDashboard/1.0")
    observations: list[Observation] = []
    sources: dict[str, int] = {}
    errors: dict[str, str] = {}

    def collect(name: str, function, *args, **kwargs):
        try:
            values = function(*args, **kwargs)
        except (
            IndexError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            requests.RequestException,
        ) as exc:
            errors[name] = str(exc) or exc.__class__.__name__
            return
        observations.extend(values)
        sources[name] = len(values)

    if definition.get("coingecko"):
        collect(
            "coingecko", fetch_coingecko, definition["coingecko"], client,
            timeout=timeout,
        )
        collect("coingecko_global", fetch_coingecko_global, client, timeout=timeout)
        collect(
            "coingecko_breadth", fetch_coingecko_breadth, client, timeout=timeout
        )
        collect(
            "coingecko_snapshot", fetch_coingecko_snapshot,
            definition["coingecko"], client, timeout=timeout,
        )
        collect(
            "coingecko_trending", fetch_coingecko_trending,
            definition["coingecko"], client, timeout=timeout,
        )
        if definition["coingecko"] != "bitcoin":
            collect("coingecko_btc", fetch_bitcoin_benchmark, client, timeout=timeout)
    if definition.get("binance"):
        collect(
            "binance", fetch_binance_derivatives, definition["binance"], client,
            timeout=timeout,
        )
        collect(
            "binance_spot", fetch_binance_spot, definition["binance"], client,
            timeout=timeout,
        )
    if definition.get("defillama"):
        collect(
            "defillama", fetch_defillama, definition["defillama"], client,
            timeout=timeout,
        )
    if definition.get("defillama_chain"):
        collect(
            "defillama_chain", fetch_defillama_chain,
            definition["defillama_chain"], client, timeout=timeout,
        )
    if definition.get("coingecko"):
        collect(
            "defillama_stablecoins", fetch_stablecoin_supply, client,
            timeout=timeout,
        )
    if definition.get("sec"):
        collect("sec", fetch_sec_events, normalized, client, timeout=timeout)
    if definition.get("cftc"):
        collect(
            "cftc", fetch_cftc, definition["cftc"], client, timeout=timeout
        )

    fred_key = os.getenv("FRED_API_KEY")
    if fred_key:
        collect(
            "fred", fetch_fred, config.get("fred_series", {}), client,
            api_key=fred_key, timeout=timeout,
        )
    eia_key = os.getenv("EIA_API_KEY")
    if eia_key and definition.get("eia"):
        collect(
            "eia", fetch_eia, config.get("eia_series", {}), client,
            api_key=eia_key, timeout=timeout,
        )
    glassnode_key = os.getenv("GLASSNODE_API_KEY")
    if glassnode_key and definition.get("glassnode"):
        collect(
            "glassnode", fetch_glassnode, definition["glassnode"],
            config.get("glassnode_metrics", {}), client,
            api_key=glassnode_key, timeout=timeout,
        )

    path = output_dir / f"{normalized}_context.csv"
    merged = merge_rows(read_existing(path), observations)
    if observations or path.exists():
        write_rows(path, merged)
    return {
        "symbol": normalized,
        "fetched": len(observations),
        "saved": len(merged),
        "sources": sources,
        "errors": errors,
        "path": path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    for symbol in args.assets:
        result = collect_asset_context(
            symbol,
            config_path=args.config,
            output_dir=args.output_dir,
            timeout=args.timeout,
        )
        print(json.dumps({**result, "path": str(result["path"])}, indent=2))


if __name__ == "__main__":
    main()
