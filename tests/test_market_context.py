import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "market_context.py"
SPEC = importlib.util.spec_from_file_location("market_context", MODULE_PATH)
market_context = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = market_context
SPEC.loader.exec_module(market_context)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.headers = {}
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        for suffix, payload in self.responses.items():
            if suffix in url:
                if isinstance(payload, Exception):
                    raise payload
                return FakeResponse(payload)
        raise AssertionError(f"Unexpected URL: {url}")


class MarketContextTests(unittest.TestCase):
    def test_coingecko_breadth_summarizes_participation_and_concentration(self):
        session = FakeSession(
            {
                "/coins/markets": [
                    {
                        "market_cap": 70,
                        "price_change_percentage_24h": 10,
                        "price_change_percentage_7d_in_currency": 5,
                        "price_change_percentage_30d_in_currency": 2,
                    },
                    {
                        "market_cap": 30,
                        "price_change_percentage_24h": -10,
                        "price_change_percentage_7d_in_currency": 1,
                        "price_change_percentage_30d_in_currency": -2,
                    },
                ]
            }
        )

        result = market_context.fetch_coingecko_breadth(session, timeout=1)
        values = {item.feature: item.value for item in result}

        self.assertEqual(values["crypto_advance_share_24h"], 0.5)
        self.assertEqual(values["crypto_median_return_24h"], 0.0)
        self.assertAlmostEqual(values["crypto_cap_weighted_return_24h"], 4.0)
        self.assertEqual(values["crypto_top10_market_cap_share"], 1.0)

    def test_fred_uses_initial_release_date_instead_of_revised_history(self):
        session = FakeSession(
            {
                "/series/observations": {
                    "observations": [
                        {
                            "date": "2026-01-01",
                            "realtime_start": "2026-01-03",
                            "value": "4.25",
                        }
                    ]
                }
            }
        )

        result = market_context.fetch_fred(
            {"fed_funds_rate": "DFF"},
            session,
            api_key="test",
            timeout=1,
        )

        self.assertEqual(
            result[0].available_at.isoformat(), "2026-01-03T00:00:00+00:00"
        )
        self.assertEqual(session.calls[0][1]["params"]["output_type"], 4)

    def test_binance_derivatives_are_normalized_to_long_schema(self):
        session = FakeSession(
            {
                "fundingRate": [
                    {"fundingTime": 1_700_000_000_000, "fundingRate": "0.0001"}
                ],
                "openInterestHist": [
                    {"timestamp": 1_700_000_000_000, "sumOpenInterestValue": "10"}
                ],
                "globalLongShortAccountRatio": [
                    {"timestamp": 1_700_000_000_000, "longShortRatio": "1.2"}
                ],
                "topLongShortPositionRatio": [
                    {"timestamp": 1_700_000_000_000, "longShortRatio": "1.1"}
                ],
                "takerlongshortRatio": [
                    {"timestamp": 1_700_000_000_000, "buySellRatio": "0.9"}
                ],
                "premiumIndexKlines": [
                    [1_700_000_000_000, "0", "0", "0", "0.002", "0", 1_700_003_600_000]
                ],
            }
        )

        result = market_context.fetch_binance_derivatives(
            "BTCUSDT", session, timeout=1
        )

        self.assertEqual(len(result), 6)
        self.assertEqual(
            {item.feature for item in result},
            {
                "funding_rate",
                "open_interest_usd",
                "global_long_short_ratio",
                "top_trader_long_short_ratio",
                "taker_buy_sell_ratio",
                "perpetual_basis",
            },
        )

    def test_collection_keeps_partial_results_and_merges_existing_rows(self):
        config = {
            "assets": {"BTC": {"coingecko": "bitcoin"}},
            "fred_series": {},
            "glassnode_metrics": {},
        }
        session = FakeSession(
            {
                "/market_chart": {
                    "prices": [[1_700_000_000_000, 100]],
                    "market_caps": [[1_700_000_000_000, 1000]],
                    "total_volumes": [[1_700_000_000_000, 50]],
                },
                "/global": RuntimeError("global endpoint unavailable"),
                "/coins/markets": [],
                "/coins/bitcoin": {"market_data": {}},
                "/search/trending": {"coins": []},
                "stablecoincharts/all": [],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config_path = directory / "context.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = market_context.collect_asset_context(
                "btc",
                config_path=config_path,
                output_dir=directory / "output",
                timeout=1,
                session=session,
            )
            second = market_context.collect_asset_context(
                "btc",
                config_path=config_path,
                output_dir=directory / "output",
                timeout=1,
                session=session,
            )

        self.assertEqual(result["fetched"], 4)
        self.assertEqual(second["saved"], 4)
        self.assertIn("coingecko_global", result["errors"])


if __name__ == "__main__":
    unittest.main()
