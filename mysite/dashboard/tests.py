import csv
import gzip
import json
import struct
import tempfile
from unittest.mock import patch
from pathlib import Path

import pandas as pd
from django.test import SimpleTestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from dashboard import forexsb_service, market_data_service
from dashboard import price_export_service
from dashboard.data_paths import has_bundled_data, list_available_assets, resolve_data_csv
from dashboard.forms import NameForm
from dashboard.forecaster import build_plotly_chart, news_overlapping_price_csv
from dashboard.news_service import load_asset_news
from sequence_forecaster import calibrate_return_scales, load_market_data


class DataPathTests(SimpleTestCase):
    def test_bundled_assets_are_discovered(self):
        assets = {
            symbol
            for symbols in list_available_assets().values()
            for symbol in symbols
        }
        self.assertTrue({"BTC", "ETH", "EUR", "XAU"}.issubset(assets))

    def test_supported_crypto_assets_are_listed_without_bundled_csvs(self):
        crypto = list_available_assets()["Cryptocurrency"]
        self.assertTrue({"SOL", "XRP", "DOGE", "ADA", "LINK"}.issubset(crypto))
        self.assertFalse(has_bundled_data("SOL"))

    def test_fx_majors_and_commodities_are_listed(self):
        assets = list_available_assets()
        self.assertTrue(
            {"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD"}.issubset(
                assets["FX majors"]
            )
        )
        self.assertTrue(
            {"BRENT", "USOIL", "XAG", "XAU"}.issubset(assets["Commodities"])
        )

    def test_symbol_resolves_to_existing_csv(self):
        self.assertTrue(resolve_data_csv("BTC").is_file())

    def test_xau_uses_the_most_complete_bundled_history(self):
        self.assertEqual(resolve_data_csv("XAU").name, "XAUUSD_M15.csv")


class MarketDataServiceTests(SimpleTestCase):
    def test_provider_symbols_use_conventional_fx_pairs_and_oil_tickers(self):
        self.assertEqual(market_data_service.api_symbol("JPY"), "USDJPY")
        self.assertEqual(market_data_service.api_symbol("CAD"), "USDCAD")
        self.assertEqual(market_data_service.api_symbol("BRENT"), "UKOIL")

    def test_latest_prices_are_mapped_and_only_fresh_open_quotes_are_live(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "USDJPY": {
                        "mid": 142.125,
                        "stale": False,
                        "marketState": "open",
                        "timestamp": "2026-09-16T19:18:49Z",
                    },
                    "UKOIL": {
                        "mid": 95.25,
                        "stale": False,
                        "marketState": "closed",
                        "timestamp": "2026-09-16T17:00:00Z",
                    },
                }

        observed = {}
        session = type("Session", (), {})()

        def get(url, **kwargs):
            observed["url"] = url
            observed["params"] = kwargs["params"]
            return Response()

        session.get = get
        result = market_data_service.fetch_latest_prices(
            ["JPY", "BRENT"], session=session
        )

        self.assertEqual(observed["url"], market_data_service.LATEST_URL)
        self.assertEqual(
            observed["params"], [("symbols", "USDJPY"), ("symbols", "UKOIL")]
        )
        self.assertEqual(result["JPY"]["price"], 142.125)
        self.assertTrue(result["JPY"]["live"])
        self.assertFalse(result["BRENT"]["live"])

    def test_biquote_history_excludes_open_candle_and_sorts_rows(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "bars": [
                        {
                            "openTime": "2026-09-16T18:00:00Z",
                            "open": 3,
                            "high": 4,
                            "low": 2,
                            "close": 3.5,
                            "tickVolume": 30,
                            "isOpen": True,
                        },
                        {
                            "openTime": "2026-09-16T17:00:00Z",
                            "open": 2,
                            "high": 3,
                            "low": 1,
                            "close": 2.5,
                            "tickVolume": 20,
                            "isOpen": False,
                        },
                        {
                            "openTime": "2026-09-16T16:00:00Z",
                            "open": 1,
                            "high": 2,
                            "low": 0.5,
                            "close": 1.5,
                            "tickVolume": 10,
                            "isOpen": False,
                        },
                    ]
                }

        session = type("Session", (), {})()
        session.get = lambda *args, **kwargs: Response()
        result = market_data_service.download_h1_history("USOIL", session=session)

        self.assertEqual(
            result["Date"].tolist(),
            ["2026-09-16 16:00:00", "2026-09-16 17:00:00"],
        )
        self.assertEqual(result["Close"].tolist(), [1.5, 2.5])

    @patch("dashboard.market_data_service.fetch_closed_bars")
    def test_csv_and_api_candles_are_merged_in_utc(self, fetch_closed_bars):
        fetch_closed_bars.return_value = [
            {
                "openTime": "2026-09-12T07:00:00Z",
                "open": 11,
                "high": 13,
                "low": 10,
                "close": 12,
                "tickVolume": 250,
                "isOpen": False,
            }
        ]
        source = (
            "Date,Open,High,Low,Close,Volume\n"
            "2026-09-12 06:00:00,10,12,9,11,100\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BTCUSD.csv"
            path.write_text(source, encoding="utf-8")
            result = market_data_service.merge_latest_prices(path, "BTCUSD")

        self.assertEqual(
            result["Date"].tolist(),
            ["2026-09-12 06:00:00", "2026-09-12 07:00:00"],
        )
        self.assertEqual(
            fetch_closed_bars.call_args.args[2].isoformat(),
            "2026-09-12T06:00:00+00:00",
        )

    def test_forecast_csv_offsets_are_normalized_to_utc(self):
        source = "Date,Close\n2026-09-12T09:30:00+03:30,100\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text(source, encoding="utf-8")
            with market_data_service.utc_price_csv(path) as normalized:
                result = pd.read_csv(normalized, dtype="string")

        self.assertEqual(result.iloc[0]["Date"], "2026-09-12 06:00:00")

    @patch("dashboard.market_data_service.merge_latest_prices")
    def test_refreshed_training_csv_is_temporary(self, merge_latest_prices):
        merge_latest_prices.return_value = pd.DataFrame(
            {
                "Date": ["2026-09-12 06:00:00", "2026-09-12 07:00:00"],
                "Close": [100, 101],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "BTC.csv"
            original.write_text(
                "Date,Close\n2026-09-12 06:00:00,100\n", encoding="utf-8"
            )
            with market_data_service.refreshed_price_csv(original, "BTC") as path:
                result = pd.read_csv(path)
                self.assertTrue(path.exists())
            self.assertFalse(path.exists())
            self.assertEqual(original.read_text(encoding="utf-8").count("\n"), 2)

        self.assertEqual(len(result), 2)
        merge_latest_prices.assert_called_once_with(original, "BTC")


class ForecastFormTests(SimpleTestCase):
    def test_forecast_steps_are_bounded(self):
        self.assertFalse(
            NameForm(
                {
                    "model_name": "gru",
                    "data_source": "price_only",
                    "price_source": "bundled",
                    "predict_steps": 0,
                }
            ).is_valid()
        )
        self.assertTrue(
            NameForm(
                {
                    "model_name": "lstm",
                    "data_source": "with_news",
                    "price_source": "bundled",
                    "predict_steps": 24,
                }
            ).is_valid()
        )

    def test_asset_without_bundled_data_defaults_to_automatic_history(self):
        form = NameForm(bundled_available=False)
        self.assertEqual(form.fields["price_source"].choices[0][0], "forexsb")

        automatic = NameForm(
            {
                "model_name": "xgboost",
                "data_source": "price_only",
                "price_source": "forexsb",
                "predict_steps": 24,
            },
            bundled_available=False,
        )
        self.assertTrue(automatic.is_valid())

        submitted = NameForm(
            {
                "model_name": "xgboost",
                "data_source": "price_only",
                "price_source": "bundled",
                "predict_steps": 24,
            },
            bundled_available=False,
        )
        self.assertFalse(submitted.is_valid())
        self.assertTrue(
            NameForm(
                {
                    "model_name": "xgboost",
                    "data_source": "price_only",
                    "price_source": "bundled",
                    "predict_steps": 24,
                }
            ).is_valid()
        )


class NewsPriceOverlapTests(SimpleTestCase):
    def test_prices_keep_pre_news_context(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            prices = directory / "BTC.csv"
            prices.write_text(
                "Date,Close\n"
                "2026-08-19 06:00:00,100\n"
                "2026-08-19 07:00:00,101\n"
                "2026-08-19 08:00:00,102\n",
                encoding="utf-8",
            )
            news = directory / "BTC_news.csv"
            news.write_text(
                "published_at,asset_impact\n"
                "2026-08-19T07:00:00+00:00,0.5\n",
                encoding="utf-8",
            )

            with news_overlapping_price_csv(prices, news) as cropped:
                result = pd.read_csv(cropped)

        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]["Date"], "2026-08-19 06:00:00")

    def test_news_crop_keeps_exactly_one_model_window_before_news(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            prices = directory / "BTC.csv"
            dates = pd.date_range("2026-08-01", periods=100, freq="h", tz="UTC")
            pd.DataFrame(
                {"Date": dates.strftime("%Y-%m-%d %H:%M:%S"), "Close": range(100)}
            ).to_csv(prices, index=False)
            news = directory / "BTC_news.csv"
            news.write_text(
                f"published_at,asset_impact\n{dates[80].isoformat()},0.5\n",
                encoding="utf-8",
            )

            with news_overlapping_price_csv(prices, news) as cropped:
                result = pd.read_csv(cropped)

        self.assertEqual(len(result), 80)
        self.assertEqual(result.iloc[0]["Date"], "2026-08-01 20:00:00")

    def test_news_after_last_candle_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            prices = directory / "BTC.csv"
            prices.write_text(
                "Date,Close\n2026-08-19 06:00:00,100\n", encoding="utf-8"
            )
            news = directory / "BTC_news.csv"
            news.write_text(
                "published_at,asset_impact\n"
                "2026-08-20T07:00:00+00:00,0.5\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "begins after the final"):
                with news_overlapping_price_csv(prices, news):
                    pass


class ForexSbHistoryTests(SimpleTestCase):
    def test_fx_and_commodity_symbols_map_to_available_feeds(self):
        expected = {
            "AUD": "AUDUSD",
            "CAD": "USDCAD",
            "CHF": "USDCHF",
            "EUR": "EURUSD",
            "GBP": "GBPUSD",
            "JPY": "USDJPY",
            "NZD": "NZDUSD",
            "BRENT": "BRENTCMDUSD",
        }
        for symbol, feed_symbol in expected.items():
            with self.subTest(symbol=symbol):
                self.assertTrue(forexsb_service.supports_symbol(symbol))
                self.assertEqual(forexsb_service._feed_symbol(symbol), feed_symbol)

        self.assertFalse(forexsb_service.supports_symbol("USOIL"))

    def test_m30_binary_feed_is_aggregated_to_h1(self):
        metadata = gzip.compress(
            json.dumps(
                {"BTCUSDT": {"priceScale": 100, "volumeScale": 1}}
            ).encode()
        )
        records = gzip.compress(
            b"".join(
                (
                    struct.pack("<6i", 0, 10000, 10200, 9900, 10100, 4),
                    struct.pack("<6i", 30, 10100, 10300, 10000, 10200, 6),
                    struct.pack("<6i", 60, 10200, 10400, 10100, 10300, 5),
                )
            )
        )

        class Response:
            def __init__(self, content):
                self.content = content

            def raise_for_status(self):
                return None

        session = type("Session", (), {})()
        session.get = lambda url, **kwargs: Response(
            metadata if url == forexsb_service.INFO_URL else records
        )
        result = forexsb_service.download_h1_history("BTC", session=session)

        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["Open"], 100.0)
        self.assertEqual(result.iloc[0]["High"], 103.0)
        self.assertEqual(result.iloc[0]["Low"], 99.0)
        self.assertEqual(result.iloc[0]["Close"], 102.0)
        self.assertEqual(result.iloc[0]["Volume"], 10.0)


class PriceExportServiceTests(SimpleTestCase):
    @patch("dashboard.price_export_service.market_data_service.refreshed_price_csv")
    @patch("dashboard.price_export_service.forexsb_service.forexsb_h1_csv")
    def test_supported_asset_combines_forexsb_and_biquote(
        self, forexsb_h1_csv, refreshed_price_csv
    ):
        historical = Path("forexsb.csv")
        refreshed = Path("combined.csv")
        forexsb_h1_csv.return_value.__enter__.return_value = historical
        refreshed_price_csv.return_value.__enter__.return_value = refreshed

        with price_export_service.latest_price_csv("JPY") as result:
            self.assertEqual(result, refreshed)

        forexsb_h1_csv.assert_called_once_with("JPY")
        refreshed_price_csv.assert_called_once_with(historical, "JPY")

    @patch("dashboard.price_export_service.market_data_service.download_h1_history")
    def test_asset_without_forexsb_uses_biquote_history(self, download_history):
        download_history.return_value = pd.DataFrame(
            {"Date": ["2026-09-16 17:00:00"], "Close": [100.0]}
        )

        with price_export_service.latest_price_csv("USOIL") as path:
            self.assertTrue(path.exists())
            result = pd.read_csv(path)

        self.assertEqual(result["Close"].tolist(), [100.0])
        download_history.assert_called_once_with("USOIL")

class DashboardViewTests(SimpleTestCase):
    def test_dashboard_lists_available_asset(self):
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "BTC")
        self.assertNotContains(response, "Complete a price CSV")
        self.assertContains(response, "SOL")
        self.assertContains(
            response,
            reverse("download_prices", kwargs={"symbol_name": "BTC"}),
        )
        self.assertContains(response, "Download CSV")
        self.assertContains(response, 'class="live-price-indicator"')
        self.assertContains(response, "refreshLatestPrices()")

    @patch("dashboard.views.market_data_service.fetch_latest_prices")
    def test_latest_prices_endpoint_returns_dashboard_quotes(self, fetch_latest):
        fetch_latest.return_value = {
            "EUR": {
                "price": 1.14677,
                "live": True,
                "market_state": "open",
                "timestamp": "2026-09-16T19:18:49Z",
            }
        }
        response = self.client.get(reverse("latest_prices"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prices"]["EUR"]["price"], 1.14677)
        requested_symbols = fetch_latest.call_args.args[0]
        self.assertIn("EUR", requested_symbols)
        self.assertIn("XAU", requested_symbols)

    @patch("dashboard.views.latest_price_csv")
    def test_latest_price_csv_is_returned_as_attachment(self, latest_price_csv):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BTC_H1.csv"
            path.write_text(
                "Date,Close\n2026-09-16 17:00:00,100\n", encoding="utf-8"
            )
            latest_price_csv.return_value.__enter__.return_value = path
            response = self.client.get(
                reverse("download_prices", kwargs={"symbol_name": "BTC"})
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
        self.assertEqual(
            response["Content-Disposition"],
            'attachment; filename="BTC_latest_H1.csv"',
        )
        self.assertIn(b"2026-09-16 17:00:00,100", response.content)
        latest_price_csv.assert_called_once_with("BTC")

    def test_unknown_asset_price_download_returns_404(self):
        response = self.client.get(
            reverse("download_prices", kwargs={"symbol_name": "UNKNOWN"})
        )
        self.assertEqual(response.status_code, 404)

    def test_compact_market_groups_share_a_row_before_crypto(self):
        response = self.client.get(reverse("dashboard"))
        html = response.content.decode()
        compact_classes = (
            'class="market-group market-group-compact order-1 '
            'col-sm-6 col-lg-5"'
        )
        self.assertEqual(html.count(compact_classes), 2)
        self.assertContains(response, 'data-market-category="Commodities"')
        self.assertContains(response, 'data-market-category="FX majors"')
        self.assertContains(
            response, 'class="market-group order-2 col-12"', count=1
        )

    def test_crypto_without_bundled_history_opens_in_upload_mode(self):
        response = self.client.get(
            reverse("predict", kwargs={"symbol_name": "SOL"})
        )
        self.assertContains(
            response, "SOL history can be downloaded automatically from ForexSB"
        )
        self.assertContains(response, "Download H1 history automatically")
        self.assertContains(response, "Choose CSV from this computer")
        self.assertNotContains(response, "Use project /data CSV")

    def test_fx_major_without_bundled_history_uses_automatic_download(self):
        response = self.client.get(
            reverse("predict", kwargs={"symbol_name": "JPY"})
        )
        self.assertContains(
            response, "JPY history can be downloaded automatically from ForexSB"
        )
        self.assertContains(response, "Download H1 history automatically")

    def test_wti_without_bundled_history_uses_upload(self):
        response = self.client.get(
            reverse("predict", kwargs={"symbol_name": "USOIL"})
        )
        self.assertContains(response, "USOIL currently requires an uploaded")
        self.assertContains(response, "Choose CSV from this computer")
        self.assertNotContains(response, "Download H1 history automatically")

    def test_asset_page_displays_saved_news_section(self):
        response = self.client.get(
            reverse("predict", kwargs={"symbol_name": "BTC"})
        )
        self.assertContains(response, "Saved BTC news")
        self.assertContains(response, "Refresh news")

    @patch("dashboard.views.news_service.load_asset_news")
    def test_news_impact_chart_is_available_for_news_selection(self, load_news):
        load_news.return_value = {
            "total": 1,
            "path": Path("news.csv"),
            "rows": [{
                "published_iso": "2026-09-02T10:00:00+00:00",
                "published_at": "2026-09-02 10:00 UTC",
                "headline": "Market update",
                "source": "Example",
                "url": "https://example.com/news",
                "impact": 0.7,
            }],
        }
        response = self.client.get(reverse("predict", kwargs={"symbol_name": "BTC"}))

        self.assertContains(response, 'id="news-impact-section"')
        self.assertContains(response, 'id="news-impact-chart"')
        self.assertContains(response, "dataSource.value === 'with_news'")

    @patch("dashboard.views.forecaster.run_dashboard_forecast")
    def test_prediction_post_renders_chart(self, run_forecast):
        run_forecast.return_value = {
            "title": "BTC forecast",
            "actual_dates": ["2026-09-01 00:00"],
            "actual_prices": [100.0],
            "test_dates": ["2026-09-01 00:00"],
            "test_predictions": [101.0],
            "future_dates": ["2026-09-01 01:00"],
            "future_predictions": [102.0],
            "metrics": {
                "mae": 1.0,
                "rmse": 1.0,
                "directional_accuracy_percent": 50.0,
                "bullish_percent": 60.0,
                "bearish_percent": 40.0,
                "news_enabled": False,
            },
        }

        response = self.client.post(
            reverse("predict", kwargs={"symbol_name": "BTC"}),
            {
                "model_name": "gru",
                "data_source": "price_only",
                "price_source": "bundled",
                "predict_steps": 24,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "price-chart")
        self.assertContains(response, "View detailed chart")
        self.assertContains(response, "Download detailed PNG")
        self.assertContains(response, "rangeslider: {visible: true")
        self.assertContains(response, "tickformat: '%b %d<br>%H:%M'")
        html = response.content.decode()
        self.assertLess(html.index("plotly-latest.min.js"), html.index("Plotly.newPlot"))
        run_forecast.assert_called_once_with(
            symbol="BTC",
            model_name="gru",
            predict_steps=24,
            price_csv=None,
            price_source="bundled",
            use_news=False,
        )

    @patch("dashboard.views.forecaster.run_dashboard_forecast")
    def test_prediction_error_is_shown(self, run_forecast):
        run_forecast.side_effect = ValueError("Dataset is too small")
        response = self.client.post(
            reverse("predict", kwargs={"symbol_name": "BTC"}),
            {
                "model_name": "gru",
                "data_source": "with_news",
                "price_source": "bundled",
                "predict_steps": 24,
            },
        )
        self.assertContains(response, "Dataset is too small")
        run_forecast.assert_called_once_with(
            symbol="BTC",
            model_name="gru",
            predict_steps=24,
            price_csv=None,
            price_source="bundled",
            use_news=True,
        )

    @patch("dashboard.views.forecaster.run_dashboard_forecast")
    def test_uploaded_historical_csv_is_used_temporarily(self, run_forecast):
        observed = {}

        def inspect_upload(**kwargs):
            observed["path"] = kwargs["price_csv"]
            observed["contents"] = kwargs["price_csv"].read_text()
            raise ValueError("Stop after inspecting upload")

        run_forecast.side_effect = inspect_upload
        upload = SimpleUploadedFile(
            "history.csv",
            b"Date,Close\n2026-09-01 00:00,100\n",
            content_type="text/csv",
        )
        response = self.client.post(
            reverse("predict", kwargs={"symbol_name": "BTC"}),
            {
                "model_name": "gru",
                "data_source": "price_only",
                "price_source": "upload",
                "predict_steps": 24,
                "market_csv": upload,
            },
        )
        self.assertContains(response, "Stop after inspecting upload")
        self.assertIn("Date,Close", observed["contents"])
        self.assertFalse(observed["path"].exists())

    @patch("dashboard.views.news_service.crawl_asset_news")
    def test_crawl_news_updates_one_asset_and_redirects(self, crawl_news):
        crawl_news.return_value = {
            "symbol": "BTC",
            "fetched": 4,
            "saved": 10,
            "path": "data/news/BTC_news.csv",
        }
        response = self.client.post(
            reverse("crawl_news", kwargs={"symbol_name": "BTC"}),
            follow=True,
        )
        self.assertRedirects(
            response, reverse("predict", kwargs={"symbol_name": "BTC"})
        )
        self.assertContains(response, "fetched 4 relevant articles")
        crawl_news.assert_called_once_with("BTC")

    def test_crawl_news_is_post_only(self):
        response = self.client.get(
            reverse("crawl_news", kwargs={"symbol_name": "BTC"})
        )
        self.assertEqual(response.status_code, 405)


class ChartAdapterTests(SimpleTestCase):
    def test_metrics_are_converted_for_template(self):
        result = build_plotly_chart(
            {
                "chart": {"title": "test", "actual_dates": []},
                "model_metrics": {
                    "mae": 1.0,
                    "rmse": 2.0,
                    "directional_accuracy_percent": 55.0,
                },
                "forecast_direction_probability": {
                    "bullish_percent": 70.0,
                    "bearish_percent": 30.0,
                },
                "validation_selected_return_scales": [0.0, 0.0],
                "news": None,
            }
        )
        self.assertEqual(result["metrics"]["bullish_percent"], 70.0)
        self.assertFalse(result["metrics"]["news_enabled"])
        self.assertTrue(result["metrics"]["forecast_is_naive"])

    def test_chart_dates_are_displayed_in_tehran_time(self):
        result = build_plotly_chart(
            {
                "chart": {
                    "actual_dates": ["2026-09-12 06:00"],
                    "test_dates": ["2026-09-12 06:00"],
                    "future_dates": ["2026-09-12 07:00"],
                },
                "model_metrics": {
                    "mae": 1.0,
                    "rmse": 1.0,
                    "directional_accuracy_percent": 50.0,
                },
                "forecast_direction_probability": {
                    "bullish_percent": 50.0,
                    "bearish_percent": 50.0,
                },
                "news": None,
            }
        )

        self.assertEqual(result["actual_dates"], ["2026-09-12 09:30"])
        self.assertEqual(result["future_dates"], ["2026-09-12 10:30"])
        self.assertEqual(result["timezone"], "Asia/Tehran")

    def test_each_forecast_horizon_gets_its_own_calibration(self):
        actual = [[1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]]
        predicted = [[-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]]

        scales = calibrate_return_scales(actual, predicted)

        self.assertEqual(scales[0], 0.0)
        self.assertEqual(scales[1], 1.0)

    def test_split_decimal_comma_volume_is_reconstructed(self):
        upload = "Date,Close,Volume,\n2026-09-01 00:00,100,15,761\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "market.csv"
            path.write_text(upload, encoding="utf-8")
            frame = load_market_data(path, "Date", "Close")

        self.assertAlmostEqual(frame["Volume"].iloc[0], 15.761)
        self.assertNotIn("Unnamed: 3", frame)


class NewsDisplayTests(SimpleTestCase):
    def test_load_news_sorts_latest_first_and_rejects_unsafe_url(self):
        with tempfile.TemporaryDirectory() as directory:
            news_dir = Path(directory) / "news"
            news_dir.mkdir()
            path = news_dir / "BTC_news.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "published_at",
                        "headline",
                        "asset_impact",
                        "source",
                        "url",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "published_at": "2026-09-01T10:00:00+00:00",
                            "headline": "Older",
                            "asset_impact": "-0.5",
                            "source": "One",
                            "url": "https://example.com/older",
                        },
                        {
                            "published_at": "2026-09-02T10:00:00+00:00",
                            "headline": "Newer",
                            "asset_impact": "0.7",
                            "source": "Two",
                            "url": "javascript:alert(1)",
                        },
                    ]
                )

            with patch("dashboard.news_service.DATA_DIR", Path(directory)):
                result = load_asset_news("BTC")

        self.assertEqual(result["total"], 2)
        self.assertEqual(result["rows"][0]["headline"], "Newer")
        self.assertEqual(
            result["rows"][0]["published_iso"], "2026-09-02T10:00:00+00:00"
        )
        self.assertEqual(result["rows"][0]["url"], "")
