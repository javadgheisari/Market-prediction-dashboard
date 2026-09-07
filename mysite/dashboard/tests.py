import csv
import tempfile
from unittest.mock import patch
from pathlib import Path

from django.test import SimpleTestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from dashboard.data_paths import list_available_assets, resolve_data_csv
from dashboard.forms import NameForm
from dashboard.forecaster import build_plotly_chart
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

    def test_symbol_resolves_to_existing_csv(self):
        self.assertTrue(resolve_data_csv("BTC").is_file())

    def test_xau_uses_the_most_complete_bundled_history(self):
        self.assertEqual(resolve_data_csv("XAU").name, "XAUUSD_M15.csv")


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


class DashboardViewTests(SimpleTestCase):
    def test_dashboard_lists_available_asset(self):
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "BTC")

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
        html = response.content.decode()
        self.assertLess(html.index("plotly-latest.min.js"), html.index("Plotly.newPlot"))
        run_forecast.assert_called_once_with(
            symbol="BTC",
            model_name="gru",
            predict_steps=24,
            price_csv=None,
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
