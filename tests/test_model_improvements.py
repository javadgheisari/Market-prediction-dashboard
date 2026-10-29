import importlib.util
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


news_scorer = load_module("news_scorer")


class FakeResponse:
    def __init__(self, structured_output):
        self.structured_output = structured_output

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(self.structured_output),
                        }
                    ],
                }
            ]
        }


class FakeSession:
    def __init__(self, output):
        self.response = FakeResponse(output)
        self.request = None

    def post(self, *args, **kwargs):
        self.request = (args, kwargs)
        return self.response


class StructuredNewsScorerTests(unittest.TestCase):
    def test_scores_are_validated_and_mapped_by_article_id(self):
        session = FakeSession(
            {
                "scores": [
                    {
                        "article_id": "a1",
                        "asset_impact": 0.7,
                        "relevance": 0.9,
                        "importance": 0.8,
                        "confidence": 0.75,
                        "novelty": 0.6,
                        "impact_horizon_hours": 36,
                        "event_type": "macro",
                    }
                ]
            }
        )
        result = news_scorer.score_headlines_openai(
            "XAU",
            "gold",
            [{"article_id": "a1", "headline": "Fed signals rate cuts"}],
            api_key="test-key",
            session=session,
        )
        self.assertEqual(result["a1"]["asset_impact"], "0.7000")
        self.assertEqual(result["a1"]["event_type"], "macro")
        request_json = session.request[1]["json"]
        self.assertEqual(request_json["model"], "gpt-6-luna")
        self.assertTrue(request_json["text"]["format"]["strict"])

    def test_missing_api_key_is_actionable(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                news_scorer.score_headlines_openai("BTC", "Bitcoin", [])


class NewsFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.forecaster = load_module("sequence_forecaster")
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(str(error))

    def test_event_features_are_causal_and_horizon_aware(self):
        dates = pd.Series(pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"))
        news = pd.DataFrame(
            {
                "published_at": [dates.iloc[1]],
                "impact": [0.8],
                "quality": [1.0],
                "impact_horizon_hours": [2.0],
                "event_type": ["macro"],
            }
        )
        features = self.forecaster.build_news_features(dates, news, 24.0, 168.0)
        self.assertEqual(features.iloc[0]["news_attention_weight"], 0.0)
        self.assertEqual(features.iloc[1]["news_event_macro"], 1.0)
        self.assertGreater(
            features.iloc[1]["news_attention_weight"],
            features.iloc[3]["news_attention_weight"],
        )


class HorizonBacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.forecaster = load_module("sequence_forecaster")
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(str(error))

    def test_complete_direct_paths_are_scored_against_their_origin(self):
        close = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
        dates = pd.Series(pd.date_range("2026-01-01", periods=6, freq="h"))
        predictions = np.array([
            [np.log(130 / 120), np.log(140 / 130)],
            [np.log(140 / 130), np.log(150 / 140)],
            [10.0, 10.0],  # Its second target is outside the holdout.
        ])
        result = self.forecaster.horizon_backtest(
            close, dates, 3, predictions,
            validation_actual_returns=np.array([[0.1, 0.1], [-0.1, -0.1]]),
            validation_predicted_returns=np.zeros((2, 2)),
        )

        self.assertEqual(result["horizon_steps"], 2)
        self.assertEqual(result["sample_count"], 2)
        self.assertAlmostEqual(result["model_metrics"]["mae"], 0.0)
        self.assertAlmostEqual(result["naive_last_price_baseline"]["mae"], 20.0)
        self.assertEqual(result["model_metrics"]["directional_accuracy_percent"], 100.0)
        self.assertEqual(result["interval_coverage_percent"], 100.0)
        self.assertEqual(result["first_target_at"], "2026-01-01 04:00")
        self.assertEqual(result["last_target_at"], "2026-01-01 05:00")

    def test_short_holdout_reports_no_horizon_score(self):
        close = np.array([100.0, 101.0, 102.0, 103.0])
        dates = pd.Series(pd.date_range("2026-01-01", periods=4, freq="h"))
        result = self.forecaster.horizon_backtest(
            close, dates, 3, np.zeros((1, 2))
        )
        self.assertEqual(result["sample_count"], 0)
        self.assertNotIn("model_metrics", result)


class ChronosDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.forecaster = load_module("chronos_challenger")
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(str(error))

    def test_dashboard_contract_without_downloading_model(self):
        class FakeChronosPipeline:
            def predict_df(self, context, future_df, prediction_length, **kwargs):
                result = future_df[["id", "timestamp"]].copy()
                result["predictions"] = np.repeat(0.0002, prediction_length)
                result["0.1"] = np.repeat(-0.0004, prediction_length)
                result["0.5"] = np.repeat(0.0001, prediction_length)
                result["0.9"] = np.repeat(0.0006, prediction_length)
                return result

        dates = pd.date_range("2026-01-01", periods=120, freq="h")
        prices = 100 * np.exp(np.arange(120) * 0.00015)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BTC.csv"
            pd.DataFrame({"Date": dates, "Close": prices}).to_csv(path, index=False)
            result = self.forecaster.train_and_forecast_chronos(
                path,
                "Date",
                forecast_steps=3,
                context_length=48,
                pipeline=FakeChronosPipeline(),
            )

        self.assertEqual(result["model"], "CHRONOS_2")
        self.assertEqual(len(result["chart"]["future_predictions"]), 3)
        self.assertEqual(len(result["chart"]["future_lower"]), 3)
        self.assertIn("directional_accuracy_percent", result["model_metrics"])
        self.assertEqual(result["horizon_backtest"]["sample_count"], 1)
        self.assertEqual(result["horizon_backtest"]["method"], "single held-out origin")
        self.assertEqual(result["news"], None)


class PortfolioSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.forecaster = load_module("xgboost_forecaster")
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(str(error))

    def test_naive_member_can_reject_bad_models(self):
        actual = np.zeros((5, 2))
        predictions = np.ones((2, 5, 2))
        weights, combined, diagnostics = self.forecaster._select_portfolio_weights(
            actual, predictions
        )
        np.testing.assert_array_equal(weights, np.zeros_like(weights))
        np.testing.assert_array_equal(combined, np.zeros_like(actual))
        self.assertEqual(diagnostics[0]["selected"], "naive_zero")


if __name__ == "__main__":
    unittest.main()
