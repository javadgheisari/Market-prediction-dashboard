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
