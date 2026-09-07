import csv
import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timezone
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "news_crawler.py"
SPEC = importlib.util.spec_from_file_location("news_crawler", MODULE_PATH)
news_crawler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = news_crawler
SPEC.loader.exec_module(news_crawler)


RSS = b"""<?xml version="1.0"?>
<rss><channel><item>
  <title>Apple stock rises after earnings beat</title>
  <link>https://example.com/apple-story</link>
  <pubDate>Fri, 05 Sep 2026 10:00:00 GMT</pubDate>
  <source>Example News</source>
</item></channel></rss>"""


class NewsCrawlerTests(unittest.TestCase):
    def test_parse_rss_and_score(self):
        article = news_crawler.parse_feed(RSS)[0]
        self.assertEqual(article.published_at.tzinfo, timezone.utc)
        self.assertEqual(article.source, "Example News")
        row = news_crawler.article_row(article, ["Apple", "AAPL"])
        self.assertGreater(float(row["asset_impact"]), 0)
        self.assertGreater(float(row["relevance"]), 0.5)

    def test_merge_deduplicates_by_url(self):
        article = news_crawler.parse_feed(RSS)[0]
        row = news_crawler.article_row(article, ["Apple"])
        self.assertEqual(len(news_crawler.merge_rows([row.copy()], [row.copy()])), 1)

    def test_relevance_uses_symbol_boundaries(self):
        self.assertTrue(news_crawler.is_relevant("Mattel (MAT) reports earnings", ["MAT"]))
        self.assertFalse(news_crawler.is_relevant("Markets rally today", ["MAT"]))

    def test_write_rows_uses_training_schema(self):
        article = news_crawler.parse_feed(RSS)[0]
        row = news_crawler.article_row(article, ["Apple"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "AAPL_news.csv"
            news_crawler.write_rows(path, [row])
            with path.open(newline="", encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))
        self.assertEqual(list(saved[0]), list(news_crawler.CSV_FIELDS))
        self.assertEqual(saved[0]["headline"], article.headline)

    def test_crawl_one_asset_saves_news(self):
        article = news_crawler.Article(
            datetime.now(timezone.utc),
            "Bitcoin BTC rises after ETF news",
            "Example News",
            "https://example.com/bitcoin-story",
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with mock.patch.object(
                news_crawler, "fetch_feed", return_value=[article]
            ):
                result = news_crawler.crawl_asset_news(
                    "btc", output_dir=output_dir
                )

            self.assertEqual(result["symbol"], "BTC")
            self.assertEqual(result["fetched"], 1)
            self.assertTrue((output_dir / "BTC_news.csv").is_file())


if __name__ == "__main__":
    unittest.main()
