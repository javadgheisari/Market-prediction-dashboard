"""Load the project's legacy market CSV observations into PostgreSQL."""

from pathlib import Path

from django.core.management.base import BaseCommand

from dashboard.data_paths import DATA_DIR
from dashboard.market_repository import import_context_file, import_news_file


class Command(BaseCommand):
    help = "Idempotently import data/news and data/context CSV files into the dashboard tables."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=Path, default=DATA_DIR)

    def handle(self, *args, **options):
        data_dir = options["data_dir"]
        totals = {"news": 0, "context": 0}
        for kind, suffix, importer in (
            ("news", "_news.csv", import_news_file),
            ("context", "_context.csv", import_context_file),
        ):
            paths = sorted((data_dir / kind).glob(f"*{suffix}"))
            for path in paths:
                totals[kind] += importer(path)
            self.stdout.write(f"{kind}: {len(paths)} files, {totals[kind]} unique rows imported")
        self.stdout.write(self.style.SUCCESS("Market data import complete."))
