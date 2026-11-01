"""Check pending forecast outcomes without needing a dashboard visit."""

from django.core.management.base import BaseCommand

from dashboard.forecast_track_record import refresh_due_records
from dashboard.models import ForecastRecord


class Command(BaseCommand):
    help = "Verify pending live forecasts against matching closed BiQuote candles."

    def add_arguments(self, parser):
        parser.add_argument("--symbol", help="Check only one dashboard symbol")

    def handle(self, *args, **options):
        if options["symbol"]:
            symbols = [options["symbol"].strip().upper()]
        else:
            symbols = list(
                ForecastRecord.objects.filter(status=ForecastRecord.Status.PENDING)
                .order_by().values_list("symbol", flat=True).distinct()
            )
        for symbol in symbols:
            unavailable = refresh_due_records(symbol)
            remaining = ForecastRecord.objects.filter(
                symbol=symbol, status=ForecastRecord.Status.PENDING
            ).count()
            self.stdout.write(
                f"{symbol}: {remaining} pending"
                + ("; price feed temporarily unavailable" if unavailable else "")
            )
