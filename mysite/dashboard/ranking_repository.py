"""Persistence and lookup for generated buy rankings."""

from django.db import transaction

from dashboard.models import BuyRankingRecord, BuyRankingSuggestion


def save_buy_ranking(*, market_name, model_name, settings, suggestions):
    """Save a run and its ticker results together."""
    with transaction.atomic():
        ranking = BuyRankingRecord.objects.create(
            market_name=market_name,
            model_name=model_name,
            settings=settings,
            suggestions=suggestions,
        )
        BuyRankingSuggestion.objects.bulk_create(
            [
                BuyRankingSuggestion(
                    ranking=ranking,
                    symbol=str(item["symbol"]).upper(),
                    suggestion=item,
                )
                for item in suggestions
                if item.get("symbol")
            ]
        )
    return ranking


def latest_aggregate_for_symbol(symbol):
    """Return the newest successful price projection for a ticker."""
    rows = (
        BuyRankingSuggestion.objects.filter(symbol=symbol)
        .select_related("ranking")
        .order_by("-ranking__generated_at", "-ranking_id", "-pk")
    )
    for row in rows.iterator():
        suggestion = row.suggestion
        projection = suggestion.get("projection") or {}
        if suggestion.get("error") or not projection.get("aggregate"):
            continue
        return row
    return None
