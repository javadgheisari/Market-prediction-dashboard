from django.contrib import admin

from dashboard.models import (
    BuyRankingRecord,
    ContextObservation,
    ForecastRecord,
    NewsArticle,
)


@admin.register(BuyRankingRecord)
class BuyRankingRecordAdmin(admin.ModelAdmin):
    list_display = ("generated_at", "market_name", "model_name")
    list_filter = ("model_name", "market_name")
    ordering = ("-generated_at",)


@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    list_display = ("symbol", "published_at", "source", "headline", "asset_impact")
    list_filter = ("symbol", "scoring_method")
    search_fields = ("headline", "source", "article_id")


@admin.register(ContextObservation)
class ContextObservationAdmin(admin.ModelAdmin):
    list_display = ("symbol", "available_at", "feature", "value", "source")
    list_filter = ("symbol", "feature", "source")


@admin.register(ForecastRecord)
class ForecastRecordAdmin(admin.ModelAdmin):
    list_display = (
        "symbol", "model_name", "issued_at", "target_at", "status",
        "predicted_close", "actual_close",
    )
    list_filter = ("symbol", "model_name", "status")
    ordering = ("-issued_at",)
