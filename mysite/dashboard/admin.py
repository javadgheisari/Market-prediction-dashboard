from django.contrib import admin

from dashboard.models import ContextObservation, NewsArticle


@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    list_display = ("symbol", "published_at", "source", "headline", "asset_impact")
    list_filter = ("symbol", "scoring_method")
    search_fields = ("headline", "source", "article_id")


@admin.register(ContextObservation)
class ContextObservationAdmin(admin.ModelAdmin):
    list_display = ("symbol", "available_at", "feature", "value", "source")
    list_filter = ("symbol", "feature", "source")
