"""Market observations used by the dashboard."""

from django.db import models


class NewsArticle(models.Model):
    symbol = models.CharField(max_length=20, db_index=True)
    article_id = models.CharField(max_length=64)
    published_at = models.DateTimeField(db_index=True)
    headline = models.TextField()
    asset_impact = models.FloatField(default=0)
    relevance = models.FloatField(default=0.5)
    importance = models.FloatField(default=0.5)
    source_quality = models.FloatField(default=0.5)
    confidence = models.FloatField(default=0.35)
    novelty = models.FloatField(default=0.5)
    impact_horizon_hours = models.FloatField(default=48)
    event_type = models.CharField(max_length=100, blank=True)
    scoring_method = models.CharField(max_length=100, blank=True)
    source = models.CharField(max_length=255, blank=True)
    url = models.TextField(blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["symbol", "article_id"], name="news_symbol_article_unique")
        ]
        ordering = ["-published_at", "-pk"]


class ContextObservation(models.Model):
    symbol = models.CharField(max_length=20, db_index=True)
    available_at = models.DateTimeField(db_index=True)
    feature = models.CharField(max_length=120)
    value = models.FloatField()
    source = models.CharField(max_length=120)
    max_age_hours = models.FloatField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["symbol", "available_at", "feature", "source"],
                name="context_observation_unique",
            )
        ]
        indexes = [models.Index(fields=["symbol", "feature", "available_at"])]
