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


class ForecastRecord(models.Model):
    """An issued forecast and its later, independently observed outcome."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        CORRECT = "correct", "Correct"
        INCORRECT = "incorrect", "Incorrect"
        NO_DIRECTION = "no_direction", "No directional call"

    symbol = models.CharField(max_length=20, db_index=True)
    model_name = models.CharField(max_length=40)
    price_source = models.CharField(max_length=20)
    used_news = models.BooleanField(default=False)
    issued_at = models.DateTimeField()
    origin_at = models.DateTimeField()
    target_at = models.DateTimeField()
    interval_minutes = models.PositiveIntegerField()
    horizon_steps = models.PositiveIntegerField()
    origin_close = models.FloatField()
    predicted_close = models.FloatField()
    actual_close = models.FloatField(null=True, blank=True)
    actual_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    last_checked_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-issued_at", "-pk"]
        indexes = [models.Index(fields=["symbol", "status", "target_at"])]


class BuyRankingRecord(models.Model):
    """A generated buy ranking and the suggestions shown for that run."""

    generated_at = models.DateTimeField(auto_now_add=True, db_index=True)
    market_name = models.CharField(max_length=120)
    model_name = models.CharField(max_length=40)
    settings = models.JSONField(default=dict, blank=True)
    suggestions = models.JSONField(default=list)

    class Meta:
        ordering = ["-generated_at", "-pk"]


class BuyRankingSuggestion(models.Model):
    """One ticker's result from a saved buy ranking run."""

    ranking = models.ForeignKey(
        BuyRankingRecord, on_delete=models.CASCADE, related_name="ticker_suggestions"
    )
    symbol = models.CharField(max_length=20, db_index=True)
    suggestion = models.JSONField()

    class Meta:
        indexes = [models.Index(fields=["symbol", "ranking"])]
