"""REST Framework serializers for the ticker APIs."""

from rest_framework import serializers

from dashboard.models import NewsArticle


class TickerNewsQuerySerializer(serializers.Serializer):
    page = serializers.IntegerField(min_value=1, default=1, required=False)
    page_size = serializers.IntegerField(
        min_value=1, max_value=100, default=25, required=False
    )


class NewsArticleSerializer(serializers.ModelSerializer):
    published_at = serializers.SerializerMethodField()

    class Meta:
        model = NewsArticle
        fields = (
            "id", "article_id", "published_at", "headline", "source", "url",
            "asset_impact", "relevance", "importance", "source_quality",
            "confidence", "novelty", "impact_horizon_hours", "event_type",
            "scoring_method",
        )

    def get_published_at(self, article: NewsArticle) -> str:
        return article.published_at.isoformat()


class LatestBuyRankingSerializer(serializers.Serializer):
    ticker = serializers.CharField()
    ranking_id = serializers.IntegerField()
    generated_at = serializers.CharField()
    market_name = serializers.CharField()
    model_name = serializers.CharField()
    settings = serializers.JSONField()
    recommendation = serializers.JSONField()
    aggregate = serializers.DictField()


class ErrorResponseSerializer(serializers.Serializer):
    error = serializers.CharField()


class TickerNewsResponseSerializer(serializers.Serializer):
    ticker = serializers.CharField()
    page = serializers.IntegerField()
    page_size = serializers.IntegerField()
    total = serializers.IntegerField()
    total_pages = serializers.IntegerField()
    next_page = serializers.IntegerField(allow_null=True)
    previous_page = serializers.IntegerField(allow_null=True)
    results = NewsArticleSerializer(many=True)
