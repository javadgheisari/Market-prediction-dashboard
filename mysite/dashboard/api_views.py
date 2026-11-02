"""Read-only REST Framework endpoints for saved rankings and news."""

from rest_framework.exceptions import NotFound, ValidationError
from rest_framework.pagination import PageNumberPagination
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, OpenApiResponse, extend_schema

from dashboard.api_serializers import (
    LatestBuyRankingSerializer,
    NewsArticleSerializer,
    ErrorResponseSerializer,
    TickerNewsQuerySerializer,
    TickerNewsResponseSerializer,
)
from dashboard.models import NewsArticle
from dashboard.ranking_repository import latest_aggregate_for_symbol
from dashboard.views import normalize_symbol


def _normalize_api_ticker(ticker):
    try:
        return normalize_symbol(ticker)
    except ValueError as exc:
        raise ValidationError({"error": str(exc)}) from exc


class TickerNewsPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100
    template = None

    def get_paginated_response(self, data):
        return Response({
            "ticker": self.ticker,
            "page": self.page.number,
            "page_size": self.get_page_size(self.request),
            "total": self.page.paginator.count,
            "total_pages": self.page.paginator.num_pages,
            "next_page": self.page.next_page_number() if self.page.has_next() else None,
            "previous_page": self.page.previous_page_number() if self.page.has_previous() else None,
            "results": data,
        })


class LatestBuyRankingApiView(APIView):
    renderer_classes = [JSONRenderer]

    @extend_schema(
        tags=["Tickers"],
        summary="Get the latest aggregate buy ranking for a ticker",
        description=(
            "Returns the newest saved price-model ranking for this ticker, "
            "including the aggregate projected path and component model paths. "
            "News-only rankings do not contain a numeric price projection."
        ),
        parameters=[OpenApiParameter(
            name="ticker",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            description="Ticker symbol, such as BTC or EUR.",
        )],
        responses={
            200: LatestBuyRankingSerializer,
            400: ErrorResponseSerializer,
            404: OpenApiResponse(description="No saved model projection is available."),
        },
    )
    def get(self, request, ticker):
        symbol = _normalize_api_ticker(ticker)
        row = latest_aggregate_for_symbol(symbol)
        if row is None:
            raise NotFound({"error": f"No saved model projection is available for {symbol}."})

        suggestion = row.suggestion.copy()
        projection = suggestion.pop("projection")
        data = {
            "ticker": symbol,
            "ranking_id": row.ranking_id,
            "generated_at": row.ranking.generated_at.isoformat(),
            "market_name": row.ranking.market_name,
            "model_name": row.ranking.model_name,
            "settings": row.ranking.settings,
            "recommendation": suggestion,
            "aggregate": {
                "unit": "percent_change_from_latest_price",
                "path": projection["aggregate"],
                "model_paths": projection["paths"],
            },
        }
        return Response(LatestBuyRankingSerializer(data).data)


class TickerNewsApiView(APIView):
    renderer_classes = [JSONRenderer]

    @extend_schema(
        tags=["Tickers"],
        summary="List saved news for a ticker",
        description="Returns saved news from newest to oldest with page metadata.",
        parameters=[
            OpenApiParameter(
                name="ticker",
                type=OpenApiTypes.STR,
                location=OpenApiParameter.PATH,
                description="Ticker symbol, such as BTC or EUR.",
            ),
            TickerNewsQuerySerializer,
        ],
        responses={
            200: TickerNewsResponseSerializer,
            400: OpenApiResponse(description="Ticker or pagination parameters are invalid."),
            404: OpenApiResponse(description="Requested page is out of range."),
        },
    )
    def get(self, request, ticker):
        symbol = _normalize_api_ticker(ticker)
        query = TickerNewsQuerySerializer(data=request.query_params)
        query.is_valid(raise_exception=True)

        articles = NewsArticle.objects.filter(symbol=symbol).order_by(
            "-published_at", "-pk"
        )
        paginator = TickerNewsPagination()
        paginator.ticker = symbol
        page = paginator.paginate_queryset(articles, request, view=self)
        return paginator.get_paginated_response(
            NewsArticleSerializer(page, many=True).data
        )
