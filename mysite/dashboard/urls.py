from django.urls import path

from dashboard import views

urlpatterns = [
    path('', views.DashboardView.as_view(), name='dashboard'),
    path(
        'recommendations/',
        views.GeneralRecommendationsView.as_view(),
        name='recommendations',
    ),
    path('prices/latest/', views.LatestPricesView.as_view(), name='latest_prices'),
    path('predict/<str:symbol_name>/', views.PredictView.as_view(), name='predict'),
    path('prices/<str:symbol_name>/download/', views.PriceCsvDownloadView.as_view(), name='download_prices'),
    path('news/<str:symbol_name>/crawl/', views.NewsCrawlView.as_view(), name='crawl_news'),
    path('news/<str:symbol_name>/clear/', views.NewsClearView.as_view(), name='clear_news'),
    path(
        'markets/<slug:market_slug>/recommendations/',
        views.MarketRecommendationsView.as_view(),
        name='market_recommendations',
    ),
]
