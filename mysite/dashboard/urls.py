from django.urls import path

from dashboard import views

urlpatterns = [
    path('', views.DashboardView.as_view(), name='dashboard'),
    path('prices/complete/', views.CompleteCsvView.as_view(), name='complete_csv'),
    path('predict/<str:symbol_name>/', views.PredictView.as_view(), name='predict'),
    path('news/<str:symbol_name>/crawl/', views.NewsCrawlView.as_view(), name='crawl_news'),
]
