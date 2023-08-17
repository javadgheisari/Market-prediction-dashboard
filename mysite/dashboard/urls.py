from django.urls import path

from dashboard import views

urlpatterns = [
    path('', views.DashboardView.as_view(), name='dashboard'),
    path('predict/<str:symbol_name>/', views.PredictView.as_view(), name='predict'),
]