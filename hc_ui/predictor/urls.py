from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_view, name='predict'),
    path('download-pdf/', views.download_pdf, name='download_pdf'),
]