from django.urls import path
from .views import detect_disease, hello_world

urlpatterns = [
    path('hello/', hello_world),
    path('detect-disease/', detect_disease),
]
