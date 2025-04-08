from django.urls import path
from .views import crop_recommendation, detect_disease, hello_world

urlpatterns = [
    path('hello/', hello_world),
    path('detect-disease/', detect_disease),
    path('crop-recommendation/', crop_recommendation),
]
