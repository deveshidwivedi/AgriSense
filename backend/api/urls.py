from django.urls import path
from .views import  detect_disease, hello_world, crop_recommendation 

urlpatterns = [
    path('hello/', hello_world),
    path('detect-disease/', detect_disease),
    path('crop-recommendation/', crop_recommendation),
    # path('user-history/', user_history),
]
