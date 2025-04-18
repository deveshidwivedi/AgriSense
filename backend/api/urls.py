from django.urls import path
from .views import crop_recommendation, detect_disease, hello_world
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('hello/', hello_world),
    path('detect-disease/', detect_disease),
    path('crop-recommendation/', crop_recommendation),
    # path('user-history/', user_history),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)