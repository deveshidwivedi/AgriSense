from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import random
# Create your views here.
from django.http import JsonResponse

def hello_world(request):
    return JsonResponse({"message": "Hello, World!"})

@csrf_exempt
@require_POST
def detect_disease(request):
    image = request.FILES.get('image')

    if not image:
        return JsonResponse({'error': 'No image uploaded.'}, status=400)

    # Dummy prediction logic
    diseases = ['Leaf Blight', 'Powdery Mildew', 'Rust', 'Healthy']
    detected = random.choice(diseases)

    return JsonResponse({'disease': detected})