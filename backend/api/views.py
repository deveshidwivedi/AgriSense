import json
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

@csrf_exempt
def crop_recommendation(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Extract values
            soil_type = data.get("soil_type")
            ph = float(data.get("ph"))
            nitrogen = float(data.get("nitrogen"))
            phosphorus = float(data.get("phosphorus"))
            potassium = float(data.get("potassium"))
            temperature = float(data.get("temperature"))
            humidity = float(data.get("humidity"))
            rainfall = float(data.get("rainfall"))

            # Mock logic for now (replace with ML model later)
            if nitrogen > 100 and ph > 6:
                recommendation = "Wheat, Rice"
            else:
                recommendation = "Maize, Barley"

            return JsonResponse({"recommended_crops": recommendation})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Only POST method allowed."}, status=405)