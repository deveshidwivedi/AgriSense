import io
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR') 

import json
import random
from django.views.decorators.http import require_GET
from django.core.files.storage import default_storage
from datetime import datetime
import joblib
from rest_framework.decorators import api_view
from rest_framework import status

import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from rest_framework.response import Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2



def hello_world(request):
    return JsonResponse({"message": "Hello, World!"})
# Load the model at the beginning
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fast_plant_disease_model.h5")
model = load_model(MODEL_PATH)

# Load class names using the correct ordering (you can extract from a saved dictionary or hardcode)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
@api_view(['POST'])
def detect_disease(request):
    try:
        # Get the image file from the request
        image_file = request.FILES['image']
        
        # Save the image to a temporary path
        file_path = default_storage.save(f"temp/{image_file.name}", image_file)

        # Read and preprocess the image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (96, 96))
        img = img / 255.0
        img_array = np.expand_dims(img, axis=0)
        print(f"Image shape: {img_array.shape}")

        # Predict the disease using the model
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)

        # Check if the predicted class index is valid
        if predicted_index >= len(CLASS_NAMES):
            return Response({'error': 'Predicted class index is out of range.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Get the predicted disease name and confidence
        predicted_disease = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions))

        # Clean up: remove the temporary image file
        os.remove(file_path)

        # Return the prediction response
        return Response({
            'predicted_disease': predicted_disease,
            'confidence': confidence
        })

    except Exception as e:
        # Print the error for debugging
        print(f"Error during detection: {e}")
        
        # Return a 500 internal server error with the error message
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct full paths to model and scaler
MODEL_PATHH = os.path.join(BASE_DIR, 'models', 'crop_recommendation_model.pkl')
SCALER_PATHH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Load the model and scaler
model2 = joblib.load(MODEL_PATHH)
scaler = joblib.load(SCALER_PATHH)

@csrf_exempt
def crop_recommendation(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Extract feature values
            N = float(data.get("N"))
            P = float(data.get("P"))
            K = float(data.get("K"))
            temperature = float(data.get("temperature"))
            humidity = float(data.get("humidity"))
            ph = float(data.get("ph"))
            rainfall = float(data.get("rainfall"))

            # Arrange input data
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model2.predict(input_scaled)

            return JsonResponse({"recommended_crop": prediction[0]})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Only POST method is allowed."}, status=405)

# @require_GET
# def user_history(request):
#     # This is mock data; replace it with data from the database if needed
#     history_data = [
#         {
#             "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "disease": "Rust"
#         },
#         {
#             "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "crop_recommendation": "Wheat, Rice"
#         }
#     ]
#     return JsonResponse(history_data, safe=False)