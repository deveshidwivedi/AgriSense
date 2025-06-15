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
from ultralytics import YOLO
import tempfile



def hello_world(request):
    return JsonResponse({"message": "Hello, World!"})
# Load the model at the beginning
# MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fast_plant_disease_model.h5")
# model = load_model(MODEL_PATH)

# Load class names using the correct ordering (you can extract from a saved dictionary or hardcode)
# CLASS_NAMES = [
#     'Apple___Apple_scab',
#     'Apple___Black_rot',
#     'Apple___Cedar_apple_rust',
#     'Apple___healthy',
#     'Blueberry___healthy',
#     'Cherry_(including_sour)___Powdery_mildew',
#     'Cherry_(including_sour)___healthy',
#     'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
#     'Corn_(maize)___Common_rust_',
#     'Corn_(maize)___Northern_Leaf_Blight',
#     'Corn_(maize)___healthy',
#     'Grape___Black_rot',
#     'Grape___Esca_(Black_Measles)',
#     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#     'Grape___healthy',
#     'Orange___Haunglongbing_(Citrus_greening)',
#     'Peach___Bacterial_spot',
#     'Peach___healthy',
#     'Pepper,_bell___Bacterial_spot',
#     'Pepper,_bell___healthy',
#     'Potato___Early_blight',
#     'Potato___Late_blight',
#     'Potato___healthy',
#     'Raspberry___healthy',
#     'Soybean___healthy',
#     'Squash___Powdery_mildew',
#     'Strawberry___Leaf_scorch',
#     'Strawberry___healthy',
#     'Tomato___Bacterial_spot',
#     'Tomato___Early_blight',
#     'Tomato___Late_blight',
#     'Tomato___Leaf_Mold',
#     'Tomato___Septoria_leaf_spot',
#     'Tomato___Spider_mites_Two-spotted_spider_mite',
#     'Tomato___Target_Spot',
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#     'Tomato___Tomato_mosaic_virus',
#     'Tomato___healthy'
# ]
# @api_view(['POST'])
# def detect_disease(request):
#     try:
#         # Get the image file from the request
#         image_file = request.FILES['image']
        
#         # Save the image to a temporary path
#         file_path = default_storage.save(f"temp/{image_file.name}", image_file)

#         # Read and preprocess the image
#         img = cv2.imread(file_path)
#         img = cv2.resize(img, (96, 96))
#         img = img / 255.0
#         img_array = np.expand_dims(img, axis=0)
#         print(f"Image shape: {img_array.shape}")

#         # Predict the disease using the model
#         predictions = model.predict(img_array)
#         predicted_index = np.argmax(predictions)

#         # Check if the predicted class index is valid
#         if predicted_index >= len(CLASS_NAMES):
#             return Response({'error': 'Predicted class index is out of range.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#         # Get the predicted disease name and confidence
#         predicted_disease = CLASS_NAMES[predicted_index]
#         confidence = float(np.max(predictions))

#         # Clean up: remove the temporary image file
#         os.remove(file_path)

#         # Return the prediction response
#         return Response({
#             'predicted_disease': predicted_disease,
#             'confidence': confidence
#         })

#     except Exception as e:
#         # Print the error for debugging
#         print(f"Error during detection: {e}")
        
#         # Return a 500 internal server error with the error message
#         return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolov8s.pt")
model = YOLO(MODEL_PATH)

# Class names
CLASS_NAMES = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf spot",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf late blight",
    "Potato leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato leaf",
    "Tomato mold leaf",
    "Tomato two spotted spider mites leaf",
    "grape leaf black rot",
    "grape leaf"
]

def draw_bounding_boxes(image_path, detections):
    """
    Draw bounding boxes on the image and return the annotated image as base64
    """
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Define colors for different detections
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    # Draw bounding boxes and labels
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bounding_box'].values()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get color for this detection
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)
        
        # Prepare label
        label = f"{detection['disease'].replace('_', ' ')}: {detection['confidence']:.2f}"
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw label background
        cv2.rectangle(img_rgb, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"

@api_view(['POST'])
def detect_disease(request):
    try:
        # Get the image file from the request
        image_file = request.FILES['image']
        
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            for chunk in image_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        try:
            # Make prediction using YOLO model
            results = model.predict(
                source=temp_file_path,
                conf=0.5,  # Confidence threshold
                verbose=False
            )
            
            detections = []
            
            # Process the results
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get detection data
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Ensure class_id is within valid range
                        if class_id < len(CLASS_NAMES):
                            class_name = CLASS_NAMES[class_id]
                            
                            detections.append({
                                'disease': class_name,
                                'confidence': confidence,
                                'bounding_box': {
                                    'x1': float(x1),
                                    'y1': float(y1),
                                    'x2': float(x2),
                                    'y2': float(y2)
                                }
                            })
            
            # Generate annotated image if detections found
            annotated_image = None
            if detections:
                annotated_image = draw_bounding_boxes(temp_file_path, detections)
            
            # Clean up: remove the temporary image file
            os.unlink(temp_file_path)
            
            # Return response based on detections
            if detections:
                # Find the detection with highest confidence
                best_detection = max(detections, key=lambda x: x['confidence'])
                
                return Response({
                    'predicted_disease': best_detection['disease'],
                    'confidence': best_detection['confidence'],
                    'all_detections': detections,
                    'detection_count': len(detections),
                    'annotated_image': annotated_image  # Base64 image with bounding boxes
                })
            else:
                return Response({
                    'predicted_disease': 'No disease detected',
                    'confidence': 0.0,
                    'all_detections': [],
                    'detection_count': 0,
                    'annotated_image': None
                })
        
        except Exception as model_error:
            # Clean up temp file in case of error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise model_error
            
    except Exception as e:
        # Print the error for debugging
        print(f"Error during detection: {e}")
        
        # Return a 500 internal server error with the error message
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct full paths to model and scaler
MODEL_PATHH = os.path.join(BASE_DIR, 'models', 'crop_r1.pkl')
SCALER_PATHH = os.path.join(BASE_DIR, 'models', 'scaler_1.pkl')

# Load the model and scaler
model2 = joblib.load(MODEL_PATHH)
scaler = joblib.load(SCALER_PATHH)


@csrf_exempt
def crop_recommendation(request):
    if request.method != 'POST':
        return JsonResponse({"detail": "Only POST method is allowed."}, status=405)

    try:
        data = json.loads(request.body)

        # Extract and validate input values
        try:
            N = float(data.get("N"))
            P = float(data.get("P"))
            K = float(data.get("K"))
            temperature = float(data.get("temperature"))
            humidity = float(data.get("humidity"))
            ph = float(data.get("ph"))
            rainfall = float(data.get("rainfall"))
        except (TypeError, ValueError):
            return JsonResponse({"detail": "All input values must be valid numbers."}, status=400)

        # Validate ranges
        if not (0 <= N <= 140):
            return JsonResponse({"detail": "Nitrogen must be between 0 and 140."}, status=400)
        if not (0 <= P <= 145):
            return JsonResponse({"detail": "Phosphorus must be between 0 and 145."}, status=400)
        if not (0 <= K <= 205):
            return JsonResponse({"detail": "Potassium must be between 0 and 205."}, status=400)
        if not (10 <= temperature <= 45):
            return JsonResponse({"detail": "Temperature must be between 10°C and 45°C."}, status=400)
        if not (20 <= humidity <= 100):
            return JsonResponse({"detail": "Humidity must be between 20% and 100%."}, status=400)
        if not (3.5 <= ph <= 9.5):
            return JsonResponse({"detail": "pH must be between 3.5 and 9.5."}, status=400)
        if not (0 <= rainfall <= 300):
            return JsonResponse({"detail": "Rainfall must be between 0 and 300mm."}, status=400)

        # Prepare input for prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        prediction = model2.predict(input_scaled)

        return JsonResponse({"recommended_crop": prediction[0]})

    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON format."}, status=400)

    except Exception as e:
        return JsonResponse({"detail": f"Server error: {str(e)}"}, status=500)

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