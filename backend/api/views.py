import base64
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
from PIL import Image, ImageDraw, ImageFont
import cv2
from ultralytics import YOLO

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

# YOLO model classes
YOLO_CLASS_NAMES = {
    0: 'Apple Scab Leaf',
    1: 'Apple leaf',
    2: 'Apple rust leaf',
    3: 'Bell_pepper leaf spot',
    4: 'Bell_pepper leaf',
    5: 'Blueberry leaf',
    6: 'Cherry leaf',
    7: 'Corn Gray leaf spot',
    8: 'Corn leaf blight',
    9: 'Corn rust leaf',
    10: 'Peach leaf',
    11: 'Potato leaf early blight',
    12: 'Potato leaf late blight',
    13: 'Potato leaf',
    14: 'Raspberry leaf',
    15: 'Soyabean leaf',
    16: 'Soybean leaf',
    17: 'Squash Powdery mildew leaf',
    18: 'Strawberry leaf',
    19: 'Tomato Early blight leaf',
    20: 'Tomato Septoria leaf spot',
    21: 'Tomato leaf bacterial spot',
    22: 'Tomato leaf late blight',
    23: 'Tomato leaf mosaic virus',
    24: 'Tomato leaf yellow virus',
    25: 'Tomato leaf',
    26: 'Tomato mold leaf',
    27: 'Tomato two spotted spider mites leaf',
    28: 'grape leaf black rot',
    29: 'grape leaf'
}

# Create mapping between classification and YOLO classes
def get_yolo_class_for_classification(classification_result):
    """Map classification result to expected YOLO class"""
    mapping = {
        'Potato___Early_blight': 11,  # 'Potato leaf early blight'
        'Potato___Late_blight': 12,   # 'Potato leaf late blight'
        'Apple___Apple_scab': 0,      # 'Apple Scab Leaf'
        'Apple___Cedar_apple_rust': 2, # 'Apple rust leaf'
        'Tomato___Early_blight': 19,  # 'Tomato Early blight leaf'
        'Tomato___Late_blight': 22,   # 'Tomato leaf late blight'
        'Tomato___Bacterial_spot': 21, # 'Tomato leaf bacterial spot'
        'Tomato___Septoria_leaf_spot': 20, # 'Tomato Septoria leaf spot'
        'Tomato___Leaf_Mold': 26,     # 'Tomato mold leaf'
        'Tomato___Spider_mites_Two-spotted_spider_mite': 27, # 'Tomato two spotted spider mites leaf'
        'Tomato___Tomato_mosaic_virus': 23, # 'Tomato leaf mosaic virus'
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 24, # 'Tomato leaf yellow virus'
        'Pepper,_bell___Bacterial_spot': 3, # 'Bell_pepper leaf spot'
        'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': 7, # 'Corn Gray leaf spot'
        'Corn_(maize)___Northern_Leaf_Blight': 8, # 'Corn leaf blight'
        'Corn_(maize)___Common_rust_': 9, # 'Corn rust leaf'
        'Grape___Black_rot': 28,      # 'grape leaf black rot'
        'Squash___Powdery_mildew': 17, # 'Squash Powdery mildew leaf'
    }
    return mapping.get(classification_result, None)

def hello_world(request):
    return JsonResponse({"message": "Hello, World!"})
# Load the model at the beginning
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fast_plant_disease_model.h5")
model = load_model(MODEL_PATH)
YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best.pt")

classification_model = load_model(MODEL_PATH)
detection_model = YOLO(YOLO_MODEL_PATH)

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

@api_view(['POST'])
def detect_disease(request):
    try:
        # Get the image file from the request
        image_file = request.FILES['image']

        # Save the image to a temporary path
        file_path = default_storage.save(f"temp/{image_file.name}", image_file)
        full_file_path = default_storage.path(file_path)

        # Read original image for YOLO detection
        original_img = cv2.imread(full_file_path)
        original_height, original_width = original_img.shape[:2]

        # Preprocess image for classification
        img_classification = cv2.resize(original_img, (96, 96))
        img_classification = img_classification / 255.0
        img_array = np.expand_dims(img_classification, axis=0)

        # Predict the disease using the classification model
        predictions = classification_model.predict(img_array)
        predicted_index = np.argmax(predictions)

        # Check if the predicted class index is valid
        if predicted_index >= len(CLASS_NAMES):
            return Response({'error': 'Predicted class index is out of range.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Get the predicted disease name and confidence
        predicted_disease = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions))

        # Initialize response data
        response_data = {
            'predicted_disease': predicted_disease,
            'confidence': confidence
        }

        # Check if disease is detected and not healthy
        is_healthy = 'healthy' in predicted_disease.lower()
        
        # Run YOLO detection if disease is detected with reasonable confidence
        if not is_healthy and confidence > 0.5:  # Increased threshold for better reliability
            print(f"Running YOLO detection for: {predicted_disease}")
            
            # Run YOLO detection with lower confidence threshold
            results = detection_model(full_file_path, conf=0.1, iou=0.5, verbose=False)

            # Convert original image to PIL for drawing
            pil_image = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            detected_objects = []
            
            # Get expected YOLO class for this classification result
            expected_yolo_class = get_yolo_class_for_classification(predicted_disease)

            # Process detection results
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                print(f"YOLO found {len(boxes)} potential detections")

                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence_det = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    print(f"Detection {i}: class_id={class_id}, confidence={confidence_det:.3f}, expected_class={expected_yolo_class}")

                    # Check if this is a disease class (not healthy leaf)
                    class_name = YOLO_CLASS_NAMES.get(class_id, f"Class_{class_id}")
                    is_healthy_class = any(word in class_name.lower() for word in ['leaf', 'healthy']) and not any(word in class_name.lower() for word in ['spot', 'blight', 'rust', 'mildew', 'rot', 'virus', 'mites'])
                    is_disease_class = not is_healthy_class
                    
                    # More flexible confidence thresholds
                    if class_id == expected_yolo_class:
                        min_confidence = 0.05  # Very low for exact match
                        box_color = "red"
                        label_suffix = " (exact match)"
                    elif is_disease_class:
                        min_confidence = 0.1   # Low for any disease
                        box_color = "orange"
                        label_suffix = " (disease detected)"
                    else:
                        min_confidence = 0.3   # Higher for healthy leaves
                        box_color = "green"
                        label_suffix = ""
                    
                    if confidence_det > min_confidence:
                        # Draw bounding box
                        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

                        # Add label with class name and confidence
                        label = f"{class_name}: {confidence_det:.2f}{label_suffix}"
                        
                        # Calculate text size and draw background
                        try:
                            # Try to load a font, fall back to default if not available
                            font = ImageFont.truetype("arial.ttf", 14)
                        except:
                            font = ImageFont.load_default()
                        
                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        # Draw background rectangle for text
                        draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill=box_color)
                        draw.text((x1+5, y1-text_height-2), label, fill="white", font=font)

                        detected_objects.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence_det),
                            'class_id': class_id,
                            'class_name': class_name,
                            'is_expected_class': class_id == expected_yolo_class,
                            'is_disease_class': is_disease_class,
                            'detection_type': 'exact_match' if class_id == expected_yolo_class else ('disease' if is_disease_class else 'healthy')
                        })

            # If still no detections, show any disease detection with very low threshold
            if len(detected_objects) == 0:
                print("No detections found, trying with very low threshold for any disease...")
                results_low = detection_model(full_file_path, conf=0.05, iou=0.3, verbose=False)
                
                if len(results_low) > 0 and results_low[0].boxes is not None:
                    boxes_low = results_low[0].boxes
                    for box in boxes_low:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence_det = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        class_name = YOLO_CLASS_NAMES.get(class_id, f"Class_{class_id}")
                        is_healthy_class = any(word in class_name.lower() for word in ['leaf', 'healthy']) and not any(word in class_name.lower() for word in ['spot', 'blight', 'rust', 'mildew', 'rot', 'virus', 'mites'])
                        is_disease_class = not is_healthy_class
                        
                        # Show any disease detection or exact match with very low confidence
                        if (is_disease_class or class_id == expected_yolo_class) and confidence_det > 0.05:
                            color = "yellow" if class_id == expected_yolo_class else "purple"
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            label = f"{class_name}: {confidence_det:.2f} (low conf)"
                            
                            try:
                                font = ImageFont.truetype("arial.ttf", 12)
                            except:
                                font = ImageFont.load_default()
                            
                            bbox = draw.textbbox((0, 0), label, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                            
                            draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill=color)
                            draw.text((x1+5, y1-text_height-2), label, fill="white", font=font)
                            
                            detected_objects.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(confidence_det),
                                'class_id': class_id,
                                'class_name': class_name,
                                'is_expected_class': class_id == expected_yolo_class,
                                'is_disease_class': is_disease_class,
                                'low_confidence': True,
                                'detection_type': 'low_confidence'
                            })

            print(f"Final detections: {len(detected_objects)}")

            # Convert PIL image back to base64 for frontend
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            response_data.update({
                'annotated_image': f"data:image/jpeg;base64,{img_base64}",
                'detected_objects': detected_objects,
                'has_detections': len(detected_objects) > 0,
                'expected_yolo_class': expected_yolo_class,
                'yolo_class_name': YOLO_CLASS_NAMES.get(expected_yolo_class, None) if expected_yolo_class else None
            })
        else:
            # For healthy plants or low confidence, return original image
            with open(full_file_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
            response_data.update({
                'annotated_image': f"data:image/jpeg;base64,{img_base64}",
                'detected_objects': [],
                'has_detections': False,
                'reason': 'healthy_plant' if is_healthy else 'low_confidence'
            })

        # Clean up: remove the temporary image file
        os.remove(full_file_path)

        return Response(response_data)

    except Exception as e:
        # Print the error for debugging
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

        # Clean up in case of error
        try:
            if 'full_file_path' in locals():
                os.remove(full_file_path)
        except:
            pass

        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# @csrf_exempt
# @require_POST
# def detect_disease(request):
#     return JsonResponse({
#         "message": "Test successful. API is working!"
#     })

# Dynamically get the base path relative to this file (views.py)
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Construct full paths to model and scaler
# MODEL_PATH = os.path.join(BASE_DIR, 'models', 'crop_recommendation_model.pkl')
# SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# # Load the model and scaler
# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)

# @csrf_exempt
# def crop_recommendation(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)

#             # Extract feature values
#             N = float(data.get("N"))
#             P = float(data.get("P"))
#             K = float(data.get("K"))
#             temperature = float(data.get("temperature"))
#             humidity = float(data.get("humidity"))
#             ph = float(data.get("ph"))
#             rainfall = float(data.get("rainfall"))

#             # Arrange input data
#             input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

#             # Scale input
#             input_scaled = scaler.transform(input_data)

#             # Predict
#             prediction = model.predict(input_scaled)

#             return JsonResponse({"recommended_crop": prediction[0]})

#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=400)

#     return JsonResponse({"error": "Only POST method is allowed."}, status=405)

@require_GET
def user_history(request):
    # This is mock data; replace it with data from the database if needed
    history_data = [
        {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disease": "Rust"
        },
        {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "crop_recommendation": "Wheat, Rice"
        }
    ]
    return JsonResponse(history_data, safe=False)