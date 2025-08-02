import base64
import io
import os
import json
import numpy as np
import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.core.files.storage import default_storage
from datetime import datetime
from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.response import Response
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf
import joblib

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

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fast_plant_disease_model.h5")
classification_model = load_model(MODEL_PATH)

def get_mobilenet_last_conv_layer(model):
    """
    Get a suitable convolutional layer for MobileNetV2-based models.
    We'll try layers from different depths to get better spatial resolution.
    """
    # Try layers with different spatial resolutions for better visualization
    target_layers = [
        # Higher resolution layers (better for visualization)
        'block_13_expand_relu',   # Earlier block with more spatial detail
        'block_12_expand_relu',   # Even earlier for more detail
        'block_10_expand_relu',   # Mid-level features
        
        # Standard end layers
        'out_relu',               # ReLU after last conv (3x3)
        'Conv_1',                 # Last conv layer (3x3)
        'block_16_project',       # Last MobileNet block conv
        'block_16_expand_relu',   # Alternative in block 16
        'block_15_project',       # Previous block
        'block_15_expand_relu'
    ]
    
    print("Trying to find suitable layer for GradCAM...")
    
    for layer_name in target_layers:
        try:
            layer = model.get_layer(layer_name)
            print(f"Found layer: {layer_name} - Type: {type(layer).__name__}")
            
            # Test if we can create a model with this layer
            try:
                test_model = Model(inputs=model.input, outputs=layer.output)
                output_shape = test_model.output_shape
                print(f"Layer {layer_name} output shape: {output_shape}")
                
                # Check if it's 4D (suitable for GradCAM)
                if len(output_shape) == 4:
                    # Prefer layers with higher spatial resolution for better visualization
                    spatial_size = output_shape[1] * output_shape[2] if output_shape[1] is not None else 0
                    print(f"Selected layer: {layer_name} with shape {output_shape}, spatial size: {spatial_size}")
                    return layer_name
                else:
                    print(f"Layer {layer_name} has {len(output_shape)}D output, need 4D")
                    
            except Exception as e:
                print(f"Cannot create model with layer {layer_name}: {e}")
                continue
                
        except Exception as e:
            print(f"Layer {layer_name} not found: {e}")
            continue
    
    print("No suitable layer found")
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate GradCAM heatmap for MobileNetV2-based models.
    """
    print(f"Creating GradCAM for layer: {last_conv_layer_name}")
    
    try:
        # Get the target layer
        target_layer = model.get_layer(last_conv_layer_name)
        
        # Create a model that maps the input image to the activations of the target layer
        # as well as the output predictions
        grad_model = Model(
            inputs=[model.inputs],
            outputs=[target_layer.output, model.output]
        )
        
        print(f"Grad model created successfully")
        
        # Compute the gradient of the top predicted class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        print(f"Forward pass completed, computing gradients...")
        
        # Get the gradients of the class output with respect to the feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            raise ValueError("Gradients are None - the layer might not be suitable for GradCAM")
        
        # Global average pooling of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel by its importance and sum
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        print(f"GradCAM heatmap generated successfully, shape: {heatmap.shape}")
        
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Error in make_gradcam_heatmap: {e}")
        raise e

def apply_gradcam_to_image(original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Apply GradCAM heatmap overlay to the original image with enhanced visualization.
    """
    # Get original image dimensions
    original_height, original_width = original_img.shape[:2]
    
    print(f"Original image shape: {original_img.shape}")
    print(f"Heatmap shape before resize: {heatmap.shape}")
    
    # Use high-quality interpolation for upsampling small heatmaps
    if heatmap.shape[0] < 32 or heatmap.shape[1] < 32:
        # For very small heatmaps, use cubic interpolation
        heatmap_resized = cv2.resize(heatmap, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    else:
        # For larger heatmaps, use linear interpolation
        heatmap_resized = cv2.resize(heatmap, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    
    print(f"Heatmap shape after resize: {heatmap_resized.shape}")
    
    # Apply Gaussian smoothing for better visual quality
    # Kernel size proportional to image size
    kernel_size = max(3, min(15, original_width // 32))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    heatmap_smooth = cv2.GaussianBlur(heatmap_resized, (kernel_size, kernel_size), 0)
    
    # Enhance contrast slightly
    heatmap_enhanced = np.power(heatmap_smooth, 0.8)  # Gamma correction
    
    # Convert heatmap to RGB using the specified colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_enhanced), colormap)
    
    # Convert BGR to RGB for consistency
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure original image is in RGB format
    if len(original_img.shape) == 3:
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = original_img
    
    # Create a more sophisticated mask
    # Only overlay where heatmap is meaningful (above threshold)
    threshold = 0.15
    mask = heatmap_enhanced > threshold
    
    # Create superimposed image
    superimposed_img = original_rgb.copy().astype(np.float32)
    overlay = heatmap_colored.astype(np.float32)
    
    # Apply overlay with adaptive alpha based on heatmap intensity
    for c in range(3):  # RGB channels
        # Use heatmap values to modulate alpha (stronger areas get more overlay)
        adaptive_alpha = alpha * heatmap_enhanced
        superimposed_img[:, :, c] = np.where(
            mask,
            adaptive_alpha * overlay[:, :, c] + (1 - adaptive_alpha) * superimposed_img[:, :, c],
            superimposed_img[:, :, c]
        )
    
    print(f"Final image shape: {superimposed_img.shape}")
    
    return superimposed_img.astype(np.uint8)

def hello_world(request):
    return JsonResponse({"message": "Hello, World!"})

@api_view(['POST'])
def detect_disease(request):
    try:
        # Get the image file from the request
        image_file = request.FILES['image']

        # Save the image to a temporary path
        file_path = default_storage.save(f"temp/{image_file.name}", image_file)
        full_file_path = default_storage.path(file_path)

        # Read original image
        original_img = cv2.imread(full_file_path)
        if original_img is None:
            return Response({'error': 'Could not read the uploaded image'}, status=status.HTTP_400_BAD_REQUEST)
        
        original_height, original_width = original_img.shape[:2]

        # Preprocess image for classification (MobileNetV2 expects 96x96)
        img_classification = cv2.resize(original_img, (96, 96))
        img_classification = img_classification / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_classification, axis=0)

        print(f"Input image shape: {img_array.shape}")

        # Predict the disease using the classification model
        predictions = classification_model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predicted index: {predicted_index}")

        # Check if the predicted class index is valid
        if predicted_index >= len(CLASS_NAMES):
            return Response({'error': 'Predicted class index is out of range.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Get the predicted disease name and confidence
        predicted_disease = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions))

        print(f"Predicted disease: {predicted_disease}, Confidence: {confidence}")

        # Initialize response data
        response_data = {
            'predicted_disease': predicted_disease,
            'confidence': confidence
        }

        # Check if disease is detected and not healthy
        is_healthy = 'healthy' in predicted_disease.lower()
        
        # Generate GradCAM for any prediction with reasonable confidence
        if confidence > 0.3:  # Generate GradCAM for both healthy and diseased plants
            print(f"Generating GradCAM for: {predicted_disease}")
            
            try:
                # Find the last convolutional layer for MobileNetV2
                last_conv_layer_name = get_mobilenet_last_conv_layer(classification_model)
                
                if last_conv_layer_name is None:
                    print("Warning: No suitable convolutional layer found in the model")
                    
                    # Fallback to original image
                    with open(full_file_path, "rb") as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode()
                    response_data.update({
                        'gradcam_image': f"data:image/jpeg;base64,{img_base64}",
                        'gradcam_generated': False,
                        'gradcam_error': "No suitable convolutional layer found"
                    })
                else:
                    print(f"Using layer: {last_conv_layer_name}")
                    
                    # Generate GradCAM heatmap
                    heatmap = make_gradcam_heatmap(
                        img_array, 
                        classification_model, 
                        last_conv_layer_name, 
                        pred_index=predicted_index
                    )
                    
                    print(f"Heatmap shape: {heatmap.shape}")
                    print(f"Heatmap min/max: {np.min(heatmap):.3f}/{np.max(heatmap):.3f}")
                    
                    # Apply GradCAM overlay to original image
                    # Use different color schemes for healthy vs diseased
                    colormap = cv2.COLORMAP_VIRIDIS if is_healthy else cv2.COLORMAP_JET
                    alpha = 0.4 if is_healthy else 0.5
                    
                    gradcam_img = apply_gradcam_to_image(
                        original_img, 
                        heatmap, 
                        alpha=alpha, 
                        colormap=colormap
                    )
                    
                    # Convert to PIL and then to base64
                    gradcam_pil = Image.fromarray(gradcam_img)
                    buffered = io.BytesIO()
                    gradcam_pil.save(buffered, format="JPEG", quality=95)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    response_data.update({
                        'gradcam_image': f"data:image/jpeg;base64,{img_base64}",
                        'gradcam_generated': True,
                        'visualization_type': 'gradcam',
                        'last_conv_layer': last_conv_layer_name,
                        'colormap_used': 'viridis' if is_healthy else 'jet'
                    })
                    
            except Exception as gradcam_error:
                print(f"GradCAM generation failed: {gradcam_error}")
                import traceback
                traceback.print_exc()
                
                # Fallback to original image
                with open(full_file_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                response_data.update({
                    'gradcam_image': f"data:image/jpeg;base64,{img_base64}",
                    'gradcam_generated': False,
                    'gradcam_error': str(gradcam_error)
                })
        else:
            # For very low confidence, return original image
            print(f"Low confidence ({confidence:.3f}), skipping GradCAM")
            with open(full_file_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
            response_data.update({
                'gradcam_image': f"data:image/jpeg;base64,{img_base64}",
                'gradcam_generated': False,
                'reason': 'low_confidence'
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