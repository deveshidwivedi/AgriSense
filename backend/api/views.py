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

DISEASE_INFO = {
    'Apple___Apple_scab': {
        'symptoms': [
            "Olive-green or brown spots on leaves, fruit, and twigs.",
            "Leaves may become twisted and puckered.",
            "Infected fruit develops dark, scabby spots.",
            "Severe infections can cause premature leaf and fruit drop."
        ],
        'remedies': [
            "Prune and destroy infected twigs and leaves.",
            "Apply fungicides from bud break until midsummer.",
            "Rake up and dispose of fallen leaves in autumn to reduce overwintering spores.",
            "Plant resistant apple varieties."
        ]
    },
    'Apple___Black_rot': {
        'symptoms': [
            "Brown to black, circular spots on leaves.",
            "Cankers on branches, which can girdle and kill them.",
            "Fruit develops a firm, brown to black rot, often starting at the blossom end."
        ],
        'remedies': [
            "Prune out cankered branches and dead wood.",
            "Remove and destroy infected fruit.",
            "Apply fungicides during the growing season.",
            "Maintain good air circulation through proper pruning."
        ]
    },
    'Apple___Cedar_apple_rust': {
        'symptoms': [
            "Bright yellow-orange spots on leaves.",
            "Spots may develop small black dots in the center.",
            "On fruit, raised orange-yellow spots appear.",
            "Galls form on nearby cedar or juniper trees."
        ],
        'remedies': [
            "Remove nearby cedar and juniper trees if possible.",
            "Apply fungicides starting at bud break.",
            "Plant rust-resistant apple varieties.",
            "Prune and destroy galls on cedar trees in late winter."
        ]
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'symptoms': [
            "White, powdery patches on leaves, shoots, and sometimes fruit.",
            "Leaves may become distorted, curled, or stunted.",
            "Infected blossoms may fail to set fruit."
        ],
        'remedies': [
            "Apply fungicides (sulfur, potassium bicarbonate, or neem oil) at the first sign of disease.",
            "Ensure good air circulation by pruning.",
            "Avoid overhead watering to keep foliage dry.",
            "Remove and destroy infected plant parts."
        ]
    },
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': {
        'symptoms': [
            "Small, necrotic spots that elongate into long, rectangular, tan lesions.",
            "Lesions are typically restricted by leaf veins.",
            "Severe infections can cause leaves to blight and die prematurely."
        ],
        'remedies': [
            "Plant resistant corn hybrids.",
            "Use crop rotation with non-host crops.",
            "Manage residue by tilling to reduce fungal survival.",
            "Apply fungicides when disease is first detected."
        ]
    },
    'Corn_(maize)___Common_rust_': {
        'symptoms': [
            "Small, cinnamon-brown, powdery pustules on both upper and lower leaf surfaces.",
            "Pustules can also appear on stalks and husks.",
            "Pustules rupture to release reddish-brown spores."
        ],
        'remedies': [
            "Plant resistant hybrids.",
            "Fungicide applications are effective but often not economically necessary for common rust.",
            "Early planting can help the crop mature before rust becomes severe."
        ]
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'symptoms': [
            "Large, cigar-shaped, grayish-green to tan lesions on leaves.",
            "Lesions can be 1 to 6 inches long.",
            "Severe infection can lead to significant yield loss."
        ],
        'remedies': [
            "Plant resistant hybrids.",
            "Crop rotation and tillage to manage crop residue.",
            "Apply fungicides based on scouting and disease pressure."
        ]
    },
    'Grape___Black_rot': {
        'symptoms': [
            "Small, yellowish spots on leaves that enlarge and turn brown to black.",
            "Fruit develops small, whitish spots that enlarge, turning the entire berry black, hard, and mummified."
        ],
        'remedies': [
            "Apply fungicides starting early in the season and continuing through veraison.",
            "Prune vines to improve air circulation.",
            "Remove and destroy infected canes, leaves, and mummified fruit.",
            "Practice good sanitation in the vineyard."
        ]
    },
    'Grape___Esca_(Black_Measles)': {
        'symptoms': [
            "Leaves show 'tiger-stripe' patterns of chlorosis and necrosis between veins.",
            "Small, dark spots on grape berries, often in a circular pattern.",
            "In chronic form, causes dieback of cordons and trunk."
        ],
        'remedies': [
            "No effective chemical cure. Management focuses on prevention.",
            "Prune out and destroy infected wood.",
            "Protect pruning wounds with a sealant.",
            "Delayed pruning (late winter/early spring) can reduce infection risk."
        ]
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'symptoms': [
            "Dark reddish-brown to black spots on leaves, often with a lighter tan center.",
            "Spots may merge, causing large areas of the leaf to die.",
            "Severe infection can cause premature defoliation."
        ],
        'remedies': [
            "Fungicide sprays used for other grape diseases (like black rot) are usually effective.",
            "Improve air circulation through canopy management.",
            "Rake and destroy fallen leaves to reduce inoculum."
        ]
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'symptoms': [
            "Blotchy, mottled leaves (asymmetrical yellowing).",
            "Stunted growth and twig dieback.",
            "Fruit is small, lopsided, and remains green at the bottom.",
            "Fruit tastes bitter and salty."
        ],
        'remedies': [
            "There is no cure for citrus greening.",
            "Management involves removing and destroying infected trees.",
            "Control the Asian citrus psyllid, the insect that spreads the disease, with insecticides.",
            "Plant certified disease-free trees."
        ]
    },
    'Peach___Bacterial_spot': {
        'symptoms': [
            "Small, water-soaked spots on leaves that turn purple to black.",
            "The center of the spots may fall out, creating a 'shot-hole' appearance.",
            "Fruit develops pitted, cracked, or sunken spots."
        ],
        'remedies': [
            "Plant resistant peach varieties.",
            "Apply bactericides (copper-based sprays) during the dormant season and early in the growing season.",
            "Maintain tree vigor with proper fertilization and watering."
        ]
    },
    'Pepper,_bell___Bacterial_spot': {
        'symptoms': [
            "Small, water-soaked spots on leaves that become dark and greasy.",
            "Spots may have a yellow halo.",
            "Fruit develops raised, scab-like spots.",
            "Severe infection can cause leaf and blossom drop."
        ],
        'remedies': [
            "Use disease-free seeds and transplants.",
            "Rotate crops, avoiding fields where peppers or tomatoes were recently grown.",
            "Apply copper-based bactericides.",
            "Avoid working in fields when plants are wet."
        ]
    },
    'Potato___Early_blight': {
        'symptoms': [
            "Dark, circular to irregular spots on lower leaves, often with a 'target-like' pattern of concentric rings.",
            "A yellow halo may surround the spots.",
            "Tubers can develop dark, sunken lesions."
        ],
        'remedies': [
            "Plant certified disease-free seed potatoes.",
            "Rotate crops with non-susceptible crops.",
            "Apply fungicides preventatively, especially during warm, humid weather.",
            "Destroy volunteer potato plants and weeds."
        ]
    },
    'Potato___Late_blight': {
        'symptoms': [
            "Large, dark, water-soaked spots on leaves, often with a pale green border.",
            "A white, fuzzy mold may appear on the underside of leaves in humid conditions.",
            "Tubers develop a reddish-brown, dry rot."
        ],
        'remedies': [
            "Plant resistant varieties and certified seed potatoes.",
            "Apply fungicides on a regular schedule, especially during cool, wet weather.",
            "Destroy infected plants and cull piles to reduce inoculum.",
            "Ensure good air circulation."
        ]
    },
    'Squash___Powdery_mildew': {
        'symptoms': [
            "White, powdery spots on leaves, stems, and petioles.",
            "Spots can spread to cover entire leaves.",
            "Infected leaves may turn yellow and die.",
            "Fruit may be smaller and of poor quality."
        ],
        'remedies': [
            "Plant resistant varieties.",
            "Apply fungicides (neem oil, sulfur, or potassium bicarbonate) at first sign of disease.",
            "Improve air circulation by spacing plants properly.",
            "Water at the base of the plant to keep foliage dry."
        ]
    },
    'Strawberry___Leaf_scorch': {
        'symptoms': [
            "Irregular, purplish blotches on leaves.",
            "The center of the blotches turns brown, and the tissue dies, giving a 'scorched' appearance.",
            "Petioles and runners can also be affected."
        ],
        'remedies': [
            "Plant resistant varieties.",
            "Renovate strawberry beds after harvest by mowing and removing old leaves.",
            "Apply fungicides if the disease is severe.",
            "Maintain good air circulation and sunlight exposure."
        ]
    },
    'Tomato___Bacterial_spot': {
        'symptoms': [
            "Small, water-soaked, circular spots on leaves and stems that turn greasy and black.",
            "Spots may have a yellow halo.",
            "Fruit develops raised, black, scabby spots."
        ],
        'remedies': [
            "Use certified disease-free seed and transplants.",
            "Rotate crops, avoiding fields where tomatoes or peppers were grown.",
            "Apply copper-based bactericides.",
            "Avoid overhead irrigation."
        ]
    },
    'Tomato___Early_blight': {
        'symptoms': [
            "Dark spots with concentric rings ('target spots') on lower leaves.",
            "A yellow halo often surrounds the spots.",
            "Can cause 'collar rot' on stems near the soil line.",
            "Fruit can develop dark, leathery spots near the stem."
        ],
        'remedies': [
            "Plant resistant varieties.",
            "Stake or cage plants to improve air circulation.",
            "Apply fungicides preventatively.",
            "Mulch around plants to reduce soil splash.",
            "Remove and destroy infected lower leaves."
        ]
    },
    'Tomato___Late_blight': {
        'symptoms': [
            "Large, greasy, gray-green spots on leaves that quickly turn brown.",
            "White, fuzzy mold may appear on the underside of leaves.",
            "Stems develop large black lesions.",
            "Fruit develops large, firm, brown, greasy spots."
        ],
        'remedies': [
            "Plant resistant varieties.",
            "Apply fungicides on a regular schedule, especially in cool, wet weather.",
            "Destroy infected plants immediately to prevent spread.",
            "Ensure good spacing for air circulation."
        ]
    },
    'Tomato___Leaf_Mold': {
        'symptoms': [
            "Pale green or yellowish spots on the upper surface of leaves.",
            "Olive-green to brownish, velvety mold on the underside of leaves corresponding to the spots.",
            "Primarily a greenhouse disease."
        ],
        'remedies': [
            "Use resistant varieties.",
            "Improve air circulation and reduce humidity in greenhouses.",
            "Water at the base of plants to keep foliage dry.",
            "Apply fungicides if necessary."
        ]
    },
    'Tomato___Septoria_leaf_spot': {
        'symptoms': [
            "Many small, circular spots with dark borders and tan or gray centers on lower leaves.",
            "Small black dots (pycnidia) may be visible in the center of the spots.",
            "Leaves turn yellow, wither, and fall off."
        ],
        'remedies': [
            "Remove and destroy infected leaves.",
            "Mulch around plants to prevent soil splash.",
            "Improve air circulation.",
            "Apply fungicides.",
            "Rotate crops."
        ]
    },
    'Tomato___Spider_mites_Two-spotted_spider_mite': {
        'symptoms': [
            "Yellow stippling or tiny dots on leaves.",
            "Fine webbing on the underside of leaves or between stems.",
            "Leaves may turn yellow or bronze and become dry.",
            "Heavy infestations can cause plant death."
        ],
        'remedies': [
            "Spray plants with a strong jet of water to dislodge mites.",
            "Apply insecticidal soap or horticultural oils.",
            "Introduce predatory mites (a form of biological control).",
            "Keep plants well-watered to reduce stress."
        ]
    },
    'Tomato___Target_Spot': {
        'symptoms': [
            "Small, water-soaked spots on leaves that enlarge into lesions with distinct concentric rings ('target spots').",
            "Spots are typically dark brown to black.",
            "Can also affect stems and fruit."
        ],
        'remedies': [
            "Improve air circulation.",
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Rotate crops.",
            "Remove crop debris after harvest."
        ]
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'symptoms': [
            "Severe stunting of the plant.",
            "Upward curling and yellowing of leaves.",
            "Leaves are often smaller than normal.",
            "Reduced fruit set."
        ],
        'remedies': [
            "No cure for the virus.",
            "Control the whitefly vector with insecticides or physical barriers (netting).",
            "Plant resistant varieties.",
            "Remove and destroy infected plants immediately."
        ]
    },
    'Tomato___Tomato_mosaic_virus': {
        'symptoms': [
            "Light and dark green mottling or mosaic pattern on leaves.",
            "Leaves may be curled, malformed, or stunted.",
            "Internal browning of fruit.",
            "Overall plant stunting."
        ],
        'remedies': [
            "No cure for the virus.",
            "Use certified disease-free seed.",
            "Wash hands thoroughly before handling plants, especially after using tobacco products.",
            "Remove and destroy infected plants.",
            "Control weeds that may host the virus."
        ]
    }
}

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fast_plant_disease_model.h5")
classification_model = load_model(MODEL_PATH)

def create_plant_mask(image, method='combined'):
    """
    Create a mask to identify plant regions and exclude background.
    Uses multiple approaches for robustness.
    
    Args:
        image: Input image (BGR format from cv2)
        method: 'hsv', 'grabcut', 'combined'
    
    Returns:
        Binary mask where 1 represents plant regions
    """
    height, width = image.shape[:2]
    
    if method == 'hsv' or method == 'combined':
        # Method 1: HSV color-based segmentation for green vegetation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for green colors (plants/leaves)
        # Lower and upper bounds for green hues
        lower_green1 = np.array([35, 40, 40])   # Light green
        upper_green1 = np.array([85, 255, 255]) # Dark green
        
        # Create mask for green regions
        green_mask = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # Also include yellow-green regions (diseased leaves)
        lower_yellow_green = np.array([25, 40, 40])
        upper_yellow_green = np.array([35, 255, 255])
        yellow_green_mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)
        
        # Combine green and yellow-green masks
        hsv_mask = cv2.bitwise_or(green_mask, yellow_green_mask)
        
        # Include brown/orange regions (diseased/dead plant parts)
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([25, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        hsv_mask = cv2.bitwise_or(hsv_mask, brown_mask)
        
        if method == 'hsv':
            final_mask = hsv_mask
    
    if method == 'grabcut' or method == 'combined':
        # Method 2: GrabCut algorithm for foreground extraction
        mask_grabcut = np.zeros((height, width), np.uint8)
        
        # Define rectangle around the center region (assuming plant is centered)
        margin_x, margin_y = width // 6, height // 6
        rect = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)
        
        # Initialize background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Run GrabCut algorithm
            cv2.grabCut(image, mask_grabcut, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
            
            # Extract probable foreground and definite foreground
            grabcut_mask = np.where((mask_grabcut == 2) | (mask_grabcut == 0), 0, 1).astype('uint8') * 255
            
            if method == 'grabcut':
                final_mask = grabcut_mask
        except:
            print("GrabCut failed, falling back to HSV method")
            if method == 'grabcut':
                # Fallback to HSV if GrabCut fails
                return create_plant_mask(image, method='hsv')
    
    if method == 'combined':
        # Combine both methods
        try:
            # Weight HSV mask more heavily as it's more reliable for plants
            final_mask = cv2.bitwise_or(hsv_mask, grabcut_mask)
            
            # If GrabCut failed, use only HSV
            if 'grabcut_mask' not in locals():
                final_mask = hsv_mask
        except:
            final_mask = hsv_mask
    
    # Post-processing to clean up the mask
    final_mask = post_process_mask(final_mask, image)
    
    return final_mask

def post_process_mask(mask, original_image):
    """
    Clean up the mask using morphological operations and region analysis.
    """
    # Convert to binary if not already
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # Remove small noise
    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # Fill small holes
    kernel_medium = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # Remove very small connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Calculate minimum area threshold (2% of image area)
    min_area = (mask.shape[0] * mask.shape[1]) * 0.02
    
    # Create new mask with only significant components
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            new_mask[labels == i] = 1
    
    # If no significant components found, use a fallback strategy
    if new_mask.sum() == 0:
        print("No significant plant regions detected, using center region as fallback")
        h, w = mask.shape
        center_h, center_w = h // 2, w // 2
        margin_h, margin_w = h // 4, w // 4
        new_mask[center_h-margin_h:center_h+margin_h, center_w-margin_w:center_w+margin_w] = 1
    
    # Smooth the mask edges
    new_mask = cv2.GaussianBlur(new_mask.astype(np.float32), (5, 5), 1.5)
    new_mask = (new_mask > 0.5).astype(np.uint8)
    
    return new_mask

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
        target_layer = model.get_layer(last_conv_layer_name)
        
        grad_model = Model(
            inputs=[model.inputs],
            outputs=[target_layer.output, model.output]
        )
        
        print(f"Grad model created successfully")
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        print(f"Forward pass completed, computing gradients...")
        
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            raise ValueError("Gradients are None - the layer might not be suitable for GradCAM")
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        print(f"GradCAM heatmap generated successfully, shape: {heatmap.shape}")
        
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Error in make_gradcam_heatmap: {e}")
        raise e

def apply_gradcam_to_image_with_mask(original_img, heatmap, plant_mask=None, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Apply GradCAM heatmap overlay to the original image with plant masking.
    """
    original_height, original_width = original_img.shape[:2]
    
    print(f"Original image shape: {original_img.shape}")
    print(f"Heatmap shape before resize: {heatmap.shape}")
    
    # Resize heatmap to match original image
    if heatmap.shape[0] < 32 or heatmap.shape[1] < 32:
        heatmap_resized = cv2.resize(heatmap, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    else:
        heatmap_resized = cv2.resize(heatmap, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    
    print(f"Heatmap shape after resize: {heatmap_resized.shape}")
    
    # Create or use provided plant mask
    if plant_mask is None:
        print("Creating plant mask...")
        plant_mask = create_plant_mask(original_img, method='combined')
        print(f"Plant mask created, plant region coverage: {plant_mask.sum() / plant_mask.size * 100:.1f}%")
    
    # Normalize plant mask to [0, 1]
    plant_mask_norm = plant_mask.astype(np.float32)
    if plant_mask_norm.max() > 1:
        plant_mask_norm = plant_mask_norm / 255.0
    
    # Apply mask to heatmap - only show heatmap in plant regions
    masked_heatmap = heatmap_resized * plant_mask_norm
    
    # Apply Gaussian smoothing for better visual quality
    kernel_size = max(3, min(15, original_width // 32))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    heatmap_smooth = cv2.GaussianBlur(masked_heatmap, (kernel_size, kernel_size), 0)
    
    # Renormalize after masking and smoothing
    if heatmap_smooth.max() > 0:
        heatmap_smooth = heatmap_smooth / heatmap_smooth.max()
    
    # Enhance contrast
    heatmap_enhanced = np.power(heatmap_smooth, 0.8)
    
    # Convert heatmap to RGB using colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_enhanced), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure original image is in RGB
    if len(original_img.shape) == 3:
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = original_img
    
    # Create overlay with masked regions
    threshold = 0.1  # Lower threshold since we're using masking
    activation_mask = (heatmap_enhanced > threshold) & (plant_mask_norm > 0.5)
    
    superimposed_img = original_rgb.copy().astype(np.float32)
    overlay = heatmap_colored.astype(np.float32)
    
    # Apply overlay only in plant regions with adaptive alpha
    for c in range(3):
        adaptive_alpha = alpha * heatmap_enhanced * plant_mask_norm
        superimposed_img[:, :, c] = np.where(
            activation_mask,
            adaptive_alpha * overlay[:, :, c] + (1 - adaptive_alpha) * superimposed_img[:, :, c],
            superimposed_img[:, :, c]
        )
    
    print(f"Final image shape: {superimposed_img.shape}")
    print(f"GradCAM applied to {activation_mask.sum()} pixels ({activation_mask.sum() / activation_mask.size * 100:.1f}% of image)")
    
    return superimposed_img.astype(np.uint8), plant_mask

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
        
        # Get symptoms and remedies
        disease_details = DISEASE_INFO.get(predicted_disease, {
            'symptoms': ["No information available for this disease."],
            'remedies': ["Please consult a local agricultural expert."]
        })

        # Add disease details to response
        response_data.update({
            'symptoms': disease_details['symptoms'],
            'remedies': disease_details['remedies']
        })

        # Generate GradCAM for any prediction with reasonable confidence
        if confidence > 0.3:  # Generate GradCAM for both healthy and diseased plants
            print(f"Generating masked GradCAM for: {predicted_disease}")
            
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
                    
                    # Apply GradCAM overlay with plant masking
                    # Use different color schemes for healthy vs diseased
                    colormap = cv2.COLORMAP_VIRIDIS if is_healthy else cv2.COLORMAP_JET
                    alpha = 0.4 if is_healthy else 0.5
                    
                    gradcam_img, plant_mask = apply_gradcam_to_image_with_mask(
                        original_img, 
                        heatmap, 
                        plant_mask=None,  # Will be generated automatically
                        alpha=alpha, 
                        colormap=colormap
                    )
                    
                    # Convert to PIL and then to base64
                    gradcam_pil = Image.fromarray(gradcam_img)
                    buffered = io.BytesIO()
                    gradcam_pil.save(buffered, format="JPEG", quality=95)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Calculate mask statistics for debugging
                    mask_coverage = plant_mask.sum() / plant_mask.size * 100
                    
                    response_data.update({
                        'gradcam_image': f"data:image/jpeg;base64,{img_base64}",
                        'gradcam_generated': True,
                        'visualization_type': 'masked_gradcam',
                        'last_conv_layer': last_conv_layer_name,
                        'colormap_used': 'viridis' if is_healthy else 'jet',
                        'mask_coverage_percent': round(mask_coverage, 1),
                        'masking_applied': True
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