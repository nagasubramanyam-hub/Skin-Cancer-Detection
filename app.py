from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import cv2
import gdown
import base64
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Configuration
MODEL_DIR = './model_files'
MODEL_PATH = os.path.join(MODEL_DIR, 'final_model.keras')
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper functions for checking file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Utility functions adapted from your Streamlit app
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def generate_gradcam(model, img_array, layer_name='conv5_block3_out'):
    """Generate Grad-CAM visualization"""
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    return heatmap

def apply_heatmap(image, heatmap, alpha=0.6):
    """Apply heatmap overlay on image"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    
    # Apply colormap
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Overlay heatmap on image
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlay

def enhance_image(image):
    """Apply image enhancement techniques"""
    from PIL import ImageEnhance
    
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(1.3)
    return enhanced

def image_to_base64(image, format="JPEG"):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def array_to_base64(array, format="JPEG"):
    """Convert numpy array to base64 string"""
    image = Image.fromarray(array.astype('uint8'))
    return image_to_base64(image, format)

# Model loading function
def load_detection_model():
    # Check if model file exists
    model_file_exists = os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10000
    
    if not model_file_exists:
        print("Model files not found. Downloading model...")
        try:
            file_id = '1OOefzvwsvstYOXpt325S-py2Vc5KPKM4'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, MODEL_PATH, quiet=False)
            
            if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10000:
                print(f"Downloaded model: {os.path.getsize(MODEL_PATH)} bytes")
            else:
                print("Download failed or file is too small.")
                return create_fallback_model()
        except Exception as e:
            print(f"Download error: {str(e)}")
            return create_fallback_model()
    
    # Attempt to load model
    try:
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10000:
            model = load_model(MODEL_PATH, compile=False)
            return model
        else:
            return create_fallback_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return create_fallback_model()

def create_fallback_model():
    """Create a basic model as fallback"""
    print("Creating a simplified model for demonstration purposes.")
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv5_block3_out')(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    print("Using fallback model - predictions will not be accurate")
    return model

# Load model at startup
print("Loading model...")
model = load_detection_model()
print("Model loaded successfully!")

@app.route('/')
def index():
    return jsonify({"status": "API is running"})

@app.route('/api/detect', methods=['POST'])
def detect():
    print("==== Request received on /api/detect ====")
    print("Method:", request.method)
    print("Content-Type:", request.headers.get('Content-Type'))
    print("Has file?", 'file' in request.files)
    print("Form data:", request.form)
    # Check if file part exists
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    try:
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Check for enhancement option
        use_enhancement = request.form.get('use_enhancement', 'false').lower() == 'true'
        
        if use_enhancement:
            enhanced_image = enhance_image(image)
        else:
            enhanced_image = image
        
        # Preprocess image for model
        img_array = preprocess_image(enhanced_image)
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Get class prediction and confidence
        class_names = ['Benign', 'Malignant']
        predicted_class = np.argmax(prediction[0])
        predicted_label = class_names[predicted_class]
        confidence = float(prediction[0][predicted_class]) * 100
        
        # Generate Grad-CAM
        try:
            heatmap = generate_gradcam(model, img_array)
            heatmap_img = apply_heatmap(enhanced_image, heatmap)
            heatmap_base64 = array_to_base64(heatmap_img)
        except Exception as e:
            print(f"Could not generate attention map: {str(e)}")
            heatmap_base64 = None
        
        # Generate original image base64
        original_base64 = image_to_base64(image)
        
        # Create probability chart
        fig, ax = plt.subplots(figsize=(5, 2))
        bars = ax.barh(['Benign', 'Malignant'], prediction[0] * 100, color=['green', 'red'])
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probability (%)')
        ax.bar_label(bars, fmt='%.1f%%')
        
        # Save chart to base64
        chart_buffer = io.BytesIO()
        plt.savefig(chart_buffer, format='png', bbox_inches='tight')
        chart_buffer.seek(0)
        chart_base64 = base64.b64encode(chart_buffer.getvalue()).decode()
        plt.close(fig)
        
        # Return results
        return jsonify({
            "prediction": {
                "label": predicted_label,
                "confidence": confidence,
                "probabilities": {
                    "benign": float(prediction[0][0]) * 100,
                    "malignant": float(prediction[0][1]) * 100
                }
            },
            "images": {
                "original": original_base64,
                "heatmap": heatmap_base64,
                "chart": chart_base64
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join('../skin-cancer-web-app/frontend/dist', path)):
        return send_from_directory('../skin-cancer-web-app/frontend/dist', path)
    else:
        return send_from_directory('../skin-cancer-web-app/frontend/dist', 'index.html')
# At the bottom of app.py, update:
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)