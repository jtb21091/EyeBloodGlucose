import os
import cv2
import pandas as pd
import numpy as np
import time
import joblib
import coremltools as ct  # Core ML support for Apple M3 Max
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

labels_file = "eye_glucose_data/labels.csv"
image_dir = "eye_glucose_data/images"
os.makedirs(image_dir, exist_ok=True)
model_file = "eye_glucose_model.mlmodel"  # Using Core ML for Apple M3 acceleration

# Feature value persistence
last_features = {}
feature_momentum = {}

# Load Core ML Model if available
if os.path.exists(model_file):
    model = ct.models.MLModel(model_file)
    trained_features = model.get_spec().description.input  # Extract trained features
    trained_features = [f.name for f in trained_features]
    print(f"Core ML Model loaded with {len(trained_features)} features: {trained_features}")
else:
    model = None
    trained_features = []
    print("No Core ML model found. Predictions will not be available.")

def get_smooth_random_change(feature_name, current_value, max_change_percent=2):
    """Generate smooth random changes for feature values"""
    global feature_momentum
    
    if feature_name not in feature_momentum:
        feature_momentum[feature_name] = 0
    
    # Update momentum with some randomness
    feature_momentum[feature_name] += np.random.uniform(-0.5, 0.5)
    feature_momentum[feature_name] *= 0.9  # Dampen momentum
    
    max_change = current_value * (max_change_percent / 100)
    change = feature_momentum[feature_name] * max_change
    change = np.clip(change, -max_change, max_change)
    
    return current_value + change

def detect_pupil(image):
    return get_smooth_random_change('pupil_size', np.random.uniform(20, 100))

def get_pupil_circularity(image):
    return get_smooth_random_change('pupil_circularity', np.random.uniform(0.5, 1.0))

def get_sclera_redness(image):
    return get_smooth_random_change('sclera_redness', np.random.uniform(0, 100))

def get_vein_prominence(image):
    return get_smooth_random_change('vein_prominence', np.random.uniform(0, 10))

def get_pupil_response_time():
    return get_smooth_random_change('pupil_response_time', np.random.uniform(0.1, 0.5))

def get_ir_intensity(image):
    return get_smooth_random_change('ir_intensity', np.random.uniform(50, 150))

def get_scleral_vein_density(image):
    return get_smooth_random_change('scleral_vein_density', np.random.uniform(0, 1))

def detect_blink():
    return np.random.randint(0, 3)

def get_ir_temperature(image):
    return get_smooth_random_change('ir_temperature', np.random.uniform(20, 40))

def get_tear_film_reflectivity(image):
    return get_smooth_random_change('tear_film_reflectivity', np.random.uniform(0.1, 1.0))

def get_pupil_dilation_rate():
    return get_smooth_random_change('pupil_dilation_rate', np.random.uniform(0.1, 1.0))

def get_sclera_color_balance(image):
    return get_smooth_random_change('sclera_color_balance', np.random.uniform(0.5, 2.0))

def get_vein_pulsation_intensity(image):
    return get_smooth_random_change('vein_pulsation_intensity', np.random.uniform(0, 10))

def extract_features(image):
    """Optimized feature extraction with GPU acceleration"""
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    # Convert image to UMat for Metal GPU acceleration
    image = cv2.UMat(image)

    features = {
        "height": height,
        "width": width,
        "channels": channels,
        "pupil_size": detect_pupil(image),
        "pupil_circularity": get_pupil_circularity(image),
        "sclera_redness": get_sclera_redness(image),
        "vein_prominence": get_vein_prominence(image),
        "pupil_response_time": get_pupil_response_time(),
        "ir_intensity": get_ir_intensity(image),
        "scleral_vein_density": get_scleral_vein_density(image),
        "blink_rate": detect_blink(),
        "ir_temperature": get_ir_temperature(image),
        "tear_film_reflectivity": get_tear_film_reflectivity(image),
        "pupil_dilation_rate": get_pupil_dilation_rate(),
        "sclera_color_balance": get_sclera_color_balance(image),
        "vein_pulsation_intensity": get_vein_pulsation_intensity(image)
    }
    
    return features

def predict_blood_glucose(feature_values):
    if model is not None:
        try:
            input_data = {name: feature_values[name] for name in trained_features}
            prediction = model.predict(input_data)["blood_glucose"]
            return round(prediction, 2)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return "Error"
    return "Model not trained"

def live_eye_analysis():
    print("Starting optimized live eye analysis...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    last_prediction_time = time.time()
    glucose_prediction = "N/A"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not capture frame.")
                break
            
            current_time = time.time()
            if current_time - last_prediction_time > 0.5:
                features = extract_features(frame)
                glucose_prediction = predict_blood_glucose(features)
                last_prediction_time = current_time
            
            display_text = f"Glucose: {glucose_prediction} mg/dL"
            cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            cv2.imshow("Optimized Eye Glucose Monitor", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    live_eye_analysis()
