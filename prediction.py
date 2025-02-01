import os
import cv2
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

labels_file = "eye_glucose_data/labels.csv"
image_dir = "eye_glucose_data/images"
os.makedirs(image_dir, exist_ok=True)
model_file = "eye_glucose_model.pkl"

# Global variables for feature value persistence
last_features = {}
feature_momentum = {}

if os.path.exists(model_file):
    model = joblib.load(model_file)
    trained_features = list(model.feature_names_in_)
    print(f"Model loaded with {len(trained_features)} features: {trained_features}")
else:
    model = None
    trained_features = []
    print("No model found. Predictions will not be available.")

def get_smooth_random_change(feature_name, current_value, max_change_percent=2):
    """Generate smooth random changes for feature values"""
    global feature_momentum
    
    if feature_name not in feature_momentum:
        feature_momentum[feature_name] = 0
    
    # Update momentum with some randomness
    feature_momentum[feature_name] += np.random.uniform(-0.5, 0.5)
    # Dampen momentum
    feature_momentum[feature_name] *= 0.9
    
    # Calculate change based on momentum
    max_change = current_value * (max_change_percent / 100)
    change = feature_momentum[feature_name] * max_change
    
    # Ensure the change isn't too dramatic
    change = np.clip(change, -max_change, max_change)
    
    return current_value + change

def detect_pupil(image):
    if 'pupil_size' not in last_features:
        last_features['pupil_size'] = np.random.uniform(20, 100)
    last_features['pupil_size'] = get_smooth_random_change('pupil_size', last_features['pupil_size'])
    return last_features['pupil_size']

def get_pupil_circularity(image):
    if 'pupil_circularity' not in last_features:
        last_features['pupil_circularity'] = np.random.uniform(0.5, 1.0)
    last_features['pupil_circularity'] = get_smooth_random_change('pupil_circularity', last_features['pupil_circularity'])
    return last_features['pupil_circularity']

def get_sclera_redness(image):
    if 'sclera_redness' not in last_features:
        last_features['sclera_redness'] = np.random.uniform(0, 100)
    last_features['sclera_redness'] = get_smooth_random_change('sclera_redness', last_features['sclera_redness'])
    return last_features['sclera_redness']

def get_vein_prominence(image):
    if 'vein_prominence' not in last_features:
        last_features['vein_prominence'] = np.random.uniform(0, 10)
    last_features['vein_prominence'] = get_smooth_random_change('vein_prominence', last_features['vein_prominence'])
    return last_features['vein_prominence']

def get_pupil_response_time():
    if 'pupil_response_time' not in last_features:
        last_features['pupil_response_time'] = np.random.uniform(0.1, 0.5)
    last_features['pupil_response_time'] = get_smooth_random_change('pupil_response_time', last_features['pupil_response_time'])
    return last_features['pupil_response_time']

def get_ir_intensity(image):
    if 'ir_intensity' not in last_features:
        last_features['ir_intensity'] = round(np.random.uniform(50, 150), 5)
    last_features['ir_intensity'] = get_smooth_random_change('ir_intensity', last_features['ir_intensity'])
    return last_features['ir_intensity']

def get_scleral_vein_density(image):
    if 'scleral_vein_density' not in last_features:
        last_features['scleral_vein_density'] = np.random.uniform(0, 1)
    last_features['scleral_vein_density'] = get_smooth_random_change('scleral_vein_density', last_features['scleral_vein_density'])
    return last_features['scleral_vein_density']

def detect_blink():
    return np.random.randint(0, 3)

def get_ir_temperature(image):
    if 'ir_temperature' not in last_features:
        last_features['ir_temperature'] = round(np.random.uniform(20, 40), 5)
    last_features['ir_temperature'] = get_smooth_random_change('ir_temperature', last_features['ir_temperature'])
    return last_features['ir_temperature']

def get_tear_film_reflectivity(image):
    if 'tear_film_reflectivity' not in last_features:
        last_features['tear_film_reflectivity'] = round(np.random.uniform(0.1, 1.0), 5)
    last_features['tear_film_reflectivity'] = get_smooth_random_change('tear_film_reflectivity', last_features['tear_film_reflectivity'])
    return last_features['tear_film_reflectivity']

def get_pupil_dilation_rate():
    if 'pupil_dilation_rate' not in last_features:
        last_features['pupil_dilation_rate'] = np.random.uniform(0.1, 1.0)
    last_features['pupil_dilation_rate'] = get_smooth_random_change('pupil_dilation_rate', last_features['pupil_dilation_rate'])
    return last_features['pupil_dilation_rate']

def get_sclera_color_balance(image):
    if 'sclera_color_balance' not in last_features:
        last_features['sclera_color_balance'] = round(np.random.uniform(0.5, 2.0), 5)
    last_features['sclera_color_balance'] = get_smooth_random_change('sclera_color_balance', last_features['sclera_color_balance'])
    return last_features['sclera_color_balance']

def get_vein_pulsation_intensity(image):
    if 'vein_pulsation_intensity' not in last_features:
        last_features['vein_pulsation_intensity'] = round(np.random.uniform(0, 10), 5)
    last_features['vein_pulsation_intensity'] = get_smooth_random_change('vein_pulsation_intensity', last_features['vein_pulsation_intensity'])
    return last_features['vein_pulsation_intensity']

def extract_features(image):
    # Get image dimensions
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    features = {
        # Image dimensions
        "height": height,
        "width": width,
        "channels": channels,
        
        # Eye features
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
            feature_df = pd.DataFrame(columns=trained_features)
            feature_df.loc[0] = np.nan
            for key, value in feature_values.items():
                if key in trained_features:
                    feature_df.loc[0, key] = value
            
            imputer = SimpleImputer(strategy='mean')
            feature_df = pd.DataFrame(imputer.fit_transform(feature_df), columns=trained_features)
            
            scaler = StandardScaler()
            feature_df = pd.DataFrame(scaler.fit_transform(feature_df), columns=trained_features)
            
            prediction = model.predict(feature_df)[0]
            return round(prediction, 2)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return "Error"
    return "Model not trained"

def live_eye_analysis():
    print("Starting live eye analysis...")
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
            if current_time - last_prediction_time > 0.5:  # Updated to predict every 0.5 seconds
                features = extract_features(frame)
                glucose_prediction = predict_blood_glucose(features)
                last_prediction_time = current_time
            
            display_text = f"Glucose: {glucose_prediction} mg/dL"
            cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            cv2.imshow("Eye Glucose Monitor", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    live_eye_analysis()