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

if os.path.exists(model_file):
    model = joblib.load(model_file)
    trained_features = list(model.feature_names_in_)
else:
    model = None
    trained_features = []

def capture_eye_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, None
    
    cv2.waitKey(500)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not capture image.")
        return None, None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eye_{timestamp}.jpg"
    filepath = os.path.join(image_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"Image saved: {filepath}")
    
    return filename, frame

def detect_pupil(image):
    return np.random.uniform(20, 100)  # Placeholder for actual detection

def get_pupil_circularity(image):
    return np.random.uniform(0.5, 1.0)

def get_sclera_redness(image):
    return np.random.uniform(0, 100)

def get_vein_prominence(image):
    return np.random.uniform(0, 10)

def get_pupil_response_time():
    return np.random.uniform(0.1, 0.5)

def get_ir_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.mean(gray), 5)

def get_scleral_vein_density(image):
    return np.random.uniform(0, 1)

def detect_blink():
    return np.random.randint(0, 3)

def get_ir_temperature(image):
    return round(np.mean(image[:, :, 2]), 5)

def get_tear_film_reflectivity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.std(gray), 5)

def get_pupil_dilation_rate():
    return np.random.uniform(0.1, 1.0)

def get_sclera_color_balance(image):
    r_mean = np.mean(image[:, :, 2])
    g_mean = np.mean(image[:, :, 1])
    return round(r_mean / g_mean, 5) if g_mean > 0 else 1.0

def get_vein_pulsation_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.mean(cv2.Laplacian(gray, cv2.CV_64F)), 5)

def get_eye_opacity(image):
    return np.random.uniform(0, 1)  # Simulated placeholder

def get_retinal_reflectivity(image):
    return np.random.uniform(0, 1)  # Simulated placeholder

def extract_features(image):
    features = {
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
        "vein_pulsation_intensity": get_vein_pulsation_intensity(image),
        "eye_opacity": get_eye_opacity(image),
        "retinal_reflectivity": get_retinal_reflectivity(image)
    }
    
    # Replace None values with NaN
    for key in trained_features:
        if key not in features:
            features[key] = np.nan  # Ensure all trained features exist in the current extraction
    
    return features

def predict_blood_glucose(feature_values):
    if model is not None:
        feature_df = pd.DataFrame([feature_values], columns=trained_features)
        
        # Handle missing values with imputation
        imputer = SimpleImputer(strategy='mean')
        feature_df = pd.DataFrame(imputer.fit_transform(feature_df), columns=trained_features)
        
        scaler = StandardScaler()
        feature_df = pd.DataFrame(scaler.fit_transform(feature_df), columns=trained_features)
        
        return round(model.predict(feature_df)[0], 2)
    return "Model not trained"

def live_eye_analysis():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    last_prediction_time = time.time()
    glucose_prediction = "N/A"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break
        
        current_time = time.time()
        if current_time - last_prediction_time > 1:
            features = extract_features(frame)
            glucose_prediction = predict_blood_glucose(features)
            last_prediction_time = current_time
        
        display_text = f"Glucose: {glucose_prediction} mg/dL"
        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Eye Glucose Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_eye_analysis()
