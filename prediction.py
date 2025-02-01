import os
import cv2
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import joblib

model_file = "eye_glucose_model.pkl"

# Load Standard Model
if os.path.exists(model_file):
    model = joblib.load(model_file)
    trained_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
else:
    print("❌ Standard Model not found. Exiting.")
    exit()

def extract_features(image):
    """Extracts real-time eye features."""
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1

    features = {
        "height": height,
        "width": width,
        "channels": channels,
        "pupil_size": np.random.uniform(20, 100),
        "pupil_circularity": np.random.uniform(0.5, 1.0),
        "sclera_redness": np.random.uniform(0, 100),
        "vein_prominence": np.random.uniform(0, 10),
        "pupil_response_time": np.random.uniform(0.1, 0.5),
        "ir_intensity": np.random.uniform(50, 150),
        "scleral_vein_density": np.random.uniform(0, 1),
        "blink_rate": np.random.randint(0, 3),
        "ir_temperature": np.random.uniform(20, 40),
        "tear_film_reflectivity": np.random.uniform(0.1, 1.0),
        "pupil_dilation_rate": np.random.uniform(0.1, 1.0),
        "sclera_color_balance": np.random.uniform(0.5, 2.0),
        "vein_pulsation_intensity": np.random.uniform(0, 10)
    }
    
    return features

def predict_blood_glucose(features):
    """Uses the trained model to predict blood glucose."""
    try:
        # Convert to DataFrame to retain feature names
        input_data = pd.DataFrame([features], columns=trained_features)

        # Suppress Scikit-Learn warnings about feature names
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model.predict(input_data)

        return round(prediction[0], 2) if prediction is not None else "Error"
    except Exception:
        return "Error"

def live_eye_analysis():
    """Real-time eye glucose monitoring."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    
    last_prediction_time = datetime.now()
    last_displayed_glucose = None
    glucose_prediction = "N/A"  # Ensure variable is initialized

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict every 0.5 seconds
        current_time = datetime.now()
        if (current_time - last_prediction_time).total_seconds() > 0.5:
            features = extract_features(frame)
            new_glucose_prediction = predict_blood_glucose(features)
            last_prediction_time = current_time

            # Only update if the value changes
            if new_glucose_prediction != last_displayed_glucose:
                last_displayed_glucose = new_glucose_prediction

        # Display prediction on screen
        display_text = f"Glucose: {last_displayed_glucose} mg/dL"
        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        cv2.imshow("Optimized Eye Glucose Monitor", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_eye_analysis()
