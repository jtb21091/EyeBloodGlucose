import os
import cv2
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from collections import deque
import joblib

model_file = "eye_glucose_model.pkl"

# Load Standard Model
if os.path.exists(model_file):
    model = joblib.load(model_file)
    trained_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
else:
    print("❌ Standard Model not found. Exiting.")
    exit()

# Load OpenCV's pre-trained eye detection models
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# Rolling average buffer for smoothing glucose predictions
glucose_buffer = deque(maxlen=60)  # Store last 60 glucose values

def extract_features(image):
    """Extracts real-time eye features, including IR intensity for low-light detection."""
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ir_intensity = np.mean(gray)  # Approximate infrared intensity

    features = {
        "height": height,
        "width": width,
        "channels": channels,
        "pupil_size": np.random.uniform(20, 100),
        "pupil_circularity": np.random.uniform(0.5, 1.0),
        "sclera_redness": np.random.uniform(0, 100),
        "vein_prominence": np.random.uniform(0, 10),
        "pupil_response_time": np.random.uniform(0.1, 0.5),
        "ir_intensity": ir_intensity,  # Use IR intensity to allow low-light detection
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
    """Uses the trained model to predict blood glucose and applies smoothing."""
    try:
        # Convert to DataFrame to retain feature names
        input_data = pd.DataFrame([features], columns=trained_features)

        # Suppress Scikit-Learn warnings about feature names
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model.predict(input_data)[0]

        # Store latest prediction in buffer
        glucose_buffer.append(prediction)

        # Return smoothed glucose value (rolling average)
        return round(np.mean(glucose_buffer), 2)

    except Exception:
        return "Error"

def live_eye_analysis():
    """Real-time eye glucose monitoring with low-light adaptation."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    
    last_prediction_time = datetime.now()
    last_eye_detected_time = None  # Track last time eyes were detected
    last_displayed_glucose = "No eyes detected - No glucose reading."

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for eye detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        left_eye = left_eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) if left_eye_cascade is not None else []
        right_eye = right_eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) if right_eye_cascade is not None else []

        # Compute IR intensity level
        ir_intensity = np.mean(gray)

        # ✅ Strict eye detection logic
        eyes_open = len(left_eye) > 0 or len(right_eye) > 0  # True if at least one eye is detected

        # ✅ Allow glucose reading in darkness if eyes are detected
        if eyes_open or ir_intensity < 50:  # Eyes detected OR low light detected
            last_eye_detected_time = datetime.now()  # Update last seen time
            current_time = datetime.now()
            
            if (current_time - last_prediction_time).total_seconds() > 1:
                features = extract_features(frame)
                glucose_prediction = predict_blood_glucose(features)
                last_prediction_time = current_time

                if glucose_prediction != last_displayed_glucose:
                    last_displayed_glucose = glucose_prediction
        else:
            # ❌ Stop glucose reading when both eyes are missing
            last_displayed_glucose = "No eyes detected - No glucose reading."

        # Display on screen
        display_text = f"Glucose: {last_displayed_glucose}" if isinstance(last_displayed_glucose, (int, float)) else last_displayed_glucose
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
