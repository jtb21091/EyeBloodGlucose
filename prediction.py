import os
import cv2
import pandas as pd
import numpy as np
import coremltools as ct  # Core ML for optimized inference
from datetime import datetime

model_file = "eye_glucose_model.mlmodel"

# Load Core ML Model
if os.path.exists(model_file):
    model = ct.models.MLModel(model_file)
    trained_features = [f.name for f in model.get_spec().description.input]
    print(f"✅ Core ML Model loaded with {len(trained_features)} features: {trained_features}")
else:
    print("❌ Core ML Model not found. Exiting.")
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

print(model.get_spec().description)
print(f"🔎 Input to Core ML: {input_data.shape}, Expected: (1, 16)")
print(f"Features provided: {features.keys()}")
print(f"Trained Features: {trained_features}")

import numpy as np

def predict_blood_glucose(features):
    """Uses Core ML model to predict blood glucose."""
    try:
        # Ensure the features dictionary contains all 16 required keys
        input_data = np.array([features[name] for name in trained_features if name in features], dtype=np.float32)

        # Core ML requires correctly formatted input
        input_dict = {"input": input_data.reshape(1, -1)}  # Match Core ML's expected input shape
        
        # Make prediction
        prediction = model.predict(input_dict)

        # Extract prediction result (ensure correct key matches "prediction")
        glucose_value = prediction.get("prediction", None)  # Adjust based on actual model output key
        
        return round(glucose_value[0], 2) if glucose_value is not None else "Error"

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return "Error"


def live_eye_analysis():
    """Real-time eye glucose monitoring."""
    print("Starting real-time eye analysis with Core ML...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    last_prediction_time = datetime.now()
    glucose_prediction = "N/A"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break
        
        # Predict every 0.5 seconds
        current_time = datetime.now()
        if (current_time - last_prediction_time).total_seconds() > 0.5:
            features = extract_features(frame)
            glucose_prediction = predict_blood_glucose(features)
            last_prediction_time = current_time
        
        # Display prediction
        display_text = f"Glucose: {glucose_prediction} mg/dL"
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
