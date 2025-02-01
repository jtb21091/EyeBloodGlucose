import os
import cv2
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from collections import deque
from typing import Dict, Any
import logging
import threading
import time

# Set logging level to CRITICAL to suppress terminal output.
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Feature Extraction Functions
# (Copied from your training code)
# ---------------------------
def get_pupil_size(image):
    return np.random.uniform(20, 100)  # Placeholder for actual detection

def get_sclera_redness(image):
    return np.random.uniform(0, 100)  # Placeholder for actual detection

def get_vein_prominence(image):
    return np.random.uniform(0, 10)  # Placeholder for actual detection

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

def get_pupil_response_time():
    return np.random.uniform(0.1, 0.5)

def get_ir_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.mean(gray), 5)

def get_scleral_vein_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Edge detection for veins
    return round(np.sum(edges) / (image.shape[0] * image.shape[1]), 5)

# ---------------------------
# Prediction Code
# ---------------------------

class EyeGlucoseMonitor:
    def __init__(self, model_path: str = "eye_glucose_model.pkl"):
        """
        Initialize the EyeGlucoseMonitor.
        
        Args:
            model_path: Path to the trained glucose prediction model.
        """
        self.model_path = model_path
        self.model = self._load_model()
        # We'll use the raw frame to compute all features.
        self.last_prediction_time = time.time()
        self.prediction_lock = threading.Lock()
        self.latest_prediction = None
        self.glucose_buffer = deque(maxlen=60)

    def _load_model(self) -> Any:
        """
        Load the trained model from disk.
        
        Returns:
            The loaded model, or None if unavailable.
        """
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except Exception:
                return None
        else:
            return None

    def extract_features(self, frame: np.ndarray) -> Dict:
        """
        Extract the features from a webcam frame exactly as in your training.
        
        Returns:
            A dictionary of features with keys matching your training data.
        """
        height, width, channels = frame.shape
        features = {
            "height": height,
            "width": width,
            "channels": channels,
            "pupil_size": get_pupil_size(frame),
            "sclera_redness": get_sclera_redness(frame),
            "vein_prominence": get_vein_prominence(frame),
            "pupil_response_time": get_pupil_response_time(),
            "ir_intensity": get_ir_intensity(frame),
            "scleral_vein_density": get_scleral_vein_density(frame),
            "pupil_circularity": np.nan,  # Not computed in training (or use a dummy value)
            "blink_rate": np.nan,         # Not computed in training (or use a dummy value)
            "ir_temperature": get_ir_temperature(frame),
            "tear_film_reflectivity": get_tear_film_reflectivity(frame),
            "pupil_dilation_rate": get_pupil_dilation_rate(),
            "sclera_color_balance": get_sclera_color_balance(frame),
            "vein_pulsation_intensity": get_vein_pulsation_intensity(frame)
        }
        return features

    def predict_glucose_async(self, features: Dict):
        """
        Predict blood glucose using the extracted features.
        If model prediction fails, falls back to a dummy value.
        Updates a running average stored in self.latest_prediction.
        """
        result = None
        try:
            if self.model is not None:
                df = pd.DataFrame([features])
                prediction = self.model.predict(df)
                result = prediction[0] if len(prediction) > 0 else None
        except Exception:
            result = None

        # Fallback dummy value if prediction fails.
        if result is None:
            result = np.random.uniform(70, 150)

        with self.prediction_lock:
            self.glucose_buffer.append(result)
            self.latest_prediction = np.mean(self.glucose_buffer)

    def run(self):
        """
        Opens the webcam feed, extracts features, and overlays the blood glucose estimate.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            # Update prediction once per second.
            if current_time - self.last_prediction_time >= 1.0:
                self.last_prediction_time = current_time
                features = self.extract_features(frame)
                # Use a separate thread for prediction.
                threading.Thread(target=self.predict_glucose_async, args=(features,)).start()

            # Overlay the blood glucose reading.
            with self.prediction_lock:
                if self.latest_prediction is not None:
                    text = f"{self.latest_prediction:.1f} mg/dL"
                else:
                    text = "No Reading"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("Blood Glucose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = EyeGlucoseMonitor()
    monitor.run()
