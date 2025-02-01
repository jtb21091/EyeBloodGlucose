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
from dataclasses import dataclass

# Set logging level to CRITICAL to suppress terminal output.
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Feature Extraction Functions
# (Copied from your training code; replace with real implementations if available)
# ---------------------------
def get_pupil_size(image):
    return np.random.uniform(20, 100)

def get_sclera_redness(image):
    return np.random.uniform(0, 100)

def get_vein_prominence(image):
    return np.random.uniform(0, 10)

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
    edges = cv2.Canny(gray, 50, 150)
    return round(np.sum(edges) / (image.shape[0] * image.shape[1]), 5)

# ---------------------------
# Main Prediction Code
# ---------------------------

@dataclass
class EyeDetection:
    """
    Data class for eye detection results.
    """
    left_eye: Any          # Detected left eye bounding boxes.
    right_eye: Any         # Detected right eye bounding boxes.
    ir_intensity: float    # Mean intensity of the grayscale frame.
    timestamp: datetime
    is_valid: bool         # Whether a valid face was detected.
    eyes_open: bool        # Whether the eyes are considered "open" (or forced open in dark).

class EyeGlucoseMonitor:
    def __init__(self, model_path: str = "eye_glucose_model.pkl"):
        """
        Initialize the EyeGlucoseMonitor.
        
        Args:
            model_path: Path to the trained glucose prediction model.
        """
        self.model_path = model_path
        self.model = self._load_model()
        # Initialize face and eye cascades.
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascades = {
            "left": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml'),
            "right": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
        }
        # Timing and prediction variables.
        self.last_prediction_time = time.time()
        self.last_valid_detection_time = time.time()  # For hysteresis
        self.invalid_detection_threshold = 3.0  # seconds without valid detection before clearing reading
        self.prediction_lock = threading.Lock()
        self.latest_prediction = None
        self.glucose_buffer = deque(maxlen=60)
        # Detection parameters.
        self.MIN_EYE_ASPECT_RATIO = 0.2  # If computed ratio is above this, eyes are open.
        self.MIN_IR_INTENSITY = 30       # Threshold below which conditions are considered "dark."

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

    def detect_face_and_eyes(self, frame: np.ndarray) -> EyeDetection:
        """
        Detect a face and eyes in the frame.
        In dark conditions (low IR intensity), force eyes to be considered "open" even if they appear closed.
        
        Returns:
            An EyeDetection object.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_intensity = np.mean(gray)
        enhanced = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(enhanced, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        # Default: invalid detection.
        detection = EyeDetection([], [], ir_intensity, datetime.now(), False, False)
        
        if len(faces) > 0:
            face = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = face
            face_roi = enhanced[y:y+h, x:x+w]
            left_eyes = self.eye_cascades["left"].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            right_eyes = self.eye_cascades["right"].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            if len(left_eyes) > 0 or len(right_eyes) > 0:
                # Compute a simple eye aspect ratio from detected eyes.
                ear_list = []
                for (ex, ey, ew, eh) in (list(left_eyes) + list(right_eyes)):
                    ear_list.append(eh / ew if ew > 0 else 0)
                avg_ear = np.mean(ear_list) if ear_list else 0
                eyes_open = avg_ear > self.MIN_EYE_ASPECT_RATIO
                # In dark conditions, force eyes_open True.
                if ir_intensity < self.MIN_IR_INTENSITY:
                    eyes_open = True
                detection = EyeDetection(left_eyes, right_eyes, ir_intensity, datetime.now(), True, eyes_open)
        return detection

    def extract_features(self, frame: np.ndarray) -> Dict:
        """
        Extract the features from the frame exactly as in training.
        
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
            "pupil_circularity": np.nan,  # Replace with real computation if available.
            "blink_rate": np.nan,         # Replace with real computation if available.
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
        If model prediction fails, do not update the reading.
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

        # Do not use a fallback dummy value.
        with self.prediction_lock:
            if result is not None:
                self.glucose_buffer.append(result)
                self.latest_prediction = np.mean(self.glucose_buffer)
            # If result is None, do not update latest_prediction.

    def run(self):
        """
        Open the webcam feed, detect the face/eyes, extract features, and overlay the blood glucose estimate.
        Predictions are updated once per second.
        Implements hysteresis: if detection becomes invalid for too long, clear the reading.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            detection = self.detect_face_and_eyes(frame)
            if detection.is_valid:
                self.last_valid_detection_time = current_time
                # Update prediction once per second.
                if current_time - self.last_prediction_time >= 1.0:
                    self.last_prediction_time = current_time
                    features = self.extract_features(frame)
                    threading.Thread(target=self.predict_glucose_async, args=(features,)).start()
            else:
                # If no valid detection for longer than threshold, clear the reading.
                if current_time - self.last_valid_detection_time > self.invalid_detection_threshold:
                    with self.prediction_lock:
                        self.latest_prediction = None

            # Overlay only the blood glucose reading.
            with self.prediction_lock:
                text = f"{self.latest_prediction:.1f} mg/dL" if self.latest_prediction is not None else "No Reading"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("Blood Glucose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = EyeGlucoseMonitor()
    monitor.run()
