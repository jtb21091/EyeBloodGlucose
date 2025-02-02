import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import threading
import time
from dataclasses import dataclass
import logging

# Set logging to show warnings (and errors) only.
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the expected feature order (must match training, ignoring the first two columns)
FEATURES_ORDER = [
    'pupil_size',
    'sclera_redness',
    'vein_prominence',
    'pupil_response_time',
    'ir_intensity',
    'scleral_vein_density',
    'ir_temperature',
    'tear_film_reflectivity',
    'pupil_dilation_rate',
    'sclera_color_balance',
    'vein_pulsation_intensity',
    'birefringence_index'
]

def get_birefringence_index(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return round(np.var(gray) / 255.0, 5)  # Normalized variance as an index
    except Exception:
        return 0.0  # Fallback value

# ---------------------------
# Feature Extraction Functions
# (Replace these dummy implementations with your real ones if available)
# ---------------------------
def get_pupil_size(image):
    return np.random.uniform(20, 100)

def get_sclera_redness(image):
    return np.random.uniform(0, 100)

def get_vein_prominence(image):
    return np.random.uniform(0, 10)

def get_ir_temperature(image):
    # Compute IR temperature from the mean of the red channel.
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
# DataClass for Detection
# ---------------------------
@dataclass
class EyeDetection:
    left_eye: Any          # Detected left eye bounding boxes.
    right_eye: Any         # Detected right eye bounding boxes.
    ir_intensity: float    # Mean intensity of the grayscale frame.
    timestamp: datetime
    is_valid: bool         # Whether a valid face was detected.
    eyes_open: bool        # Whether the eyes are considered "open" (or forced open in dark).
    face_rect: tuple = None  # The bounding box of the detected face (x, y, w, h)

# ---------------------------
# Main Prediction Code
# ---------------------------
class EyeGlucoseMonitor:
    def __init__(self):
        # Since we're using a fixed equation, there's no need to load a model.
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascades = {
            "left": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml'),
            "right": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
        }
        self.last_valid_detection_time = time.time()
        self.invalid_detection_threshold = 3.0  # Seconds without valid detection before clearing reading
        self.prediction_lock = threading.Lock()
        
        # Instead of a rolling buffer, we'll keep both instantaneous and EMA-smoothed predictions.
        self.latest_smoothed_prediction = None
        self.latest_instantaneous_prediction = None
        self.last_instantaneous_prediction = None  # For computing rate-of-change
        
        # Tolerance for considering predictions "constant" (in mg/dL)
        self.prediction_tolerance = 0.1
        
        # EMA smoothing factor (alpha): closer to 1 => more responsive, closer to 0 => more smoothing.
        self.alpha = 0.005

        self.last_features = None  # Store the most recent feature dictionary

        self.MIN_EYE_ASPECT_RATIO = 0.2  # Threshold for eyes open
        self.MIN_IR_INTENSITY = 30       # Threshold for dark conditions

    def detect_face_and_eyes(self, frame: np.ndarray) -> EyeDetection:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_intensity = np.mean(gray)
        enhanced = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(enhanced, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
        # Try a blurred image if no faces are found under dark conditions.
        if len(faces) == 0 and ir_intensity < self.MIN_IR_INTENSITY:
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            faces = self.face_cascade.detectMultiScale(blurred, scaleFactor=1.1, minNeighbors=2, minSize=(50, 50))
        detection = EyeDetection([], [], ir_intensity, datetime.now(), False, False)
        if len(faces) > 0:
            # Choose the largest face.
            face = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = face
            detection.face_rect = face  # Save the face bounding box.
            face_roi = enhanced[y:y+h, x:x+w]
            left_eyes = self.eye_cascades["left"].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=2, minSize=(15, 15))
            right_eyes = self.eye_cascades["right"].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=2, minSize=(15, 15))
            if ir_intensity < self.MIN_IR_INTENSITY:
                detection = EyeDetection(left_eyes, right_eyes, ir_intensity, datetime.now(), True, True, face)
            else:
                if len(left_eyes) > 0 or len(right_eyes) > 0:
                    ear_list = []
                    for (ex, ey, ew, eh) in (list(left_eyes) + list(right_eyes)):
                        ratio = eh / ew if ew > 0 else 0
                        ear_list.append(ratio)
                    avg_ear = np.mean(ear_list) if ear_list else 0
                    eyes_open = avg_ear > self.MIN_EYE_ASPECT_RATIO
                    detection = EyeDetection(left_eyes, right_eyes, ir_intensity, datetime.now(), True, eyes_open, face)
        return detection

    def extract_features(self, frame: np.ndarray) -> Dict:
        """
        Extract features from the eye region.
        Returns an empty dict if no eyes are detected.
        """
        detection = self.detect_face_and_eyes(frame)
        if not detection.is_valid or (len(detection.left_eye) == 0 and len(detection.right_eye) == 0):
            logging.warning("No valid eyes detected.")
            return {}
        if detection.face_rect is None:
            logging.warning("No face rectangle detected.")
            return {}

        fx, fy, fw, fh = detection.face_rect
        face_roi = frame[fy:fy+fh, fx:fx+fw]

        # Create a union of detected eye boxes (relative to face ROI)
        eye_boxes = []
        for (ex, ey, ew, eh) in detection.left_eye:
            eye_boxes.append((ex, ey, ex+ew, ey+eh))
        for (ex, ey, ew, eh) in detection.right_eye:
            eye_boxes.append((ex, ey, ex+ew, ey+eh))
        if not eye_boxes:
            logging.warning("No eye boxes found.")
            return {}
        ex_min = min(box[0] for box in eye_boxes)
        ey_min = min(box[1] for box in eye_boxes)
        ex_max = max(box[2] for box in eye_boxes)
        ey_max = max(box[3] for box in eye_boxes)
        # Convert from face ROI coordinates to full frame coordinates.
        eye_roi_x = fx + ex_min
        eye_roi_y = fy + ey_min
        eye_roi_w = ex_max - ex_min
        eye_roi_h = ey_max - ey_min

        # Crop the eye region.
        eye_roi = frame[eye_roi_y:eye_roi_y+eye_roi_h, eye_roi_x:eye_roi_x+eye_roi_w]

        features = {
            "pupil_size": get_pupil_size(eye_roi),
            "sclera_redness": get_sclera_redness(eye_roi),
            "vein_prominence": get_vein_prominence(eye_roi),
            "pupil_response_time": get_pupil_response_time(),
            "ir_intensity": get_ir_intensity(eye_roi),
            "scleral_vein_density": get_scleral_vein_density(eye_roi),
            "ir_temperature": get_ir_temperature(eye_roi),
            "tear_film_reflectivity": get_tear_film_reflectivity(eye_roi),
            "pupil_dilation_rate": get_pupil_dilation_rate(),
            "sclera_color_balance": get_sclera_color_balance(eye_roi),
            "vein_pulsation_intensity": get_vein_pulsation_intensity(eye_roi),
            "birefringence_index": get_birefringence_index(eye_roi)
        }
        # Enforce the order expected.
        ordered_features = {key: features.get(key, 0) for key in FEATURES_ORDER}
        # Save the latest features for debugging.
        self.last_features = ordered_features
        return ordered_features

    def predict_glucose(self, features: Dict):
        """
        Predict blood glucose using the provided equation:
        
        Blood Glucose = 123.1617 + (13.1023 × Scleral Vein Density) +
                        (0.1749 × Tear Film Reflectivity) + (3.9700 × Vein Pulsation Intensity) -
                        (0.4187 × Birefringence Index)
        """
        result = None
        try:
            scleral_vein_density = features.get("scleral_vein_density", 0)
            tear_film_reflectivity = features.get("tear_film_reflectivity", 0)
            vein_pulsation_intensity = features.get("vein_pulsation_intensity", 0)
            birefringence_index = features.get("birefringence_index", 0)
            result = (123.1617 +
                      (13.1023 * scleral_vein_density) +
                      (0.1749 * tear_film_reflectivity) +
                      (3.9700 * vein_pulsation_intensity) -
                      (0.4187 * birefringence_index))
        except Exception as e:
            logging.error("Prediction error: " + str(e))
            result = None

        with self.prediction_lock:
            # Save the instantaneous prediction
            if result is not None:
                self.latest_instantaneous_prediction = result
            else:
                self.latest_instantaneous_prediction = None

            # Update EMA (smoothed prediction)
            if result is not None:
                if self.latest_smoothed_prediction is None:
                    self.latest_smoothed_prediction = result
                else:
                    self.latest_smoothed_prediction = self.alpha * result + (1 - self.alpha) * self.latest_smoothed_prediction
            else:
                self.latest_smoothed_prediction = None

            # Compute rate of change (derivative) for the instantaneous prediction.
            if self.last_instantaneous_prediction is not None and result is not None:
                rate_of_change = result - self.last_instantaneous_prediction
                if abs(rate_of_change) > 5:  # Threshold value (mg/dL)
                    logging.info(f"High rate of change detected: {rate_of_change:.2f} mg/dL")
            self.last_instantaneous_prediction = result

            # Save the features for debugging purposes
            self.last_features = features

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = self.detect_face_and_eyes(frame)
            if detection.is_valid and (len(detection.left_eye) > 0 or len(detection.right_eye) > 0):
                features = self.extract_features(frame)
                self.predict_glucose(features)

                # Draw the detected face rectangle (blue)
                if detection.face_rect is not None:
                    (x, y, w, h) = detection.face_rect
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    # Draw left eye boxes (green)
                    for (ex, ey, ew, eh) in detection.left_eye:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                    # Draw right eye boxes (red)
                    for (ex, ey, ew, eh) in detection.right_eye:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)
            else:
                with self.prediction_lock:
                    self.latest_smoothed_prediction = None
                    self.latest_instantaneous_prediction = None

            # Display both instantaneous and smoothed (EMA) predictions.
            with self.prediction_lock:
                inst_text = f"Inst: {self.latest_instantaneous_prediction:.1f} mg/dL" if self.latest_instantaneous_prediction is not None else "Inst: No Reading"
                smooth_text = f"Avg: {self.latest_smoothed_prediction:.1f} mg/dL" if self.latest_smoothed_prediction is not None else "Avg: No Reading"
            cv2.putText(frame, inst_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, smooth_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.imshow("Blood Glucose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = EyeGlucoseMonitor()
    monitor.run()
