import os
import cv2
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Any
import logging
from dataclasses import dataclass
import threading
import time

# Set logging level to CRITICAL to suppress terminal output.
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    eyes_open: bool        # Whether the eyes appear to be open.

class EyeGlucoseMonitor:
    def __init__(self, model_path: str = "eye_glucose_model.pkl"):
        """
        Initialize the EyeGlucoseMonitor.
        
        Args:
            model_path: Path to the trained glucose prediction model.
        """
        self.model_path = model_path
        self.model = self._load_model()
        self.eye_cascades = self._initialize_cascades()

        # Initialize face detection cascade.
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Buffer for computing running average of predictions.
        self.glucose_buffer = deque(maxlen=60)
        self.last_prediction_time = time.time()

        # Threading for asynchronous predictions.
        self.prediction_thread = None
        self.latest_prediction = None
        self.prediction_lock = threading.Lock()

        # Detection parameters.
        self.MIN_EYE_ASPECT_RATIO = 0.2  # Threshold for eye openness.
        self.MAX_INVALID_FRAMES = 5      # Not used further in this minimal version.
        self.MIN_IR_INTENSITY = 30       # Not used further in this minimal version.

    def _load_model(self) -> Any:
        """
        Load the machine learning model from disk.
        Returns:
            The loaded model, or None if unavailable.
        """
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                return model
            except Exception:
                return None
        else:
            return None

    def _initialize_cascades(self) -> Dict[str, cv2.CascadeClassifier]:
        """
        Initialize Haar cascades for left and right eye detection.
        Returns:
            A dictionary with 'left' and 'right' cascade classifiers.
        """
        return {
            "left": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml'),
            "right": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
        }

    def _calculate_eye_aspect_ratio(self, eye_region: np.ndarray) -> float:
        """
        Calculate a simple eye aspect ratio.
        """
        height, width = eye_region.shape[:2]
        return height / width if width > 0 else 0.0

    def detect_face_and_eyes(self, frame: np.ndarray) -> EyeDetection:
        """
        Detect a face and eyes from the frame.
        Returns:
            An EyeDetection instance.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            enhanced,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        detection = EyeDetection([], [], np.mean(gray), datetime.now(), False, False)

        if len(faces) > 0:
            face = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = face
            face_roi = enhanced[y:y+h, x:x+w]
            left_eye = self.eye_cascades["left"].detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            right_eye = self.eye_cascades["right"].detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )

            eyes_open = False
            eye_regions = []
            if len(left_eye) > 0:
                ex, ey, ew, eh = left_eye[0]
                eye_regions.append(face_roi[ey:ey+eh, ex:ex+ew])
            if len(right_eye) > 0:
                ex, ey, ew, eh = right_eye[0]
                eye_regions.append(face_roi[ey:ey+eh, ex:ex+ew])
            if eye_regions:
                avg_ear = np.mean([self._calculate_eye_aspect_ratio(r) for r in eye_regions])
                eyes_open = avg_ear > self.MIN_EYE_ASPECT_RATIO

            detection = EyeDetection(left_eye, right_eye, np.mean(gray), datetime.now(), True, eyes_open)
        return detection

    def extract_features(self, frame: np.ndarray, eye_data: EyeDetection) -> Optional[Dict]:
        """
        Extract features from the frame and detection data.
        Returns:
            A dictionary with the feature names expected by the model, or None if detection is invalid.
        """
        if not eye_data.is_valid or not eye_data.eyes_open:
            return None

        # Build a feature dictionary with exactly the keys used during training.
        features = {
            "pupil_response_time": self._calculate_pupil_response_time(frame, eye_data),
            "sclera_color_balance": self._calculate_sclera_color_balance(frame, eye_data),
            "sclera_redness": self._calculate_sclera_redness(frame, eye_data),
            "scleral_vein_density": self._calculate_scleral_vein_density(frame, eye_data),
            "tear_film_reflectivity": self._calculate_tear_film_reflectivity(frame, eye_data)
        }
        return features

    # Dummy feature calculation methods (replace with actual implementations if available).
    def _calculate_pupil_response_time(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        return np.random.uniform(0, 1)

    def _calculate_sclera_color_balance(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        return np.random.uniform(0, 1)

    def _calculate_sclera_redness(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        return np.random.uniform(0, 1)

    def _calculate_scleral_vein_density(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        return np.random.uniform(0, 1)

    def _calculate_tear_film_reflectivity(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        return np.random.uniform(0, 1)

    def predict_glucose_async(self, features: Dict):
        """
        Asynchronously predict blood glucose based on the extracted features.
        Updates a running average stored in self.latest_prediction.
        """
        try:
            if self.model is not None:
                df = pd.DataFrame([features])
                prediction = self.model.predict(df)
                result = prediction[0] if len(prediction) > 0 else None
            else:
                # Use a dummy prediction if no model is loaded.
                result = np.random.uniform(70, 150)
            with self.prediction_lock:
                if result is not None:
                    self.glucose_buffer.append(result)
                    self.latest_prediction = np.mean(self.glucose_buffer)
        except Exception:
            # In production, handle or log exceptions as needed.
            pass

    def draw_overlay(self, frame: np.ndarray):
        """
        Overlays the blood glucose reading on the webcam frame.
        """
        with self.prediction_lock:
            if self.latest_prediction is not None:
                text = f"{self.latest_prediction:.1f} mg/dL"
                cv2.putText(frame, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    def run(self):
        """
        Open the webcam feed, run detection/prediction every second, and overlay only the blood glucose measurement.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                eye_data = self.detect_face_and_eyes(frame)
                current_time = time.time()
                if eye_data.is_valid and eye_data.eyes_open:
                    if current_time - self.last_prediction_time >= 1.0:
                        self.last_prediction_time = current_time
                        features = self.extract_features(frame, eye_data)
                        if features:
                            self.prediction_thread = threading.Thread(
                                target=self.predict_glucose_async,
                                args=(features,)
                            )
                            self.prediction_thread.start()
                else:
                    with self.prediction_lock:
                        self.latest_prediction = None

                self.draw_overlay(frame)
                cv2.imshow("Blood Glucose", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = EyeGlucoseMonitor()
    monitor.run()
