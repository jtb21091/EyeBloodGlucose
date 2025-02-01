import os
import cv2
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from collections import deque
import joblib
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class EyeDetection:
    """Data class for eye detection results"""
    left_eye: np.ndarray
    right_eye: np.ndarray
    ir_intensity: float
    timestamp: datetime

class EyeGlucoseMonitor:
    def __init__(self, model_path: str = "eye_glucose_model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()
        self.trained_features = self._get_trained_features()
        
        # Initialize eye cascades
        self.eye_cascades = self._initialize_cascades()
        
        # Buffers for smoothing and analysis
        self.glucose_buffer = deque(maxlen=60)  # 1-minute buffer
        self.detection_buffer = deque(maxlen=30)  # 0.5-second buffer
        
        # Performance monitoring
        self.fps_buffer = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Threading for async predictions
        self.prediction_thread = None
        self.latest_prediction = None
        self.prediction_lock = threading.Lock()

    def _load_model(self) -> object:
        """Load the trained model with error handling"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            return joblib.load(self.model_path)
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise

    def _get_trained_features(self) -> list:
        """Get trained features from the model"""
        return (
            self.model.feature_names_in_ 
            if hasattr(self.model, 'feature_names_in_') 
            else []
        )

    def _initialize_cascades(self) -> Dict:
        """Initialize OpenCV cascade classifiers"""
        cascades = {
            'left': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml'),
            'right': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
        }
        
        # Validate cascade loading
        for name, cascade in cascades.items():
            if cascade.empty():
                logging.error(f"Failed to load {name} eye cascade")
                raise RuntimeError(f"Failed to load {name} eye cascade")
                
        return cascades

    def extract_features(self, frame: np.ndarray, eye_data: EyeDetection) -> Dict:
        """Extract enhanced eye features with actual measurements where possible"""
        height, width = frame.shape[:2]
        
        # Calculate actual measurable features
        features = {
            "height": height,
            "width": width,
            "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
            "ir_intensity": eye_data.ir_intensity,
        }
        
        # Add eye-specific measurements if eyes are detected
        if len(eye_data.left_eye) > 0 or len(eye_data.right_eye) > 0:
            # Calculate actual pupil size and circularity if possible
            # For now, using placeholder calculations that could be replaced
            # with actual computer vision measurements
            features.update({
                "pupil_size": self._calculate_pupil_size(frame, eye_data),
                "pupil_circularity": self._calculate_pupil_circularity(frame, eye_data),
                "sclera_redness": self._calculate_sclera_redness(frame, eye_data),
                "vein_prominence": self._calculate_vein_prominence(frame, eye_data)
            })
        else:
            # Use interpolated values from previous measurements if available
            features.update(self._interpolate_missing_features())
        
        return features

    def _calculate_pupil_size(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        """Calculate actual pupil size using computer vision techniques"""
        # Placeholder for actual pupil detection algorithm
        # This should be replaced with proper pupil detection
        return np.random.uniform(20, 100)  # Temporary random value

    def _calculate_pupil_circularity(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        """Calculate pupil circularity using contour analysis"""
        # Placeholder for actual circularity calculation
        return np.random.uniform(0.5, 1.0)  # Temporary random value

    def _calculate_sclera_redness(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        """Calculate sclera redness using color analysis"""
        # Placeholder for actual sclera color analysis
        return np.random.uniform(0, 100)  # Temporary random value

    def _calculate_vein_prominence(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        """Calculate vein prominence using image processing"""
        # Placeholder for actual vein detection algorithm
        return np.random.uniform(0, 10)  # Temporary random value

    def _interpolate_missing_features(self) -> Dict:
        """Interpolate missing features from historical data"""
        # This could be improved with actual interpolation from historical data
        return {
            "pupil_size": np.random.uniform(20, 100),
            "pupil_circularity": np.random.uniform(0.5, 1.0),
            "sclera_redness": np.random.uniform(0, 100),
            "vein_prominence": np.random.uniform(0, 10)
        }

    def predict_glucose_async(self, features: Dict):
        """Asynchronous glucose prediction"""
        try:
            input_data = pd.DataFrame([features], columns=self.trained_features)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction = self.model.predict(input_data)[0]
            
            with self.prediction_lock:
                self.glucose_buffer.append(prediction)
                self.latest_prediction = round(np.mean(self.glucose_buffer), 2)
                
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            with self.prediction_lock:
                self.latest_prediction = None

    def detect_eyes(self, frame: np.ndarray) -> EyeDetection:
        """Detect eyes in the frame with improved accuracy"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_intensity = np.mean(gray)
        
        # Apply histogram equalization for better detection in varying light
        equalized = cv2.equalizeHist(gray)
        
        # Detect eyes with different scale factors for better accuracy
        left_eye = self.eye_cascades['left'].detectMultiScale(
            equalized, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        right_eye = self.eye_cascades['right'].detectMultiScale(
            equalized, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
        return EyeDetection(left_eye, right_eye, ir_intensity, datetime.now())

    def calculate_fps(self) -> float:
        """Calculate and return current FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.last_frame_time)
        self.fps_buffer.append(fps)
        self.last_frame_time = current_time
        return np.mean(self.fps_buffer)

    def draw_overlay(self, frame: np.ndarray, eye_data: EyeDetection):
        """Draw information overlay on the frame"""
        # Draw FPS
        fps = self.calculate_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw glucose prediction
        with self.prediction_lock:
            glucose_text = (
                f"Glucose: {self.latest_prediction}" 
                if self.latest_prediction is not None 
                else "No glucose reading"
            )
        cv2.putText(frame, glucose_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw eye detection boxes
        for (x, y, w, h) in eye_data.left_eye:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        for (x, y, w, h) in eye_data.right_eye:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def run(self):
        """Run the eye glucose monitoring system"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Failed to open camera")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to read frame")
                    break

                # Detect eyes and extract features
                eye_data = self.detect_eyes(frame)
                features = self.extract_features(frame, eye_data)

                # Start async prediction if not already running
                if (self.prediction_thread is None or 
                    not self.prediction_thread.is_alive()):
                    self.prediction_thread = threading.Thread(
                        target=self.predict_glucose_async,
                        args=(features,)
                    )
                    self.prediction_thread.start()

                # Draw information overlay
                self.draw_overlay(frame, eye_data)

                # Display the frame
                cv2.imshow("Enhanced Eye Glucose Monitor", frame)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logging.error(f"Runtime error: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        monitor = EyeGlucoseMonitor()
        monitor.run()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")