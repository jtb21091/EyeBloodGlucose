import os
import cv2
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
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
    is_valid: bool  # Indicates if the detection meets our criteria
    eyes_open: bool  # Specifically tracks if eyes are open

class EyeGlucoseMonitor:
    def __init__(self, model_path: str = "eye_glucose_model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()
        self.trained_features = self._get_trained_features()
        
        # Initialize eye cascades
        self.eye_cascades = self._initialize_cascades()
        
        # Initialize face cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Buffers for smoothing and analysis
        self.glucose_buffer = deque(maxlen=60)
        self.detection_buffer = deque(maxlen=30)
        self.fps_buffer = deque(maxlen=30)
        
        # State tracking
        self.last_frame_time = time.time()
        self.last_valid_reading_time = None
        self.consecutive_invalid_frames = 0
        self.MIN_IR_INTENSITY = 30  # Minimum IR intensity for dark conditions
        
        # Threading for async predictions
        self.prediction_thread = None
        self.latest_prediction = None
        self.prediction_lock = threading.Lock()
        
        # Detection parameters
        self.MIN_EYE_ASPECT_RATIO = 0.2  # For detecting closed eyes
        self.MAX_INVALID_FRAMES = 5  # Number of frames before considering eyes closed

    def _calculate_eye_aspect_ratio(self, eye_region) -> float:
        """Calculate the eye aspect ratio to determine if eyes are open"""
        # This is a simplified version - in practice you'd want to use facial landmarks
        # for more accurate eye openness detection
        height, width = eye_region.shape[:2]
        return height / width if width > 0 else 0

    def detect_face_and_eyes(self, frame: np.ndarray) -> EyeDetection:
        """Enhanced detection of face and eyes with IR support"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_intensity = np.mean(gray)
        
        # Enhance frame for better detection in low light
        enhanced_frame = cv2.equalizeHist(gray)
        
        # Detect face first
        faces = self.face_cascade.detectMultiScale(
            enhanced_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )
        
        if len(faces) == 0 and ir_intensity < self.MIN_IR_INTENSITY:
            # In very dark conditions, rely more on IR detection
            enhanced_frame = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)
            faces = self.face_cascade.detectMultiScale(
                enhanced_frame,
                scaleFactor=1.1,
                minNeighbors=3,  # More lenient in dark conditions
                minSize=(80, 80)
            )
        
        # Initialize detection result
        detection = EyeDetection(
            left_eye=np.array([]),
            right_eye=np.array([]),
            ir_intensity=ir_intensity,
            timestamp=datetime.now(),
            is_valid=False,
            eyes_open=False
        )
        
        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Define regions of interest for eyes
            face_roi = enhanced_frame[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            left_eye = self.eye_cascades['left'].detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            right_eye = self.eye_cascades['right'].detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            
            # Check if eyes are open by calculating eye aspect ratio
            eyes_open = False
            if len(left_eye) > 0 or len(right_eye) > 0:
                eye_regions = []
                if len(left_eye) > 0:
                    ex, ey, ew, eh = left_eye[0]
                    eye_regions.append(face_roi[ey:ey+eh, ex:ex+ew])
                if len(right_eye) > 0:
                    ex, ey, ew, eh = right_eye[0]
                    eye_regions.append(face_roi[ey:ey+eh, ex:ex+ew])
                
                # Calculate average eye aspect ratio
                avg_ear = np.mean([self._calculate_eye_aspect_ratio(region) 
                                 for region in eye_regions])
                eyes_open = avg_ear > self.MIN_EYE_ASPECT_RATIO
            
            # Update detection result
            detection = EyeDetection(
                left_eye=left_eye,
                right_eye=right_eye,
                ir_intensity=ir_intensity,
                timestamp=datetime.now(),
                is_valid=True,  # Face detected
                eyes_open=eyes_open
            )
            
            # Reset or increment invalid frame counter
            if eyes_open:
                self.consecutive_invalid_frames = 0
            else:
                self.consecutive_invalid_frames += 1
                if self.consecutive_invalid_frames >= self.MAX_INVALID_FRAMES:
                    detection.is_valid = False
        
        return detection

    def extract_features(self, frame: np.ndarray, eye_data: EyeDetection) -> Optional[Dict]:
        """Extract features with enhanced IR support and validity checking"""
        if not eye_data.is_valid or not eye_data.eyes_open:
            return None
            
        # Calculate base features
        features = {
            "ir_intensity": eye_data.ir_intensity,
            "timestamp": datetime.now().timestamp()
        }
        
        # Enhanced IR-based features for dark conditions
        if eye_data.ir_intensity < self.MIN_IR_INTENSITY:
            # Adjust feature extraction for low-light conditions
            features.update({
                "pupil_size": self._calculate_ir_pupil_size(frame),
                "pupil_circularity": self._calculate_ir_pupil_shape(frame),
                "vein_prominence": self._calculate_ir_vein_patterns(frame)
            })
        else:
            # Normal light condition features
            features.update({
                "pupil_size": self._calculate_pupil_size(frame, eye_data),
                "pupil_circularity": self._calculate_pupil_circularity(frame, eye_data),
                "vein_prominence": self._calculate_vein_prominence(frame, eye_data)
            })
        
        return features

    def _calculate_ir_pupil_size(self, frame: np.ndarray) -> float:
        """Calculate pupil size using IR-optimized detection"""
        # Implementation would use IR-specific image processing
        # This is a placeholder for the actual implementation
        return np.random.uniform(20, 100)

    def _calculate_ir_pupil_shape(self, frame: np.ndarray) -> float:
        """Calculate pupil shape metrics in IR conditions"""
        return np.random.uniform(0.5, 1.0)

    def _calculate_ir_vein_patterns(self, frame: np.ndarray) -> float:
        """Analyze vein patterns using IR imaging"""
        return np.random.uniform(0, 10)

    def draw_overlay(self, frame: np.ndarray, eye_data: EyeDetection):
        """Enhanced overlay with detection status"""
        # Draw FPS
        fps = self.calculate_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw detection status
        status_color = (0, 255, 0) if eye_data.is_valid and eye_data.eyes_open else (0, 0, 255)
        status_text = "Detection: "
        if not eye_data.is_valid:
            status_text += "No Face Detected"
        elif not eye_data.eyes_open:
            status_text += "Eyes Closed"
        else:
            status_text += "Valid"
        
        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw glucose reading
        with self.prediction_lock:
            if self.latest_prediction is not None and eye_data.is_valid and eye_data.eyes_open:
                glucose_text = f"Glucose: {self.latest_prediction} mg/dL"
                cv2.putText(frame, glucose_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Glucose: No Reading", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw IR status
        ir_text = f"IR Intensity: {eye_data.ir_intensity:.1f}"
        ir_color = (0, 255, 0) if eye_data.ir_intensity >= self.MIN_IR_INTENSITY else (255, 165, 0)
        cv2.putText(frame, ir_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ir_color, 2)

    def run(self):
        """Run the enhanced eye glucose monitoring system"""
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

                # Enhanced detection
                eye_data = self.detect_face_and_eyes(frame)
                
                # Only process if detection is valid and eyes are open
                if eye_data.is_valid and eye_data.eyes_open:
                    features = self.extract_features(frame, eye_data)
                    if features:
                        if (self.prediction_thread is None or 
                            not self.prediction_thread.is_alive()):
                            self.prediction_thread = threading.Thread(
                                target=self.predict_glucose_async,
                                args=(features,)
                            )
                            self.prediction_thread.start()
                else:
                    # Clear prediction if eyes are closed or face not detected
                    with self.prediction_lock:
                        self.latest_prediction = None

                # Draw information overlay
                self.draw_overlay(frame, eye_data)

                # Display the frame
                cv2.imshow("Enhanced Eye Glucose Monitor", frame)

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