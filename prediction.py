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

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class EyeDetection:
    """Data class for eye detection results."""
    left_eye: Any          # Typically, a list or array of detected eye bounding boxes.
    right_eye: Any
    ir_intensity: float
    timestamp: datetime
    is_valid: bool         # Indicates if the detection meets our criteria.
    eyes_open: bool        # Indicates if the eyes are open.


class EyeGlucoseMonitor:
    def __init__(self, model_path: str = "eye_glucose_model.pkl"):
        """
        Initialize the EyeGlucoseMonitor.
        
        Args:
            model_path: Path to the trained glucose prediction model.
        """
        self.model_path = model_path
        self.model = self._load_model()
        self.trained_features = self._get_trained_features()
        self.eye_cascades = self._initialize_cascades()

        # Initialize face cascade for face detection.
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Buffers for smoothing and analysis.
        self.glucose_buffer = deque(maxlen=60)
        self.detection_buffer = deque(maxlen=30)
        self.fps_buffer = deque(maxlen=30)

        # Timing and state variables.
        self.last_frame_time = time.time()
        self.consecutive_invalid_frames = 0
        self.MIN_IR_INTENSITY = 30  # Minimum IR intensity threshold for dark conditions.

        # Threading for asynchronous predictions.
        self.prediction_thread = None
        self.latest_prediction = None
        self.prediction_lock = threading.Lock()

        # Detection parameters.
        self.MIN_EYE_ASPECT_RATIO = 0.2  # Threshold for eye openness.
        self.MAX_INVALID_FRAMES = 5      # Number of invalid frames before marking detection as invalid.

    def _load_model(self) -> Any:
        """
        Load the machine learning model from disk.
        
        Returns:
            The loaded model, or None if loading fails.
        """
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                logging.info("Model loaded successfully.")
                return model
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                raise e
        else:
            logging.warning("Model file not found. Using a dummy model.")
            # Return a dummy model (or None) as a placeholder.
            return None

    def _get_trained_features(self) -> Optional[list]:
        """
        Return the list of features expected by the model.
        
        Returns:
            A list of feature names.
        """
        # Dummy implementation: update with actual feature names if available.
        return ["ir_intensity", "timestamp", "pupil_size", "pupil_circularity", "vein_prominence"]

    def _initialize_cascades(self) -> Dict[str, cv2.CascadeClassifier]:
        """
        Initialize Haar cascades for left and right eyes.
        
        Returns:
            A dictionary with cascades for 'left' and 'right' eyes.
        """
        cascades = {
            "left": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml'),
            "right": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
        }
        return cascades

    def _calculate_eye_aspect_ratio(self, eye_region: np.ndarray) -> float:
        """
        Calculate the eye aspect ratio (EAR) as a simple measure of eye openness.
        
        Args:
            eye_region: The region of the image containing the eye.
        
        Returns:
            A float representing the aspect ratio.
        """
        height, width = eye_region.shape[:2]
        return height / width if width > 0 else 0.0

    def detect_face_and_eyes(self, frame: np.ndarray) -> EyeDetection:
        """
        Detect a face and eyes in the given frame.
        
        Args:
            frame: The current video frame in BGR format.
        
        Returns:
            An EyeDetection instance containing detection results.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_intensity = np.mean(gray)

        # Enhance contrast for improved detection in low-light conditions.
        enhanced_frame = cv2.equalizeHist(gray)

        # Detect face in the enhanced frame.
        faces = self.face_cascade.detectMultiScale(
            enhanced_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        # If no face is detected and the IR intensity is low, try an alternative detection strategy.
        if len(faces) == 0 and ir_intensity < self.MIN_IR_INTENSITY:
            logging.debug("Low IR intensity; attempting alternative face detection with Gaussian blur.")
            blurred_frame = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)
            faces = self.face_cascade.detectMultiScale(
                blurred_frame,
                scaleFactor=1.1,
                minNeighbors=3,  # More lenient parameters for dark conditions.
                minSize=(80, 80)
            )

        # Initialize a default (invalid) detection.
        detection = EyeDetection(
            left_eye=[],
            right_eye=[],
            ir_intensity=ir_intensity,
            timestamp=datetime.now(),
            is_valid=False,
            eyes_open=False
        )

        if len(faces) > 0:
            # Choose the largest detected face.
            face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = face
            face_roi = enhanced_frame[y:y+h, x:x+w]

            # Detect eyes within the face region.
            left_eye = self.eye_cascades['left'].detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            right_eye = self.eye_cascades['right'].detectMultiScale(
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
                avg_ear = np.mean([self._calculate_eye_aspect_ratio(region)
                                   for region in eye_regions])
                eyes_open = avg_ear > self.MIN_EYE_ASPECT_RATIO

            # Update detection results.
            detection = EyeDetection(
                left_eye=left_eye,
                right_eye=right_eye,
                ir_intensity=ir_intensity,
                timestamp=datetime.now(),
                is_valid=True,
                eyes_open=eyes_open
            )

            # Track consecutive frames with closed eyes.
            if eyes_open:
                self.consecutive_invalid_frames = 0
            else:
                self.consecutive_invalid_frames += 1
                if self.consecutive_invalid_frames >= self.MAX_INVALID_FRAMES:
                    detection.is_valid = False

        return detection

    def extract_features(self, frame: np.ndarray, eye_data: EyeDetection) -> Optional[Dict]:
        """
        Extract features from the current frame and detection data.
        
        Args:
            frame: The current video frame.
            eye_data: The result of the face and eye detection.
        
        Returns:
            A dictionary of features or None if the detection is invalid.
        """
        if not eye_data.is_valid or not eye_data.eyes_open:
            return None

        # Base features.
        features = {
            "ir_intensity": eye_data.ir_intensity,
            "timestamp": datetime.now().timestamp()
        }

        # Adjust feature extraction based on lighting conditions.
        if eye_data.ir_intensity < self.MIN_IR_INTENSITY:
            features.update({
                "pupil_size": self._calculate_ir_pupil_size(frame),
                "pupil_circularity": self._calculate_ir_pupil_shape(frame),
                "vein_prominence": self._calculate_ir_vein_patterns(frame)
            })
        else:
            features.update({
                "pupil_size": self._calculate_pupil_size(frame, eye_data),
                "pupil_circularity": self._calculate_pupil_circularity(frame, eye_data),
                "vein_prominence": self._calculate_vein_prominence(frame, eye_data)
            })

        return features

    # Dummy implementations of feature extraction methods. Replace these with real image processing.
    def _calculate_ir_pupil_size(self, frame: np.ndarray) -> float:
        """Calculate pupil size using IR-optimized detection (dummy implementation)."""
        return np.random.uniform(20, 100)

    def _calculate_ir_pupil_shape(self, frame: np.ndarray) -> float:
        """Calculate pupil circularity in IR conditions (dummy implementation)."""
        return np.random.uniform(0.5, 1.0)

    def _calculate_ir_vein_patterns(self, frame: np.ndarray) -> float:
        """Analyze vein patterns using IR imaging (dummy implementation)."""
        return np.random.uniform(0, 10)

    def _calculate_pupil_size(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        """Calculate pupil size under normal lighting (dummy implementation)."""
        return np.random.uniform(20, 100)

    def _calculate_pupil_circularity(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        """Calculate pupil circularity under normal lighting (dummy implementation)."""
        return np.random.uniform(0.5, 1.0)

    def _calculate_vein_prominence(self, frame: np.ndarray, eye_data: EyeDetection) -> float:
        """Calculate vein prominence under normal lighting (dummy implementation)."""
        return np.random.uniform(0, 10)

    def calculate_fps(self) -> float:
        """
        Calculate the current frames per second (FPS).
        
        Returns:
            The calculated FPS value.
        """
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self.last_frame_time = current_time
        return fps

    def predict_glucose_async(self, features: Dict):
        """
        Asynchronously predict the glucose level using the extracted features.
        
        Args:
            features: A dictionary of features extracted from the frame.
        """
        try:
            if self.model is not None:
                # Convert features into a DataFrame for prediction.
                df = pd.DataFrame([features])
                prediction = self.model.predict(df)
                result = prediction[0] if len(prediction) > 0 else None
            else:
                # Dummy prediction if no model is loaded.
                result = np.random.uniform(70, 150)
            with self.prediction_lock:
                self.latest_prediction = result
        except Exception as e:
            logging.error(f"Error during glucose prediction: {e}")

    def draw_overlay(self, frame: np.ndarray, eye_data: EyeDetection):
        """
        Draw overlays on the frame showing FPS, detection status, glucose readings, and IR intensity.
        
        Args:
            frame: The current video frame.
            eye_data: The detection data to display.
        """
        fps = self.calculate_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not eye_data.is_valid:
            status_text = "Detection: No Face Detected"
            status_color = (0, 0, 255)
        elif not eye_data.eyes_open:
            status_text = "Detection: Eyes Closed"
            status_color = (0, 0, 255)
        else:
            status_text = "Detection: Valid"
            status_color = (0, 255, 0)

        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        with self.prediction_lock:
            if self.latest_prediction is not None and eye_data.is_valid and eye_data.eyes_open:
                glucose_text = f"Glucose: {self.latest_prediction:.1f} mg/dL"
                cv2.putText(frame, glucose_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Glucose: No Reading", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ir_text = f"IR Intensity: {eye_data.ir_intensity:.1f}"
        ir_color = (0, 255, 0) if eye_data.ir_intensity >= self.MIN_IR_INTENSITY else (255, 165, 0)
        cv2.putText(frame, ir_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ir_color, 2)

    def run(self):
        """
        Run the eye glucose monitoring system.
        Captures video frames, performs detection and feature extraction, and displays an overlay.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Failed to open camera.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to read frame from camera.")
                    break

                # Perform face and eye detection.
                eye_data = self.detect_face_and_eyes(frame)
                if eye_data.is_valid and eye_data.eyes_open:
                    features = self.extract_features(frame, eye_data)
                    if features:
                        # Run glucose prediction asynchronously.
                        if (self.prediction_thread is None or not self.prediction_thread.is_alive()):
                            self.prediction_thread = threading.Thread(
                                target=self.predict_glucose_async,
                                args=(features,)
                            )
                            self.prediction_thread.start()
                else:
                    # Clear the previous prediction if detection is invalid.
                    with self.prediction_lock:
                        self.latest_prediction = None

                # Draw overlays on the frame.
                self.draw_overlay(frame, eye_data)
                cv2.imshow("Enhanced Eye Glucose Monitor", frame)

                # Press 'q' to quit.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logging.error(f"Runtime error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        monitor = EyeGlucoseMonitor()
        monitor.run()
    except Exception as e:
        logging.error(f"Application error: {e}")
