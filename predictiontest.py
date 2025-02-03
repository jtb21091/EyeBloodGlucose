import os
import cv2
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Any
import threading
import time
from collections import deque
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt

# Set logging to show warnings (and errors) only.
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Updated expected feature order (the original features used during model training)
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
    # Note: "pupil_variance" is calculated for internal analysis but not passed to the model.
]

# ---------------------------------------------------------------------------
# Feature Extraction Functions with improvements.
# ---------------------------------------------------------------------------

def get_pupil_size(image):
    """
    Detect pupils using HoughCircles after applying a median blur.
    Returns the average radius as the pupil size.
    Tuning parameters for better responsiveness.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply a median blur to reduce noise.
        gray_blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,         # Slightly adjusted for a tighter accumulator resolution.
            minDist=30,     # Reduced minimum distance for potentially more circles.
            param1=30,      # Lowered edge detection threshold.
            param2=20,      # Lower accumulator threshold for circle detection.
            minRadius=5,    # Reduced minimum radius.
            maxRadius=100   # Increased maximum radius to cover a broader range.
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return round(np.mean([r for (_, _, r) in circles]), 5)
        else:
            return 0.0
    except Exception as e:
        logging.error("Error in get_pupil_size: " + str(e))
        return 0.0

def get_sclera_redness(image):
    """
    Convert the image to HSV and threshold for red hues.
    This version combines two hue ranges to capture the full red spectrum.
    Returns the percentage of red pixels as the redness index.
    """
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        redness = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1]) * 100
        return round(redness, 5)
    except Exception as e:
        logging.error("Error in get_sclera_redness: " + str(e))
        return 0.0

def get_vein_prominence(image):
    """
    Use Canny edge detection (after a small Gaussian blur) to approximate vein prominence.
    A simple metric is calculated based on the normalized density of edges.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        prominence = np.sum(edges) / (255.0 * image.shape[0] * image.shape[1])
        return round(prominence * 10, 5)
    except Exception as e:
        logging.error("Error in get_vein_prominence: " + str(e))
        return 0.0

def get_ir_intensity(image):
    """
    Calculate the IR intensity as the mean of the grayscale values.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return round(np.mean(gray), 5)
    except Exception as e:
        logging.error("Error in get_ir_intensity: " + str(e))
        return 0.0

def get_scleral_vein_density(image):
    """
    Use Canny edge detection to compute the density of edges as a proxy for scleral vein density.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = np.sum(edges) / (255.0 * image.shape[0] * image.shape[1])
        return round(density, 5)
    except Exception as e:
        logging.error("Error in get_scleral_vein_density: " + str(e))
        return 0.0

def get_ir_temperature(image):
    """
    Compute IR temperature using the mean of the red channel.
    """
    try:
        return round(np.mean(image[:, :, 2]), 5)
    except Exception as e:
        logging.error("Error in get_ir_temperature: " + str(e))
        return 0.0

def get_tear_film_reflectivity(image):
    """
    Use the standard deviation of the grayscale image as a measure of tear film reflectivity.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return round(np.std(gray), 5)
    except Exception as e:
        logging.error("Error in get_tear_film_reflectivity: " + str(e))
        return 0.0

def get_sclera_color_balance(image):
    """
    Compute the ratio of the mean red channel to the mean green channel.
    """
    try:
        r_mean = np.mean(image[:, :, 2])
        g_mean = np.mean(image[:, :, 1])
        return round(r_mean / g_mean, 5) if g_mean > 0 else 1.0
    except Exception as e:
        logging.error("Error in get_sclera_color_balance: " + str(e))
        return 1.0

def get_vein_pulsation_intensity(image):
    """
    Estimate vein pulsation intensity using the mean of the Laplacian operator.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pulsation = np.mean(cv2.Laplacian(gray, cv2.CV_64F))
        return round(pulsation, 5)
    except Exception as e:
        logging.error("Error in get_vein_pulsation_intensity: " + str(e))
        return 0.0

def get_birefringence_index(image):
    """
    Compute a birefringence index based on the normalized variance of the grayscale image.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return round(np.var(gray) / 255.0, 5)
    except Exception as e:
        logging.error("Error in get_birefringence_index: " + str(e))
        return 0.0

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
    eyes_open: bool        # Whether the eyes are considered "open".
    face_rect: tuple = None  # The bounding box of the detected face (x, y, w, h)

# ---------------------------
# Main Prediction Code
# ---------------------------
class EyeGlucoseMonitor:
    def __init__(self, model_path: str = "eye_glucose_model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()
        
        # Haar cascades for face and eye detection.
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
        
        # Store both instantaneous and EMA-smoothed predictions.
        self.latest_smoothed_prediction = None
        self.latest_instantaneous_prediction = None
        self.last_instantaneous_prediction = None  # For computing rate-of-change
        
        # EMA smoothing factor.
        self.alpha = 0.009
        
        self.last_features = None  # Store the most recent feature dictionary
        
        self.MIN_EYE_ASPECT_RATIO = 0.2  # Threshold for eyes open
        self.MIN_IR_INTENSITY = 30       # Threshold for dark conditions

        # Deques to store history of predictions for plotting.
        self.time_history = deque(maxlen=200)
        self.instantaneous_history = deque(maxlen=200)
        self.smoothed_history = deque(maxlen=200)
        
        # Temporal measurements: Maintain a history of recent pupil sizes.
        self.pupil_history = deque(maxlen=30)  # Stores tuples of (timestamp, pupil_size)
        
        # For a rolling measure of prediction variability.
        self.prediction_history = deque(maxlen=10)
        
        # Store the most recent confidence level (a value between 0 and 1)
        self.latest_confidence = None

    def _load_model(self) -> Any:
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                return model
            except Exception as e:
                logging.error("Error loading model: " + str(e))
                return None
        else:
            logging.error("Model file not found at: " + self.model_path)
            return None

    def detect_face_and_eyes(self, frame: np.ndarray) -> EyeDetection:
        """
        Detect the face and eyes using Haar cascades.
        Returns an EyeDetection dataclass instance.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_intensity = np.mean(gray)
        enhanced = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(enhanced, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
        
        if len(faces) == 0 and ir_intensity < self.MIN_IR_INTENSITY:
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            faces = self.face_cascade.detectMultiScale(blurred, scaleFactor=1.1, minNeighbors=2, minSize=(50, 50))
        
        detection = EyeDetection([], [], ir_intensity, datetime.now(), False, False)
        if len(faces) > 0:
            face = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = face
            detection.face_rect = face
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

    def update_pupil_history(self, pupil_size: float):
        current_time = time.time()
        self.pupil_history.append((current_time, pupil_size))

    def compute_pupil_dilation_rate(self) -> float:
        if len(self.pupil_history) < 2:
            return 0.0
        t0, p0 = self.pupil_history[-2]
        t1, p1 = self.pupil_history[-1]
        dt = t1 - t0
        if dt == 0:
            return 0.0
        rate = (p1 - p0) / dt
        return round(rate, 5)

    def compute_pupil_response_time(self) -> float:
        rate = self.compute_pupil_dilation_rate()
        if rate == 0:
            return 0.0
        response_time = 1.0 / abs(rate)
        return round(response_time, 5)

    def extract_features(self, frame: np.ndarray) -> Dict:
        detection = self.detect_face_and_eyes(frame)
        if not detection.is_valid or (len(detection.left_eye) == 0 and len(detection.right_eye) == 0):
            logging.warning("No valid eyes detected.")
            return {}
        if detection.face_rect is None:
            logging.warning("No face rectangle detected.")
            return {}

        fx, fy, fw, fh = detection.face_rect
        face_roi = frame[fy:fy+fh, fx:fx+fw]

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
        eye_roi_x = fx + ex_min
        eye_roi_y = fy + ey_min
        eye_roi_w = ex_max - ex_min
        eye_roi_h = ey_max - ey_min
        eye_roi = frame[eye_roi_y:eye_roi_y+eye_roi_h, eye_roi_x:eye_roi_x+eye_roi_w]

        pupil_size = get_pupil_size(eye_roi)
        self.update_pupil_history(pupil_size)
        pupil_dilation_rate = self.compute_pupil_dilation_rate()
        pupil_response_time = self.compute_pupil_response_time()
        if len(self.pupil_history) < 2:
            pupil_variance = 0.0
        else:
            sizes = [size for (_, size) in self.pupil_history]
            pupil_variance = round(np.var(sizes), 5)

        features = {
            "pupil_size": pupil_size,
            "sclera_redness": get_sclera_redness(eye_roi),
            "vein_prominence": get_vein_prominence(eye_roi),
            "pupil_response_time": pupil_response_time,
            "ir_intensity": get_ir_intensity(eye_roi),
            "scleral_vein_density": get_scleral_vein_density(eye_roi),
            "ir_temperature": get_ir_temperature(eye_roi),
            "tear_film_reflectivity": get_tear_film_reflectivity(eye_roi),
            "pupil_dilation_rate": pupil_dilation_rate,
            "sclera_color_balance": get_sclera_color_balance(eye_roi),
            "vein_pulsation_intensity": get_vein_pulsation_intensity(eye_roi),
            "birefringence_index": get_birefringence_index(eye_roi),
            "pupil_variance": pupil_variance  # Extra internal analysis
        }
        ordered_features = {key: features.get(key, 0) for key in FEATURES_ORDER}
        logging.debug("Extracted features: " + str(ordered_features))
        self.last_features = ordered_features
        return features  # Return full features; filtering is done in predict_glucose

    def predict_glucose(self, features: Dict):
        result = None
        confidence = None
        try:
            if self.model is not None and features:
                if hasattr(self.model, "feature_names_in_"):
                    expected_features = list(self.model.feature_names_in_)
                    features_to_pass = {k: features.get(k, 0) for k in expected_features}
                else:
                    features_to_pass = {k: features.get(k, 0) for k in FEATURES_ORDER}
                df = pd.DataFrame([features_to_pass])
                
                # Primary prediction
                prediction = self.model.predict(df)
                if len(prediction) > 0:
                    result = prediction[0]
                    if result is None or (isinstance(result, float) and np.isnan(result)):
                        result = None

                # Attempt to compute confidence using the model's ensemble (if available)
                if hasattr(self.model, "estimators_"):
                    try:
                        all_preds = np.array([estimator.predict(df)[0] for estimator in self.model.estimators_])
                        std_prediction = np.std(all_preds)
                        confidence = 1.0 / (1.0 + std_prediction)
                    except Exception as e:
                        logging.error("Error computing ensemble confidence: " + str(e))
                        confidence = None
                elif hasattr(self.model, "predict") and "return_std" in self.model.predict.__code__.co_varnames:
                    try:
                        mean_pred, std_prediction = self.model.predict(df, return_std=True)
                        confidence = 1.0 / (1.0 + std_prediction[0])
                    except Exception as e:
                        logging.error("Error computing GP confidence: " + str(e))
                        confidence = None
                # If no built-in uncertainty is available, use a rolling measure based on recent predictions.
                if confidence is None:
                    confidence = 1.0  # Start with default
                # Update prediction history (for our rolling confidence measure)
                if result is not None:
                    self.prediction_history.append(result)
                    if len(self.prediction_history) > 1:
                        rolling_std = np.std(self.prediction_history)
                        # Scale factor (adjust as needed) to map std to a confidence between 0 and 1.
                        confidence = 1.0 / (1.0 + rolling_std * 0.1)
                    else:
                        confidence = 1.0
        except Exception as e:
            logging.error("Prediction error: " + str(e))
            result = None
            confidence = None

        with self.prediction_lock:
            if result is not None:
                self.latest_instantaneous_prediction = result
            else:
                self.latest_instantaneous_prediction = None

            if result is not None:
                if self.latest_smoothed_prediction is None:
                    self.latest_smoothed_prediction = result
                else:
                    self.latest_smoothed_prediction = self.alpha * result + (1 - self.alpha) * self.latest_smoothed_prediction
            else:
                self.latest_smoothed_prediction = None

            if self.last_instantaneous_prediction is not None and result is not None:
                rate_of_change = result - self.last_instantaneous_prediction
                if abs(rate_of_change) > 5:
                    logging.info(f"High rate of change detected: {rate_of_change:.2f} mg/dL")
            self.last_instantaneous_prediction = result
            self.last_features = features
            self.latest_confidence = confidence

    def run(self):
        plt.ion()  # Interactive mode on.
        fig, ax = plt.subplots()
        line_inst, = ax.plot([], [], label="Instantaneous", color="green")
        line_avg, = ax.plot([], [], label="Smoothed", color="orange")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Blood Glucose (mg/dL)")
        ax.legend()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam.")
            return

        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = self.detect_face_and_eyes(frame)
            if detection.is_valid and (len(detection.left_eye) > 0 or len(detection.right_eye) > 0):
                features = self.extract_features(frame)
                self.predict_glucose(features)
                current_time = time.time() - start_time
                with self.prediction_lock:
                    inst_pred = self.latest_instantaneous_prediction if self.latest_instantaneous_prediction is not None else np.nan
                    smooth_pred = self.latest_smoothed_prediction if self.latest_smoothed_prediction is not None else np.nan
                    conf = self.latest_confidence if self.latest_confidence is not None else 0.0
                self.time_history.append(current_time)
                self.instantaneous_history.append(inst_pred)
                self.smoothed_history.append(smooth_pred)

                if detection.face_rect is not None:
                    (x, y, w, h) = detection.face_rect
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    for (ex, ey, ew, eh) in detection.left_eye:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                    for (ex, ey, ew, eh) in detection.right_eye:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)
            else:
                with self.prediction_lock:
                    self.latest_smoothed_prediction = None
                    self.latest_instantaneous_prediction = None
                    self.latest_confidence = None

            line_inst.set_data(self.time_history, self.instantaneous_history)
            line_avg.set_data(self.time_history, self.smoothed_history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

            with self.prediction_lock:
                inst_text = f"Inst: {self.latest_instantaneous_prediction:.1f} mg/dL" if self.latest_instantaneous_prediction is not None else "Inst: No Reading"
                smooth_text = f"Avg: {self.latest_smoothed_prediction:.1f} mg/dL" if self.latest_smoothed_prediction is not None else "Avg: No Reading"
                conf_text = f"Conf: {self.latest_confidence:.2f}" if self.latest_confidence is not None else "Conf: N/A"
            cv2.putText(frame, inst_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, smooth_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(frame, conf_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
            cv2.imshow("Blood Glucose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            plt.pause(0.001)

        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    monitor = EyeGlucoseMonitor()
    monitor.run()
