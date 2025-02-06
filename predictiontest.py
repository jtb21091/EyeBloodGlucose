import os
import cv2
import pandas as pd  # type: ignore
import numpy as np
import joblib  # type: ignore
from datetime import datetime
from typing import Dict, Any
import threading
import time
from collections import deque
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt  # type: ignore
import mediapipe as mp  # type: ignore

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

# ---------------------------------------------------------------------------
# Feature Extraction Functions using example “real” algorithms.
# (Adjust thresholds/parameters to match your training-time setup.)
# ---------------------------------------------------------------------------

def get_pupil_size(image):
    """
    Use HoughCircles to detect circular features (assumed pupils)
    and return the average radius.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=80
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Return the average radius as the pupil size.
            return round(np.mean([r for (_, _, r) in circles]), 5)
        else:
            return 0.0
    except Exception as e:
        logging.error("Error in get_pupil_size: " + str(e))
        return 0.0

def get_sclera_redness(image):
    """
    Convert the image to HSV and threshold the hue for red.
    The percentage of red pixels is returned as the redness index.
    """
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define a mask for red hues (adjust lower/upper bounds as needed)
        mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        redness = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1]) * 100
        return round(redness, 5)
    except Exception as e:
        logging.error("Error in get_sclera_redness: " + str(e))
        return 0.0

def get_vein_prominence(image):
    """
    Use Canny edge detection to approximate vein prominence.
    A simple metric is calculated based on the normalized density of edges.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
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
    Use Canny edge detection to compute the density of edges
    as a proxy for the density of scleral veins.
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
    (Adjust this method to your calibration.)
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

        # Initialize MediaPipe Face Mesh.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.last_valid_detection_time = time.time()
        self.invalid_detection_threshold = 3.0  # Seconds without valid detection before clearing reading
        self.prediction_lock = threading.Lock()
        
        # Store both instantaneous and EMA-smoothed predictions.
        self.latest_smoothed_prediction = None
        self.latest_instantaneous_prediction = None
        self.last_instantaneous_prediction = None  # For computing rate-of-change
        
        # EMA smoothing factor (alpha): Adjust to trade off between responsiveness and smoothing.
        self.alpha = 0.009
        
        self.last_features = None  # Store the most recent feature dictionary
        
        self.MIN_EYE_ASPECT_RATIO = 0.2  # Threshold for eyes open
        self.MIN_IR_INTENSITY = 30       # Threshold for dark conditions

        # Deques to store history of predictions for plotting.
        self.time_history = deque(maxlen=200)
        self.instantaneous_history = deque(maxlen=200)
        self.smoothed_history = deque(maxlen=200)
        
        # ---------------------------
        # Temporal Measurements: Maintain a history of recent pupil sizes.
        # ---------------------------
        self.pupil_history = deque(maxlen=30)  # Stores tuples of (timestamp, pupil_size)

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
        Detect the face and eyes using MediaPipe Face Mesh.
        Returns an EyeDetection dataclass instance.
        """
        # Convert the frame from BGR to RGB for MediaPipe.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_intensity = np.mean(gray)
        detection = EyeDetection([], [], ir_intensity, datetime.now(), False, False)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in face_landmarks.landmark]
            ys = [int(lm.y * h) for lm in face_landmarks.landmark]
            face_rect = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            detection.face_rect = face_rect
            detection.is_valid = True

            # Define landmark indices for the left and right eyes.
            # (These indices are suggested values; adjust if needed.)
            left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 246]
            right_eye_indices = [362, 263, 387, 386, 385, 384, 398]

            left_eye_points = []
            right_eye_points = []
            for i in left_eye_indices:
                if i < len(face_landmarks.landmark):
                    x = int(face_landmarks.landmark[i].x * w)
                    y = int(face_landmarks.landmark[i].y * h)
                    left_eye_points.append((x, y))
            for i in right_eye_indices:
                if i < len(face_landmarks.landmark):
                    x = int(face_landmarks.landmark[i].x * w)
                    y = int(face_landmarks.landmark[i].y * h)
                    right_eye_points.append((x, y))

            # Compute bounding boxes for each eye (if landmarks are found).
            if left_eye_points:
                lx = min(pt[0] for pt in left_eye_points)
                ly = min(pt[1] for pt in left_eye_points)
                rx = max(pt[0] for pt in left_eye_points)
                ry = max(pt[1] for pt in left_eye_points)
                left_eye_box = (lx, ly, rx - lx, ry - ly)
            else:
                left_eye_box = None

            if right_eye_points:
                lx = min(pt[0] for pt in right_eye_points)
                ly = min(pt[1] for pt in right_eye_points)
                rx = max(pt[0] for pt in right_eye_points)
                ry = max(pt[1] for pt in right_eye_points)
                right_eye_box = (lx, ly, rx - lx, ry - ly)
            else:
                right_eye_box = None

            detection.left_eye = [left_eye_box] if left_eye_box else []
            detection.right_eye = [right_eye_box] if right_eye_box else []

            # Determine if eyes are open based on the aspect ratio (height/width) of each eye box.
            ear_list = []
            if left_eye_box:
                _, _, ew, eh = left_eye_box
                ear_list.append(eh / ew if ew > 0 else 0)
            if right_eye_box:
                _, _, ew, eh = right_eye_box
                ear_list.append(eh / ew if ew > 0 else 0)
            if ear_list:
                avg_ear = np.mean(ear_list)
                detection.eyes_open = avg_ear > self.MIN_EYE_ASPECT_RATIO
            else:
                detection.eyes_open = False

        return detection

    def update_pupil_history(self, pupil_size: float):
        """
        Update the temporal buffer with the current pupil size and timestamp.
        """
        current_time = time.time()
        self.pupil_history.append((current_time, pupil_size))

    def compute_pupil_dilation_rate(self) -> float:
        """
        Compute the rate of change of pupil size (per second) using the two most recent measurements.
        """
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
        """
        Provide a crude estimate of the pupil response time based on the dilation rate.
        Here, the response time is defined as the inverse of the absolute dilation rate.
        """
        rate = self.compute_pupil_dilation_rate()
        if rate == 0:
            return 0.0
        response_time = 1.0 / abs(rate)
        return round(response_time, 5)

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

        # Use the union of the detected eye boxes (in absolute coordinates) to define the eye region.
        eye_boxes = []
        for box in detection.left_eye:
            if box is not None:
                eye_boxes.append(box)  # Each box is (x, y, w, h)
        for box in detection.right_eye:
            if box is not None:
                eye_boxes.append(box)
        if not eye_boxes:
            logging.warning("No eye boxes found.")
            return {}
        ex_min = min(box[0] for box in eye_boxes)
        ey_min = min(box[1] for box in eye_boxes)
        ex_max = max(box[0] + box[2] for box in eye_boxes)
        ey_max = max(box[1] + box[3] for box in eye_boxes)
        eye_roi = frame[ey_min:ey_max, ex_min:ex_max]

        # --- Temporal Measurements ---
        pupil_size = get_pupil_size(eye_roi)
        self.update_pupil_history(pupil_size)
        pupil_dilation_rate = self.compute_pupil_dilation_rate()
        pupil_response_time = self.compute_pupil_response_time()

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
            "birefringence_index": get_birefringence_index(eye_roi)
        }
        ordered_features = {key: features.get(key, 0) for key in FEATURES_ORDER}
        logging.debug("Extracted features: " + str(ordered_features))
        self.last_features = ordered_features
        return ordered_features

    def predict_glucose(self, features: Dict):
        """
        Predict blood glucose from features.
        Uses both the instantaneous prediction from the model and an exponential moving average (EMA)
        for smoothing.
        """
        result = None
        try:
            if self.model is not None and features:
                df = pd.DataFrame([features], columns=FEATURES_ORDER)
                prediction = self.model.predict(df)
                if len(prediction) > 0:
                    result = prediction[0]
                    if result is None or (isinstance(result, float) and np.isnan(result)):
                        result = None
        except Exception as e:
            logging.error("Prediction error: " + str(e))
            result = None

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
                if abs(rate_of_change) > 5:  # Adjust threshold as needed.
                    logging.info(f"High rate of change detected: {rate_of_change:.2f} mg/dL")
            self.last_instantaneous_prediction = result
            self.last_features = features

    def run(self):
        # Create the live plot on the main thread.
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
        else:
            print("Webcam successfully opened.")

        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received from the webcam.")
                break

            detection = self.detect_face_and_eyes(frame)
            if detection.is_valid and (len(detection.left_eye) > 0 or len(detection.right_eye) > 0):
                features = self.extract_features(frame)
                self.predict_glucose(features)
                current_time = time.time() - start_time
                with self.prediction_lock:
                    inst_pred = self.latest_instantaneous_prediction if self.latest_instantaneous_prediction is not None else np.nan
                    smooth_pred = self.latest_smoothed_prediction if self.latest_smoothed_prediction is not None else np.nan
                self.time_history.append(current_time)
                self.instantaneous_history.append(inst_pred)
                self.smoothed_history.append(smooth_pred)

                # Draw the detected face rectangle and eye boxes.
                if detection.face_rect is not None:
                    (x, y, w_box, h_box) = detection.face_rect
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                for box in detection.left_eye:
                    if box is not None:
                        (ex, ey, ew, eh) = box
                        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                for box in detection.right_eye:
                    if box is not None:
                        (ex, ey, ew, eh) = box
                        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            else:
                with self.prediction_lock:
                    self.latest_smoothed_prediction = None
                    self.latest_instantaneous_prediction = None

            # Update the live Matplotlib plot.
            line_inst.set_data(self.time_history, self.instantaneous_history)
            line_avg.set_data(self.time_history, self.smoothed_history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Display prediction text on the frame.
            with self.prediction_lock:
                inst_text = f"Inst: {self.latest_instantaneous_prediction:.1f} mg/dL" if self.latest_instantaneous_prediction is not None else "Inst: No Reading"
                smooth_text = f"Avg: {self.latest_smoothed_prediction:.1f} mg/dL" if self.latest_smoothed_prediction is not None else "Avg: No Reading"
            cv2.putText(frame, inst_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, smooth_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            # ----- Added Warning Checks -----
            # Check the instantaneous prediction first.
            if self.latest_instantaneous_prediction is not None:
                if self.latest_instantaneous_prediction < 40:
                    warning_inst = "Instantaneous Low. Please check yourself."
                    # Warning displayed on frame only.
                    cv2.putText(frame, warning_inst, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif self.latest_instantaneous_prediction > 400:
                    warning_inst = "Instantaneous High. Please check yourself."
                    # Warning displayed on frame only.
                    cv2.putText(frame, warning_inst, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Now check the EMA (smoothed) prediction.
            if self.latest_smoothed_prediction is not None:
                if self.latest_smoothed_prediction < 40:
                    warning_avg = "Average Low. Please check yourself."
                    # Warning displayed on frame only.
                    cv2.putText(frame, warning_avg, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif self.latest_smoothed_prediction > 400:
                    warning_avg = "Average High. Please check yourself."
                    # Warning displayed on frame only.
                    cv2.putText(frame, warning_avg, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # ----- End Added Warning Checks -----

            cv2.imshow("Blood Glucose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # A brief pause to update the plot.
            plt.pause(0.001)

        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    monitor = EyeGlucoseMonitor()
    monitor.run()
