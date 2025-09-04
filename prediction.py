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
    """Detect circular features (assumed pupils) and return the average radius."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=50, param2=30, minRadius=10, maxRadius=80
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
    """HSV threshold for red; percentage of red pixels."""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        redness = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1]) * 100
        return round(redness, 5)
    except Exception as e:
        logging.error("Error in get_sclera_redness: " + str(e))
        return 0.0

def get_vein_prominence(image):
    """Canny edges density as proxy for prominence."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        prominence = np.sum(edges) / (255.0 * image.shape[0] * image.shape[1])
        return round(prominence * 10, 5)
    except Exception as e:
        logging.error("Error in get_vein_prominence: " + str(e))
        return 0.0

def get_ir_intensity(image):
    """Mean grayscale intensity."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return round(np.mean(gray), 5)
    except Exception as e:
        logging.error("Error in get_ir_intensity: " + str(e))
        return 0.0

def get_scleral_vein_density(image):
    """Edge density as proxy for scleral vein density."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = np.sum(edges) / (255.0 * image.shape[0] * image.shape[1])
        return round(density, 5)
    except Exception as e:
        logging.error("Error in get_scleral_vein_density: " + str(e))
        return 0.0

def get_ir_temperature(image):
    """Mean of red channel (placeholder for calibrated IR temp)."""
    try:
        return round(np.mean(image[:, :, 2]), 5)
    except Exception as e:
        logging.error("Error in get_ir_temperature: " + str(e))
        return 0.0

def get_tear_film_reflectivity(image):
    """Std dev of grayscale as reflectivity measure."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return round(np.std(gray), 5)
    except Exception as e:
        logging.error("Error in get_tear_film_reflectivity: " + str(e))
        return 0.0

def get_sclera_color_balance(image):
    """Ratio of mean red to mean green."""
    try:
        r_mean = np.mean(image[:, :, 2])
        g_mean = np.mean(image[:, :, 1])
        return round(r_mean / g_mean, 5) if g_mean > 0 else 1.0
    except Exception as e:
        logging.error("Error in get_sclera_color_balance: " + str(e))
        return 1.0

def get_vein_pulsation_intensity(image):
    """Mean Laplacian as pulsation proxy."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pulsation = np.mean(cv2.Laplacian(gray, cv2.CV_64F))
        return round(pulsation, 5)
    except Exception as e:
        logging.error("Error in get_vein_pulsation_intensity: " + str(e))
        return 0.0

def get_birefringence_index(image):
    """Normalized variance of grayscale."""
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
    left_eye: Any
    right_eye: Any
    ir_intensity: float
    timestamp: datetime
    is_valid: bool
    eyes_open: bool
    face_rect: tuple = None

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
        self.invalid_detection_threshold = 3.0
        self.prediction_lock = threading.Lock()

        # Predictions (instant + EMA)
        self.latest_smoothed_prediction = None
        self.latest_instantaneous_prediction = None
        self.last_instantaneous_prediction = None
        self.alpha = 0.009

        self.last_features = None
        self.MIN_EYE_ASPECT_RATIO = 0.2
        self.MIN_IR_INTENSITY = 30

        # Histories
        self.time_history = deque(maxlen=200)
        self.instantaneous_history = deque(maxlen=200)
        self.smoothed_history = deque(maxlen=200)

        # Pupil history for temporal metrics
        self.pupil_history = deque(maxlen=30)

        # Stop flag for Matplotlib window close
        self._stop = False

    def _load_model(self) -> Any:
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except Exception as e:
                logging.error("Error loading model: " + str(e))
                return None
        else:
            logging.error("Model file not found at: " + self.model_path)
            return None

    def detect_face_and_eyes(self, frame: np.ndarray) -> EyeDetection:
        """Detect face + eye boxes using MediaPipe Face Mesh."""
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

            left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 246]
            right_eye_indices = [362, 263, 387, 386, 385, 384, 398]

            def gather_points(indices):
                pts = []
                for i in indices:
                    if i < len(face_landmarks.landmark):
                        x = int(face_landmarks.landmark[i].x * w)
                        y = int(face_landmarks.landmark[i].y * h)
                        pts.append((x, y))
                return pts

            left_eye_points = gather_points(left_eye_indices)
            right_eye_points = gather_points(right_eye_indices)

            def bbox(points):
                if not points:
                    return None
                lx = min(pt[0] for pt in points)
                ly = min(pt[1] for pt in points)
                rx = max(pt[0] for pt in points)
                ry = max(pt[1] for pt in points)
                return (lx, ly, rx - lx, ry - ly)

            left_eye_box = bbox(left_eye_points)
            right_eye_box = bbox(right_eye_points)

            detection.left_eye = [left_eye_box] if left_eye_box else []
            detection.right_eye = [right_eye_box] if right_eye_box else []

            ear_list = []
            if left_eye_box:
                _, _, ew, eh = left_eye_box
                ear_list.append(eh / ew if ew > 0 else 0)
            if right_eye_box:
                _, _, ew, eh = right_eye_box
                ear_list.append(eh / ew if ew > 0 else 0)
            detection.eyes_open = (np.mean(ear_list) > self.MIN_EYE_ASPECT_RATIO) if ear_list else False

        return detection

    def update_pupil_history(self, pupil_size: float):
        self.pupil_history.append((time.time(), pupil_size))

    def compute_pupil_dilation_rate(self) -> float:
        if len(self.pupil_history) < 2:
            return 0.0
        t0, p0 = self.pupil_history[-2]
        t1, p1 = self.pupil_history[-1]
        dt = t1 - t0
        if dt == 0:
            return 0.0
        return round((p1 - p0) / dt, 5)

    def compute_pupil_response_time(self) -> float:
        rate = self.compute_pupil_dilation_rate()
        if rate == 0:
            return 0.0
        return round(1.0 / abs(rate), 5)

    def extract_features(self, frame: np.ndarray) -> Dict:
        detection = self.detect_face_and_eyes(frame)
        if not detection.is_valid or (len(detection.left_eye) == 0 and len(detection.right_eye) == 0):
            logging.warning("No valid eyes detected.")
            return {}
        if detection.face_rect is None:
            logging.warning("No face rectangle detected.")
            return {}

        eye_boxes = []
        for box in detection.left_eye:
            if box is not None:
                eye_boxes.append(box)
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
            self.latest_instantaneous_prediction = result if result is not None else None
            if result is not None:
                self.latest_smoothed_prediction = (
                    result if self.latest_smoothed_prediction is None
                    else self.alpha * result + (1 - self.alpha) * self.latest_smoothed_prediction
                )
            else:
                self.latest_smoothed_prediction = None

            if self.last_instantaneous_prediction is not None and result is not None:
                rate_of_change = result - self.last_instantaneous_prediction
                if abs(rate_of_change) > 5:
                    logging.info(f"High rate of change detected: {rate_of_change:.2f} mg/dL")
            self.last_instantaneous_prediction = result
            self.last_features = features

    def run(self):
        # Create the live plot on the main thread.
        plt.ion()
        fig, ax = plt.subplots()
        line_inst, = ax.plot([], [], label="Instantaneous", color="green")
        line_avg, = ax.plot([], [], label="Smoothed", color="orange")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Blood Glucose (mg/dL)")
        ax.legend()

        # Let closing the Matplotlib window stop the loop.
        def _on_close(_):
            self._stop = True
        fig.canvas.mpl_connect("close_event", _on_close)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam.")
            return
        else:
            print("Webcam successfully opened.")

        start_time = time.time()

        try:
            while True:
                if self._stop:
                    break

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

                    # Draw face rectangle and eye boxes.
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

                # Update the live plot.
                line_inst.set_data(self.time_history, self.instantaneous_history)
                line_avg.set_data(self.time_history, self.smoothed_history)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

                # Overlay text.
                with self.prediction_lock:
                    inst_text = f"Inst: {self.latest_instantaneous_prediction:.1f} mg/dL" if self.latest_instantaneous_prediction is not None else "Inst: No Reading"
                    smooth_text = f"Avg: {self.latest_smoothed_prediction:.1f} mg/dL" if self.latest_smoothed_prediction is not None else "Avg: No Reading"
                cv2.putText(frame, inst_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, smooth_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

                # Warnings
                if self.latest_instantaneous_prediction is not None:
                    if self.latest_instantaneous_prediction < 40:
                        cv2.putText(frame, "Instantaneous Low. Please check yourself.", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    elif self.latest_instantaneous_prediction > 400:
                        cv2.putText(frame, "Instantaneous High. Please check yourself.", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if self.latest_smoothed_prediction is not None:
                    if self.latest_smoothed_prediction < 40:
                        cv2.putText(frame, "Average Low. Please check yourself.", (10, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    elif self.latest_smoothed_prediction > 400:
                        cv2.putText(frame, "Average High. Please check yourself.", (10, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Blood Glucose", frame)

                # Quit if 'q' or ESC is pressed, or if the window is closed
                key = cv2.waitKey(1)
                if key in (ord('q'), 27):  # 27 = ESC
                    break
                if cv2.getWindowProperty("Blood Glucose", cv2.WND_PROP_VISIBLE) < 1:
                    break

                # Keep Matplotlib responsive
                plt.pause(0.001)

        except KeyboardInterrupt:
            # Allow Ctrl+C to stop the loop cleanly
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    monitor = EyeGlucoseMonitor()
    monitor.run()
