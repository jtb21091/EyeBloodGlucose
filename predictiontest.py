import os
import cv2
import pandas as pd  # type: ignore
import numpy as np
import joblib  # type: ignore
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import threading
import time
from collections import deque
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# -------- feature functions (return NaN on failure; no zeros) --------
def get_pupil_size(image):
    try:
        if image is None or image.size == 0: return np.nan
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=50, param2=30, minRadius=10, maxRadius=80
        )
        if circles is None: return np.nan
        circles = np.round(circles[0, :]).astype("int")
        return float(np.mean([r for (_, _, r) in circles]))
    except Exception as e:
        logging.warning(f"get_pupil_size error: {e}")
        return np.nan

def get_sclera_redness(image):
    try:
        if image is None or image.size == 0: return np.nan
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        mask = mask1 | mask2
        if mask.size == 0: return np.nan
        redness = cv2.countNonZero(mask) / mask.size * 100.0
        return float(round(redness, 5))
    except Exception as e:
        logging.warning(f"get_sclera_redness error: {e}")
        return np.nan

def get_vein_prominence(image):
    try:
        if image is None or image.size == 0: return np.nan
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        if edges.size == 0: return np.nan
        prominence = np.sum(edges) / (255.0 * edges.size)
        return float(round(prominence * 10, 5))
    except Exception as e:
        logging.warning(f"get_vein_prominence error: {e}")
        return np.nan

def get_ir_intensity(image):
    try:
        if image is None or image.size == 0: return np.nan
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(round(np.mean(gray), 5))
    except Exception as e:
        logging.warning(f"get_ir_intensity error: {e}")
        return np.nan

def get_scleral_vein_density(image):
    try:
        if image is None or image.size == 0: return np.nan
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        if edges.size == 0: return np.nan
        density = np.sum(edges) / (255.0 * edges.size)
        return float(round(density, 5))
    except Exception as e:
        logging.warning(f"get_scleral_vein_density error: {e}")
        return np.nan

def get_ir_temperature(image):
    try:
        if image is None or image.size == 0: return np.nan
        return float(round(np.mean(image[:, :, 2]), 5))
    except Exception as e:
        logging.warning(f"get_ir_temperature error: {e}")
        return np.nan

def get_tear_film_reflectivity(image):
    try:
        if image is None or image.size == 0: return np.nan
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(round(np.std(gray), 5))
    except Exception as e:
        logging.warning(f"get_tear_film_reflectivity error: {e}")
        return np.nan

def get_sclera_color_balance(image):
    try:
        if image is None or image.size == 0: return np.nan
        r_mean = np.mean(image[:, :, 2])
        g_mean = np.mean(image[:, :, 1])
        if g_mean <= 0: return np.nan
        return float(round(r_mean / g_mean, 5))
    except Exception as e:
        logging.warning(f"get_sclera_color_balance error: {e}")
        return np.nan

def get_vein_pulsation_intensity(image):
    try:
        if image is None or image.size == 0: return np.nan
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pulsation = np.mean(cv2.Laplacian(gray, cv2.CV_64F))
        return float(round(pulsation, 5))
    except Exception as e:
        logging.warning(f"get_vein_pulsation_intensity error: {e}")
        return np.nan

def get_birefringence_index(image):
    try:
        if image is None or image.size == 0: return np.nan
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(round(np.var(gray) / 255.0, 5))
    except Exception as e:
        logging.warning(f"get_birefringence_index error: {e}")
        return np.nan

# ---------------------------
@dataclass
class EyeDetection:
    left_eye: Any
    right_eye: Any
    ir_intensity: float
    timestamp: datetime
    is_valid: bool
    eyes_open: bool
    face_rect: Optional[Tuple[int,int,int,int]] = None
    eye_roi: Optional[np.ndarray] = None

# ---------------------------
class EyeGlucoseMonitor:
    def __init__(self, model_path: str = "best_model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()

        # feature schema + means from training
        self.feature_order = list(FEATURES_ORDER)
        self.feature_means: Dict[str, float] = {}
        self._init_feature_info_from_model()

        # Try to init MediaPipe; fallback gracefully
        self.face_mesh = None
        try:
            import mediapipe as mp  # type: ignore
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logging.info("MediaPipe initialized.")
        except Exception as e:
            logging.warning(f"MediaPipe unavailable, continuing without it: {e}")
            self.face_mesh = None

        self.prediction_lock = threading.Lock()
        self.latest_smoothed_prediction = None
        self.latest_instantaneous_prediction = None
        self.last_instantaneous_prediction = None
        self.alpha = 0.009

        self.MIN_EYE_ASPECT_RATIO = 0.16
        self.MIN_IR_INTENSITY = 15

        self.time_history = deque(maxlen=200)
        self.instantaneous_history = deque(maxlen=200)
        self.smoothed_history = deque(maxlen=200)

        self.pupil_history = deque(maxlen=30)
        self._stop = False

    def _load_model(self) -> Any:
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                logging.info(f"Loaded model from {self.model_path}")
                return model
            except Exception as e:
                logging.error("Error loading model: " + str(e))
                return None
        else:
            logging.error("Model file not found at: " + self.model_path)
            return None

    def _init_feature_info_from_model(self):
        if self.model is None: return
        if hasattr(self.model, "feature_names_in_"):
            self.feature_order = list(self.model.feature_names_in_)
            logging.info(f"Using feature order from model: {self.feature_order}")
        try:
            preproc = self.model.named_steps.get("preprocessor") if hasattr(self.model, "named_steps") else None
            if preproc and hasattr(preproc, "named_steps") and "imputer" in preproc.named_steps:
                imputer = preproc.named_steps["imputer"]
                if hasattr(imputer, "statistics_") and self.feature_order:
                    self.feature_means = {n: float(s) for n, s in zip(self.feature_order, imputer.statistics_)}
                    logging.info(f"Loaded feature means from imputer: {self.feature_means}")
        except Exception as e:
            logging.warning(f"Could not extract feature means: {e}")

    def _fallback_eye_roi(self, frame: np.ndarray) -> np.ndarray:
        h, w, _ = frame.shape
        side = int(min(h, w) * 0.35)
        cx, cy = w // 2, h // 2
        x1, y1 = max(cx - side // 2, 0), max(cy - side // 2, 0)
        x2, y2 = min(x1 + side, w), min(y1 + side, h)
        return frame[y1:y2, x1:x2]

    def detect_face_and_eyes(self, frame: np.ndarray) -> EyeDetection:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_intensity = float(np.mean(gray))
        detection = EyeDetection([], [], ir_intensity, datetime.now(), False, False)

        if self.face_mesh is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    h, w, _ = frame.shape
                    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                    ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                    detection.face_rect = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
                    detection.is_valid = True

                    left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 246]
                    right_eye_indices = [362, 263, 387, 386, 385, 384, 398]

                    def gather_points(idxs):
                        pts=[]; 
                        for i in idxs:
                            if i < len(face_landmarks.landmark):
                                x = int(face_landmarks.landmark[i].x * w)
                                y = int(face_landmarks.landmark[i].y * h)
                                pts.append((x,y))
                        return pts

                    L = gather_points(left_eye_indices)
                    R = gather_points(right_eye_indices)

                    def bbox(points):
                        if not points: return None
                        xs = [p[0] for p in points]; ys=[p[1] for p in points]
                        return (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))

                    lb, rb = bbox(L), bbox(R)
                    detection.left_eye = [lb] if lb else []
                    detection.right_eye = [rb] if rb else []

                    ear_list=[]
                    for b in [lb, rb]:
                        if b:
                            _,_,ew,eh = b
                            ear_list.append(eh/ew if ew>0 else 0)
                    detection.eyes_open = (np.mean(ear_list) > self.MIN_EYE_ASPECT_RATIO) if ear_list else False

                    if lb or rb:
                        boxes=[b for b in [lb,rb] if b]
                        ex_min=min(b[0] for b in boxes); ey_min=min(b[1] for b in boxes)
                        ex_max=max(b[0]+b[2] for b in boxes); ey_max=max(b[1]+b[3] for b in boxes)
                        detection.eye_roi = frame[ey_min:ey_max, ex_min:ex_max]
                    else:
                        detection.eye_roi = self._fallback_eye_roi(frame)
                else:
                    detection.eye_roi = self._fallback_eye_roi(frame)
            except Exception as e:
                logging.warning(f"FaceMesh failed; using fallback ROI. {e}")
                detection.eye_roi = self._fallback_eye_roi(frame)
        else:
            detection.eye_roi = self._fallback_eye_roi(frame)

        return detection

    def update_pupil_history(self, pupil_size: float):
        if np.isfinite(pupil_size):
            self.pupil_history.append((time.time(), pupil_size))

    def compute_pupil_dilation_rate(self) -> float:
        if len(self.pupil_history) < 2: return np.nan
        t0, p0 = self.pupil_history[-2]
        t1, p1 = self.pupil_history[-1]
        dt = t1 - t0
        if dt <= 0: return np.nan
        return float(round((p1 - p0) / dt, 5))

    def compute_pupil_response_time(self) -> float:
        rate = self.compute_pupil_dilation_rate()
        if not np.isfinite(rate) or rate == 0: return np.nan
        return float(round(1.0 / abs(rate), 5))

    def extract_features(self, frame: np.ndarray):
        det = self.detect_face_and_eyes(frame)
        roi = det.eye_roi

        pupil_size = get_pupil_size(roi)
        self.update_pupil_history(pupil_size)

        feats = {
            "pupil_size": pupil_size,
            "sclera_redness": get_sclera_redness(roi),
            "vein_prominence": get_vein_prominence(roi),
            "pupil_response_time": self.compute_pupil_response_time(),
            "ir_intensity": get_ir_intensity(roi),
            "scleral_vein_density": get_scleral_vein_density(roi),
            "ir_temperature": get_ir_temperature(roi),
            "tear_film_reflectivity": get_tear_film_reflectivity(roi),
            "pupil_dilation_rate": self.compute_pupil_dilation_rate(),
            "sclera_color_balance": get_sclera_color_balance(roi),
            "vein_pulsation_intensity": get_vein_pulsation_intensity(roi),
            "birefringence_index": get_birefringence_index(roi)
        }

        ordered = {k: feats.get(k, np.nan) for k in self.feature_order}
        return ordered, det

    def _fill_with_means(self, row: Dict[str, float]) -> Dict[str, float]:
        cleaned = dict(row)
        finite_vals = [v for v in cleaned.values() if np.isfinite(v)]
        fallback = float(np.mean(finite_vals)) if finite_vals else (
            float(np.mean(list(self.feature_means.values()))) if self.feature_means else 0.0
        )
        out = {}
        for name in self.feature_order:
            v = cleaned.get(name, np.nan)
            if np.isfinite(v):
                out[name] = float(v)
            elif name in self.feature_means:
                out[name] = self.feature_means[name]  # training mean
            else:
                out[name] = fallback               # never 0 unless everything is missing
        return out

    def predict_glucose(self, features: Dict):
        result = None
        try:
            if self.model is not None and features:
                filled = self._fill_with_means(features)
                df = pd.DataFrame([filled], columns=self.feature_order)
                pred = self.model.predict(df)
                if len(pred) > 0 and np.isfinite(pred[0]):
                    result = float(pred[0])
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
                roc = result - self.last_instantaneous_prediction
                if abs(roc) > 5:
                    logging.info(f"High rate of change: {roc:.2f} mg/dL")
            self.last_instantaneous_prediction = result

    def run(self):
        plt.ion()
        fig, ax = plt.subplots()
        line_inst, = ax.plot([], [], label="Instantaneous", color="green")
        line_avg, = ax.plot([], [], label="Smoothed", color="orange")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Blood Glucose (mg/dL)"); ax.legend()

        def _on_close(_): self._stop.__setattr__("val", True)
        self._stop = type("Stop", (), {"val": False})()
        fig.canvas.mpl_connect("close_event", _on_close)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam."); return
        print("Webcam successfully opened.")

        start = time.time()
        try:
            while True:
                if self._stop.val: break
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("No frame received from the webcam."); break

                feats, det = self.extract_features(frame)
                self.predict_glucose(feats)

                t = time.time() - start
                inst = self.latest_instantaneous_prediction if self.latest_instantaneous_prediction is not None else np.nan
                avg  = self.latest_smoothed_prediction if self.latest_smoothed_prediction is not None else np.nan
                self.time_history.append(t); self.instantaneous_history.append(inst); self.smoothed_history.append(avg)

                if det.face_rect is not None:
                    x,y,w,h = det.face_rect
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                for bx in det.left_eye or []:
                    if bx: ex,ey,ew,eh = bx; cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                for bx in det.right_eye or []:
                    if bx: ex,ey,ew,eh = bx; cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,0,255), 2)

                # plot
                line_inst.set_data(self.time_history, self.instantaneous_history)
                line_avg.set_data(self.time_history, self.smoothed_history)
                ax.relim(); ax.autoscale_view()
                fig.canvas.draw(); fig.canvas.flush_events()

                # HUD
                inst_txt = f"Inst: {inst:.1f} mg/dL" if np.isfinite(inst) else "Inst: No Reading"
                avg_txt  = f"Avg: {avg:.1f} mg/dL"  if np.isfinite(avg)  else "Avg: No Reading"
                cv2.putText(frame, inst_txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if np.isfinite(inst) else (0,0,255), 2)
                cv2.putText(frame, avg_txt,  (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0) if np.isfinite(avg) else (0,0,255), 2)

                if np.isfinite(inst) and (inst < 40 or inst > 400):
                    cv2.putText(frame, "Instantaneous out of range.", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                if np.isfinite(avg) and (avg < 40 or avg > 400):
                    cv2.putText(frame, "Average out of range.", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                cv2.imshow("Blood Glucose", frame)
                key = cv2.waitKey(1)
                if key in (ord('q'), 27) or cv2.getWindowProperty("Blood Glucose", cv2.WND_PROP_VISIBLE) < 1:
                    break
                plt.pause(0.001)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            plt.ioff(); plt.show()

if __name__ == "__main__":
    import sys
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "best_model.pkl"
    EyeGlucoseMonitor(model_path=model_arg).run()
