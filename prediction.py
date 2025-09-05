import os, cv2, numpy as np, pandas as pd, joblib, logging, time, json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURES_ORDER = [
    'pupil_size','sclera_redness','vein_prominence','pupil_response_time','ir_intensity',
    'scleral_vein_density','ir_temperature','tear_film_reflectivity','pupil_dilation_rate',
    'sclera_color_balance','vein_pulsation_intensity','birefringence_index'
]

# ---------- feature extractors (return NaN on failure) ----------
def get_pupil_size(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (7, 7), 0)
        circ = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=80)
        if circ is None: return np.nan
        circ = np.round(circ[0, :]).astype("int")
        return float(np.mean([r for (_, _, r) in circ]))
    except Exception:
        return np.nan

def get_sclera_redness(img):
    try:
        if img is None or img.size == 0: return np.nan
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0,70,50), (10,255,255))
        m2 = cv2.inRange(hsv, (170,70,50), (180,255,255))
        m = m1 | m2
        if m.size == 0: return np.nan
        return float(round(cv2.countNonZero(m)/m.size*100.0, 5))
    except Exception:
        return np.nan

def get_vein_prominence(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, 50, 150)
        if e.size == 0: return np.nan
        return float(round(np.sum(e)/(255.0*e.size)*10, 5))
    except Exception:
        return np.nan

def get_ir_intensity(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(round(np.mean(g), 5))
    except Exception:
        return np.nan

def get_scleral_vein_density(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, 50, 150)
        if e.size == 0: return np.nan
        return float(round(np.sum(e)/(255.0*e.size), 5))
    except Exception:
        return np.nan

def get_ir_temperature(img):
    try:
        if img is None or img.size == 0: return np.nan
        return float(round(np.mean(img[:,:,2]), 5))
    except Exception:
        return np.nan

def get_tear_film_reflectivity(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(round(np.std(g), 5))
    except Exception:
        return np.nan

def get_sclera_color_balance(img):
    try:
        if img is None or img.size == 0: return np.nan
        r = np.mean(img[:,:,2]); g = np.mean(img[:,:,1])
        if g <= 0: return np.nan
        return float(round(r/g, 5))
    except Exception:
        return np.nan

def get_vein_pulsation_intensity(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(round(np.mean(cv2.Laplacian(g, cv2.CV_64F)), 5))
    except Exception:
        return np.nan

def get_birefringence_index(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(round(np.var(g)/255.0, 5))
    except Exception:
        return np.nan

@dataclass
class Detection:
    roi: Optional[np.ndarray] = None

class EyeGlucoseMonitor:
    def __init__(self, model_path="best_model.pkl"):
        self.model = self._load_model(model_path)
        if not hasattr(self.model, "feature_names_in_"):
            raise SystemExit("Model is missing feature_names_in_. Retrain with provided training.py.")
        if list(self.model.feature_names_in_) != FEATURES_ORDER:
            raise SystemExit(f"Model features {list(self.model.feature_names_in_)} do not match the required 12 features. Retrain.")

        # Extract training means for fill
        pre = self.model.named_steps["preprocessor"]
        stats = pre.named_steps["imputer"].statistics_
        self.feature_means = {n: float(s) for n, s in zip(FEATURES_ORDER, stats)}

        self.time = deque(maxlen=200)
        self.inst = deque(maxlen=200)
        self.avg = deque(maxlen=200)
        self.smooth = None
        self.alpha = 0.009

    def _load_model(self, path):
        if not os.path.exists(path): raise SystemExit(f"Model not found: {path}")
        m = joblib.load(path)
        logging.info(f"Loaded model from {path}")
        return m

    def _fallback_roi(self, frame):
        h,w,_ = frame.shape
        side = int(min(h,w)*0.35)
        cx,cy = w//2, h//2
        x1,y1 = max(cx-side//2,0), max(cy-side//2,0)
        x2,y2 = min(x1+side,w), min(y1+side,h)
        return frame[y1:y2, x1:x2]

    def detect(self, frame):
        return Detection(roi=self._fallback_roi(frame))

    def extract_features(self, frame):
        roi = self.detect(frame).roi
        feats = {
            "pupil_size": get_pupil_size(roi),
            "sclera_redness": get_sclera_redness(roi),
            "vein_prominence": get_vein_prominence(roi),
            "pupil_response_time": np.nan,   # single-frame fallback
            "ir_intensity": get_ir_intensity(roi),
            "scleral_vein_density": get_scleral_vein_density(roi),
            "ir_temperature": get_ir_temperature(roi),
            "tear_film_reflectivity": get_tear_film_reflectivity(roi),
            "pupil_dilation_rate": np.nan,   # single-frame fallback
            "sclera_color_balance": get_sclera_color_balance(roi),
            "vein_pulsation_intensity": get_vein_pulsation_intensity(roi),
            "birefringence_index": get_birefringence_index(roi)
        }
        return {k: feats.get(k, np.nan) for k in FEATURES_ORDER}

    def _fill(self, row: Dict[str,float]) -> Dict[str,float]:
        finite = [v for v in row.values() if np.isfinite(v)]
        fallback = float(np.mean(finite)) if finite else float(np.mean(list(self.feature_means.values())))
        return {k: (float(row[k]) if np.isfinite(row[k]) else self.feature_means.get(k, fallback)) for k in FEATURES_ORDER}

    def predict_once(self, feats: Dict[str,float]) -> float:
        filled = self._fill(feats)
        df = pd.DataFrame([filled], columns=FEATURES_ORDER)
        return float(self.model.predict(df)[0])

    def run(self):
        plt.ion()
        fig, ax = plt.subplots()
        li, = ax.plot([], [], label="Instantaneous", color="green")
        ls, = ax.plot([], [], label="Smoothed", color="orange")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Blood Glucose (mg/dL)"); ax.legend()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam."); return
        print("Webcam successfully opened.")
        t0 = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                feats = self.extract_features(frame)
                y = self.predict_once(feats)
                self.smooth = y if self.smooth is None else self.alpha*y + (1-self.alpha)*self.smooth

                t = time.time()-t0
                self.time.append(t); self.inst.append(y); self.avg.append(self.smooth)

                cv2.putText(frame, f"Inst: {y:.1f} mg/dL", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                cv2.putText(frame, f"Avg: {self.smooth:.1f} mg/dL", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
                cv2.imshow("Blood Glucose", frame)

                li.set_data(self.time, self.inst); ls.set_data(self.time, self.avg)
                ax.relim(); ax.autoscale_view()
                fig.canvas.draw(); fig.canvas.flush_events()

                if cv2.waitKey(1) in (ord('q'), 27) or cv2.getWindowProperty("Blood Glucose", cv2.WND_PROP_VISIBLE) < 1:
                    break
                plt.pause(0.001)
        finally:
            cap.release(); cv2.destroyAllWindows()
            plt.ioff(); plt.show()

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "best_model.pkl"
    EyeGlucoseMonitor(model).run()
