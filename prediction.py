# prediction.py
# Run with:  python prediction.py best_model.pkl

import os, cv2, numpy as np, pandas as pd, joblib, logging, time
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# UPDATED FEATURES ORDER - 15 features now
FEATURES_ORDER = [
    'pupil_size', 'sclera_redness', 'vein_prominence', 'capture_duration', 'ir_intensity',
    'scleral_vein_density', 'ir_temperature', 'tear_film_reflectivity',
    'sclera_color_balance', 'vein_pulsation_intensity', 'birefringence_index',
    'lens_clarity_score', 'sclera_yellowness', 'vessel_tortuosity', 'image_quality_score'
]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates engineered features:
    - Ratios between key features
    - Polynomial interactions
    - Domain-specific combinations
    """
    def __init__(self, add_ratios: bool = True, add_interactions: bool = True, add_polynomials: bool = True):
        self.add_ratios = add_ratios
        self.add_interactions = add_interactions
        self.add_polynomials = add_polynomials
        self.feature_names_: Optional[List[str]] = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X, columns=FEATURES_ORDER)
        X_new = X_df.copy()
        
        if self.add_ratios:
            # Vascular ratios
            X_new['vein_to_redness_ratio'] = X_df['vein_prominence'] / (X_df['sclera_redness'] + 1e-6)
            X_new['density_to_prominence_ratio'] = X_df['scleral_vein_density'] / (X_df['vein_prominence'] + 1e-6)
            
            # Optical ratios
            X_new['ir_to_reflectivity_ratio'] = X_df['ir_intensity'] / (X_df['tear_film_reflectivity'] + 1e-6)
            X_new['clarity_to_yellowness_ratio'] = X_df['lens_clarity_score'] / (X_df['sclera_yellowness'] + 1e-6)
            
            # Pupil ratios
            X_new['pupil_to_duration_ratio'] = X_df['pupil_size'] / (X_df['capture_duration'] + 1e-6)
            
            # Quality ratios
            X_new['quality_to_tortuosity_ratio'] = X_df['image_quality_score'] / (X_df['vessel_tortuosity'] + 1e-6)
        
        if self.add_interactions:
            # Vascular health composite
            X_new['vascular_health_index'] = (
                X_df['vein_prominence'] * X_df['scleral_vein_density'] * X_df['vein_pulsation_intensity']
            ) ** (1/3)  # Geometric mean
            
            # Scleral quality composite
            X_new['scleral_quality_index'] = (
                X_df['sclera_redness'] * X_df['sclera_color_balance'] * X_df['sclera_yellowness']
            ) ** (1/3)
            
            # IR composite
            X_new['ir_thermal_index'] = X_df['ir_intensity'] * X_df['ir_temperature']
            
            # Optical clarity composite
            X_new['optical_clarity_index'] = (
                X_df['lens_clarity_score'] * X_df['tear_film_reflectivity'] * X_df['birefringence_index']
            ) ** (1/3)
            
            # Pupil response composite
            X_new['pupil_response_index'] = X_df['pupil_size'] * X_df['capture_duration']
            
            # Vessel dynamics
            X_new['vessel_dynamics_index'] = X_df['vessel_tortuosity'] * X_df['vein_pulsation_intensity']
        
        if self.add_polynomials:
            # Square key features that may have non-linear relationships
            key_features = ['pupil_size', 'sclera_redness', 'vein_prominence', 'ir_intensity', 'lens_clarity_score']
            for feat in key_features:
                if feat in X_df.columns:
                    X_new[f'{feat}_squared'] = X_df[feat] ** 2
        
        self.feature_names_ = list(X_new.columns)
        return X_new.values
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return self.feature_names_ if self.feature_names_ is not None else FEATURES_ORDER

# ---------- feature extractors (return NaN on failure) ----------
def get_pupil_size(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (7, 7), 0)
        circ = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                 param1=45, param2=18, minRadius=6, maxRadius=90)
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
        e = cv2.Canny(g, 40, 120)
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
        e = cv2.Canny(g, 40, 120)
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

# ---------- NEW FEATURE EXTRACTORS ----------
def get_lens_clarity_score(img):
    """
    Measure lens opacity/clarity score.
    Higher values indicate more clarity/less opacity.
    """
    try:
        if img is None or img.size == 0: return np.nan
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Focus on central region (approximate lens area)
        center_y_start, center_y_end = h // 3, 2 * h // 3
        center_x_start, center_x_end = w // 3, 2 * w // 3
        center = gray[center_y_start:center_y_end, center_x_start:center_x_end]
        
        if center.size == 0:
            return np.nan
        
        clarity = np.std(center) / (np.mean(center) + 1e-5)
        return float(round(clarity, 5))
    except Exception:
        return np.nan

def get_sclera_yellowness(img):
    """
    Measure yellowish tint in sclera using LAB color space.
    The b channel in LAB represents yellow-blue axis.
    """
    try:
        if img is None or img.size == 0: return np.nan
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        b_channel = lab[:, :, 2]  # Yellow-blue axis (higher = more yellow)
        yellowness = np.mean(b_channel)
        return float(round(yellowness, 5))
    except Exception:
        return np.nan

def get_vessel_tortuosity(img):
    """
    Estimate blood vessel tortuosity (twistedness) using edge curvature analysis.
    High glucose levels can increase vessel tortuosity.
    """
    try:
        if img is None or img.size == 0: return np.nan
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect edges with stricter thresholds for vessel detection
        edges = cv2.Canny(filtered, 30, 90)
        
        # Find contours representing vessels
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.0
        
        # Calculate tortuosity for each contour
        tortuosity_scores = []
        for contour in contours:
            if len(contour) > 10:  # Need sufficient points
                # Calculate arc length
                arc_length = cv2.arcLength(contour, False)
                
                # Calculate chord length (straight line distance)
                if len(contour) >= 2:
                    start_point = contour[0][0]
                    end_point = contour[-1][0]
                    chord_length = np.linalg.norm(start_point - end_point)
                    
                    # Tortuosity = arc_length / chord_length
                    if chord_length > 0:
                        tortuosity = arc_length / (chord_length + 1e-5)
                        tortuosity_scores.append(tortuosity)
        
        if tortuosity_scores:
            mean_tortuosity = np.mean(tortuosity_scores)
            return float(round(mean_tortuosity, 5))
        else:
            return 0.0
    except Exception:
        return np.nan

def get_image_quality_score(img):
    """
    Calculate composite image quality metric.
    Combines blur, brightness, and contrast into a single score.
    """
    try:
        if img is None or img.size == 0: return np.nan
        
        # Blur detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness and contrast
        brightness = np.mean(img)
        contrast = np.std(img)
        
        # Normalize each component
        blur_score = min(blur_var / 100.0, 1.0)
        brightness_score = 1.0 - abs(brightness - 128) / 128.0
        contrast_score = min(contrast / 50.0, 1.0)
        
        # Composite quality (0-100 scale)
        quality = (blur_score + brightness_score + contrast_score) / 3.0 * 100
        
        return float(round(quality, 5))
    except Exception:
        return np.nan

# ---------- improved adaptive preproc ----------
def to_gray_clahe(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(g)

# ---------- improved detector ----------
class EyeDetector:
    """
    Enhanced detector with better tolerance for glasses and varied lighting.
    Prefers both eyes; accepts one eye if needed. Returns 1-2 tight rects (x,y,w,h).
    """
    def __init__(self):
        base = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(os.path.join(base, "haarcascade_frontalface_default.xml"))
        self.eye  = cv2.CascadeClassifier(os.path.join(base, "haarcascade_eye.xml"))
        self.glasses = cv2.CascadeClassifier(os.path.join(base, "haarcascade_eye_tree_eyeglasses.xml"))
        if self.face.empty() or (self.eye.empty() and self.glasses.empty()):
            logging.warning("Haar cascades not found; detection limited.")

    def _trim_box(self, x,y,w,h, shrink=0.2):
        dx, dy = int(w*shrink), int(h*shrink)
        return (x+dx, y+dy, max(1, w-2*dx), max(1, h-2*dy))

    def _detect_in_face(self, gray):
        faces = self.face.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(60,60))
        rects = []
        if len(faces):
            fx, fy, fw, fh = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
            roi = gray[fy:fy+fh, fx:fx+fw]
            cand = []
            if not self.glasses.empty():
                cand += list(self.glasses.detectMultiScale(roi, scaleFactor=1.03, minNeighbors=2, minSize=(16,16)))
            if not self.eye.empty():
                cand += list(self.eye.detectMultiScale(roi, scaleFactor=1.05, minNeighbors=2, minSize=(16,16)))
            for (ex,ey,ew,eh) in cand:
                rects.append((fx+ex, fy+ey, ew, eh))
        return rects

    def _detect_full(self, gray):
        rects = []
        if not self.glasses.empty():
            rects += list(self.glasses.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=2, minSize=(14,14)))
        if not self.eye.empty():
            rects += list(self.eye.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(14,14)))
        return rects

    def _pupil_hints(self, gray):
        g = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25, 
                                   param1=35, param2=14, minRadius=5, maxRadius=70)
        rects = []
        if circles is not None:
            c = np.uint16(np.around(circles))[0]
            c = sorted(c, key=lambda k: k[2], reverse=True)[:2]
            for (x,y,r) in c:
                s = int(r*3.0)
                rects.append((int(x-s//2), int(y-s//2), s, s))
        return rects

    def detect_eyes(self, frame) -> List[Tuple[int,int,int,int]]:
        if frame is None or frame.size == 0: return []
        gray = to_gray_clahe(frame)

        rects = self._detect_in_face(gray)
        if len(rects) < 1: rects = self._detect_full(gray)
        if len(rects) < 1: rects = self._pupil_hints(gray)

        h, w = gray.shape[:2]
        norm = []
        for (x,y,ww,hh) in rects:
            x1,y1 = max(0,x), max(0,y)
            x2,y2 = min(w, x+ww), min(h, y+hh)
            if x2>x1 and y2>y1:
                norm.append((x1,y1,x2-x1,y2-y1))
        norm.sort(key=lambda r: r[2]*r[3], reverse=True)
        norm = norm[:2]
        norm = [self._trim_box(*r, shrink=0.18) for r in norm]
        if len(norm) == 2 and norm[0][0] > norm[1][0]:
            norm = [norm[1], norm[0]]
        return norm

def draw_targets(frame, rects: List[Tuple[int,int,int,int]], show=True):
    if not show: return
    for (x,y,w,h) in rects:
        cx, cy = x + w//2, y + h//2
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        cv2.line(frame, (cx-12, cy), (cx+12, cy), (0,255,0), 2)
        cv2.line(frame, (cx, cy-12), (cx, cy+12), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)

def union_crop(frame, rects: List[Tuple[int,int,int,int]], pad_ratio: float=0.12) -> Optional[np.ndarray]:
    """Small image around eyes (tight union, small padding)."""
    if frame is None or len(rects) == 0: return None
    h, w, _ = frame.shape
    xs = [x for (x,_,_,_) in rects]; ys = [y for (_,y,_,_) in rects]
    xe = [x+w_ for (x,_,w_,_) in rects]; ye = [y+h_ for (_,y,_,h_) in rects]
    x1, y1, x2, y2 = min(xs), min(ys), max(xe), max(ye)
    pw = int((x2-x1) * pad_ratio); ph = int((y2-y1) * pad_ratio)
    x1 = max(0, x1 - pw); y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw); y2 = min(h, y2 + ph)
    if x2<=x1 or y2<=y1: return None
    return frame[y1:y2, x1:x2]

@dataclass
class Detection:
    eye_rects: List[Tuple[int,int,int,int]]
    roi: Optional[np.ndarray]

class EyeGlucoseMonitor:
    def __init__(self, model_path="best_model.pkl"):
        self.model = self._load_model(model_path)
        
        # Load schema for feature information
        schema_path = "model_schema.json"
        if os.path.exists(schema_path):
            import json
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            model_features = schema.get('features', FEATURES_ORDER)
        elif hasattr(self.model, "feature_names_in_"):
            model_features = list(self.model.feature_names_in_)
        else:
            logging.warning("Using default features order - model may not work correctly")
            model_features = FEATURES_ORDER

        # Use preprocessor statistics if available, otherwise use default values
        try:
            pre = self.model.named_steps["preprocessor"]
            stats = pre.named_steps["imputer"].statistics_
            self.feature_means = {n: float(s) for n, s in zip(FEATURES_ORDER, stats)}
        except (AttributeError, KeyError):
            # Default feature means if not available from model
            self.feature_means = {f: 50.0 for f in FEATURES_ORDER}

        self.time = deque(maxlen=200)
        self.inst = deque(maxlen=200)
        self.avg = deque(maxlen=200)
        self.smooth = None
        self.alpha = 0.009
        self._stop = False

        self.detector = EyeDetector()
        self.show_overlay = True

        # STRICT MODE: require both eyes
        self.no_eye_grace = 2
        self.hit_streak = 0
        self.miss_streak = 0

        self.eyes_mode = "none"
        self.last_rects: List[Tuple[int,int,int,int]] = []

    def _load_model(self, path):
        if not os.path.exists(path): raise SystemExit(f"Model not found: {path}")
        m = joblib.load(path)
        logging.info(f"Loaded model from {path}")
        return m

    def detect(self, frame) -> Detection:
        rects = self.detector.detect_eyes(frame)
        n = len(rects)

        # STRICT MODE: Only accept 2 eyes
        if n == 2:
            self.hit_streak += 1
            self.miss_streak = 0
            self.eyes_mode = "both"
            self.last_rects = rects
            roi = union_crop(frame, rects, pad_ratio=0.12)
            return Detection(eye_rects=rects, roi=roi)
        else:
            # No prediction if not exactly 2 eyes
            self.hit_streak = 0
            self.miss_streak += 1
            if self.miss_streak >= self.no_eye_grace:
                self.eyes_mode = "none"
            return Detection(eye_rects=[], roi=None)

    def extract_features(self, roi: Optional[np.ndarray]) -> Dict[str,float]:
        if roi is None:
            return {k: np.nan for k in FEATURES_ORDER}
        
        feats = {
            "pupil_size": get_pupil_size(roi),
            "sclera_redness": get_sclera_redness(roi),
            "vein_prominence": get_vein_prominence(roi),
            "capture_duration": np.nan,  # Can't measure this in real-time
            "ir_intensity": get_ir_intensity(roi),
            "scleral_vein_density": get_scleral_vein_density(roi),
            "ir_temperature": get_ir_temperature(roi),
            "tear_film_reflectivity": get_tear_film_reflectivity(roi),
            "sclera_color_balance": get_sclera_color_balance(roi),
            "vein_pulsation_intensity": get_vein_pulsation_intensity(roi),
            "birefringence_index": get_birefringence_index(roi),
            "lens_clarity_score": get_lens_clarity_score(roi),
            "sclera_yellowness": get_sclera_yellowness(roi),
            "vessel_tortuosity": get_vessel_tortuosity(roi),
            "image_quality_score": get_image_quality_score(roi)
        }
        return {k: feats.get(k, np.nan) for k in FEATURES_ORDER}

    def _fill(self, row: Dict[str,float]) -> Dict[str,float]:
        finite = [v for v in row.values() if np.isfinite(v)]
        fallback = float(np.mean(finite)) if finite else float(np.mean(list(self.feature_means.values())))
        return {k: (float(row[k]) if np.isfinite(row[k]) else self.feature_means.get(k, fallback)) for k in FEATURES_ORDER}

    def predict_once(self, feats: Dict[str,float]) -> float:
        df = pd.DataFrame([self._fill(feats)], columns=FEATURES_ORDER)
        return float(self.model.predict(df)[0])

    def run(self):
        plt.ion()
        fig, ax = plt.subplots()
        li, = ax.plot([], [], label="Instantaneous", color="green")
        ls, = ax.plot([], [], label="Smoothed", color="orange")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Blood Glucose (mg/dL)"); ax.legend()

        def _on_close(event): self._stop = True
        def _on_key(event):
            if event.key in ("escape", "q"): self._stop = True
            elif event.key == "d": self.show_overlay = not self.show_overlay
        fig.canvas.mpl_connect("close_event", _on_close)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam.")
            plt.close(fig); return

        print("Webcam opened. Press 'd' to toggle overlay, 'q' to quit.")
        print("STRICT MODE: Both eyes required for prediction")
        t0 = time.time()
        smooth = None
        alpha = self.alpha

        try:
            while True:
                if self._stop or not plt.fignum_exists(fig.number): break
                ok, frame = cap.read()
                if not ok: break

                det = self.detect(frame)

                # STRICT: only predict with both eyes
                if det.roi is not None and self.eyes_mode == "both" and len(det.eye_rects) == 2:
                    draw_targets(frame, det.eye_rects, self.show_overlay)
                    feats = self.extract_features(det.roi)
                    y = self.predict_once(feats)
                    smooth = y if smooth is None else alpha*y + (1-alpha)*smooth

                    t = time.time() - t0
                    self.time.append(t); self.inst.append(y); self.avg.append(smooth)

                    cv2.putText(frame, "Both eyes detected", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                    cv2.putText(frame, f"Inst: {y:.1f} mg/dL", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    cv2.putText(frame, f"Avg:  {smooth:.1f} mg/dL", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
                else:
                    msg = "EYES CLOSED OR NOT DETECTED - No prediction"
                    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (10, 10), (10+tw+10, 10+th+10), (0,0,255), -1)
                    cv2.putText(frame, msg, (15, 10+th+3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.putText(frame, f"Overlay: {'ON' if self.show_overlay else 'OFF'} (press 'd')",
                            (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                cv2.imshow("Blood Glucose", frame)

                li.set_data(self.time, self.inst); ls.set_data(self.time, self.avg)
                ax.relim(); ax.autoscale_view()
                fig.canvas.draw(); fig.canvas.flush_events()

                if (cv2.waitKey(1) in (ord('q'), 27)) or (cv2.getWindowProperty("Blood Glucose", cv2.WND_PROP_VISIBLE) < 1):
                    break
                plt.pause(0.001)
        finally:
            cap.release(); cv2.destroyAllWindows()
            try: plt.ioff(); plt.close(fig)
            except Exception: pass

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "best_model.pkl"
    EyeGlucoseMonitor(model).run()