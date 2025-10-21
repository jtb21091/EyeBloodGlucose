from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2, numpy as np, joblib, io, os, requests
from typing import Dict, Optional, List
import logging
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Download model from GitHub Release if not present
MODEL_PATH = "best_model.pkl"
MODEL_URL = "https://github.com/jtb21091/EyeBloodGlucose/releases/download/v1.0.0/best_model.pkl"

# UPDATED FEATURES ORDER - 15 features to match new training.py
FEATURES_ORDER = [
    'pupil_size', 'sclera_redness', 'vein_prominence', 'capture_duration', 'ir_intensity',
    'scleral_vein_density', 'ir_temperature', 'tear_film_reflectivity',
    'sclera_color_balance', 'vein_pulsation_intensity', 'birefringence_index',
    'lens_clarity_score', 'sclera_yellowness', 'vessel_tortuosity', 'image_quality_score'
]

# ===== FEATURE ENGINEER CLASS - MUST BE DEFINED BEFORE MODEL LOADING =====
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
# ===== END OF FEATURE ENGINEER CLASS =====

def download_model():
    """Download model from GitHub Release if not present locally"""
    if not os.path.exists(MODEL_PATH):
        logging.info(f"Model not found locally. Downloading from {MODEL_URL}...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logging.info(f"Model downloaded successfully to {MODEL_PATH}")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Could not download model from GitHub Release: {e}")
    else:
        logging.info(f"Model found locally at {MODEL_PATH}")

# Download and load model on startup
download_model()
model = joblib.load(MODEL_PATH)
logging.info("Model loaded successfully")

@app.get("/")
async def read_root():
    """Serve the index.html file"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "EyeBloodGlucose API - use /predict endpoint"}

# Feature extraction functions
def get_pupil_size(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (7, 7), 0)
        circ = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=45, param2=18, minRadius=6, maxRadius=90)
        if circ is None: return np.nan
        circ = np.round(circ[0, :]).astype("int")
        return float(np.mean([r for (_, _, r) in circ]))
    except: return np.nan

def get_sclera_redness(img):
    try:
        if img is None or img.size == 0: return np.nan
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0,70,50), (10,255,255))
        m2 = cv2.inRange(hsv, (170,70,50), (180,255,255))
        m = m1 | m2
        if m.size == 0: return np.nan
        return float(round(cv2.countNonZero(m)/m.size*100.0, 5))
    except: return np.nan

def get_vein_prominence(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, 40, 120)
        if e.size == 0: return np.nan
        return float(round(np.sum(e)/(255.0*e.size)*10, 5))
    except: return np.nan

def get_ir_intensity(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(round(np.mean(g), 5))
    except: return np.nan

def get_scleral_vein_density(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, 40, 120)
        if e.size == 0: return np.nan
        return float(round(np.sum(e)/(255.0*e.size), 5))
    except: return np.nan

def get_ir_temperature(img):
    try:
        if img is None or img.size == 0: return np.nan
        return float(round(np.mean(img[:,:,2]), 5))
    except: return np.nan

def get_tear_film_reflectivity(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(round(np.std(g), 5))
    except: return np.nan

def get_sclera_color_balance(img):
    try:
        if img is None or img.size == 0: return np.nan
        r = np.mean(img[:,:,2]); g = np.mean(img[:,:,1])
        if g <= 0: return np.nan
        return float(round(r/g, 5))
    except: return np.nan

def get_vein_pulsation_intensity(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(round(np.mean(cv2.Laplacian(g, cv2.CV_64F)), 5))
    except: return np.nan

def get_birefringence_index(img):
    try:
        if img is None or img.size == 0: return np.nan
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(round(np.var(g)/255.0, 5))
    except: return np.nan

# NEW FEATURE EXTRACTORS (matching prediction.py)
def get_lens_clarity_score(img):
    try:
        if img is None or img.size == 0: return np.nan
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_y_start, center_y_end = h // 3, 2 * h // 3
        center_x_start, center_x_end = w // 3, 2 * w // 3
        center = gray[center_y_start:center_y_end, center_x_start:center_x_end]
        
        if center.size == 0:
            return np.nan
        
        clarity = np.std(center) / (np.mean(center) + 1e-5)
        return float(round(clarity, 5))
    except: return np.nan

def get_sclera_yellowness(img):
    try:
        if img is None or img.size == 0: return np.nan
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        b_channel = lab[:, :, 2]
        yellowness = np.mean(b_channel)
        return float(round(yellowness, 5))
    except: return np.nan

def get_vessel_tortuosity(img):
    try:
        if img is None or img.size == 0: return np.nan
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(filtered, 30, 90)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.0
        
        tortuosity_scores = []
        for contour in contours:
            if len(contour) > 10:
                arc_length = cv2.arcLength(contour, False)
                if len(contour) >= 2:
                    start_point = contour[0][0]
                    end_point = contour[-1][0]
                    chord_length = np.linalg.norm(start_point - end_point)
                    
                    if chord_length > 0:
                        tortuosity = arc_length / (chord_length + 1e-5)
                        tortuosity_scores.append(tortuosity)
        
        if tortuosity_scores:
            mean_tortuosity = np.mean(tortuosity_scores)
            return float(round(mean_tortuosity, 5))
        else:
            return 0.0
    except: return np.nan

def get_image_quality_score(img):
    try:
        if img is None or img.size == 0: return np.nan
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(img)
        contrast = np.std(img)
        
        blur_score = min(blur_var / 100.0, 1.0)
        brightness_score = 1.0 - abs(brightness - 128) / 128.0
        contrast_score = min(contrast / 50.0, 1.0)
        
        quality = (blur_score + brightness_score + contrast_score) / 3.0 * 100
        
        return float(round(quality, 5))
    except: return np.nan

def detect_eyes(frame):
    """Detect eyes and return rectangles"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(60,60))
    rects = []
    
    if len(faces):
        fx, fy, fw, fh = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        roi = gray[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.05, minNeighbors=2, minSize=(16,16))
        
        for (ex,ey,ew,eh) in eyes:
            rects.append((fx+ex, fy+ey, ew, eh))
    
    # Sort left to right and take top 2
    rects.sort(key=lambda r: r[0])
    return rects[:2]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid image"}
    
    # Detect eyes
    eye_rects = detect_eyes(frame)
    
    # STRICT: Only predict if exactly 2 eyes detected
    if len(eye_rects) != 2:
        return {
            "estimate_mg_dl": None,
            "mode": "eyes_closed",
            "eyes": []
        }
    
    # Extract ROI around both eyes
    h, w = frame.shape[:2]
    xs = [x for (x,_,_,_) in eye_rects]
    ys = [y for (_,y,_,_) in eye_rects]
    xe = [x+w_ for (x,_,w_,_) in eye_rects]
    ye = [y+h_ for (_,y,_,h_) in eye_rects]
    
    x1, y1, x2, y2 = max(0, min(xs)-10), max(0, min(ys)-10), min(w, max(xe)+10), min(h, max(ye)+10)
    
    if x2 <= x1 or y2 <= y1:
        return {"estimate_mg_dl": None, "mode": "detection_failed", "eyes": []}
    
    roi = frame[y1:y2, x1:x2]
    
    # Extract ALL 15 features
    feats = {
        "pupil_size": get_pupil_size(roi),
        "sclera_redness": get_sclera_redness(roi),
        "vein_prominence": get_vein_prominence(roi),
        "capture_duration": np.nan,  # Can't measure in real-time
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
    
    # Fill NaNs with model means
    pre = model.named_steps["preprocessor"]
    means = {n: float(s) for n, s in zip(FEATURES_ORDER, pre.named_steps["imputer"].statistics_)}
    
    for k in FEATURES_ORDER:
        if not np.isfinite(feats.get(k, np.nan)):
            feats[k] = means.get(k, 100.0)
    
    # Predict
    df = pd.DataFrame([feats], columns=FEATURES_ORDER)
    pred = float(model.predict(df)[0])
    
    # Scale eye coordinates to 640x480 for frontend
    scale_x = 640 / w
    scale_y = 480 / h
    
    eyes_scaled = [
        {
            "x": int(x * scale_x),
            "y": int(y * scale_y),
            "w": int(ew * scale_x),
            "h": int(eh * scale_y)
        }
        for (x, y, ew, eh) in eye_rects
    ]
    
    return {
        "estimate_mg_dl": round(pred, 1),
        "mode": "both_eyes",
        "eyes": eyes_scaled
    }