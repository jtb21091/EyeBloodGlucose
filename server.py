from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, joblib, io
from typing import Dict

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = joblib.load("best_model.pkl")

FEATURES_ORDER = [
    'pupil_size','sclera_redness','vein_prominence','pupil_response_time','ir_intensity',
    'scleral_vein_density','ir_temperature','tear_film_reflectivity','pupil_dilation_rate',
    'sclera_color_balance','vein_pulsation_intensity','birefringence_index'
]

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

def detect_eyes(frame):
    """Detect eyes and return rectangles"""
    base = cv2.data.haarcascades
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
    
    # Extract features
    feats = {
        "pupil_size": get_pupil_size(roi),
        "sclera_redness": get_sclera_redness(roi),
        "vein_prominence": get_vein_prominence(roi),
        "pupil_response_time": np.nan,
        "ir_intensity": get_ir_intensity(roi),
        "scleral_vein_density": get_scleral_vein_density(roi),
        "ir_temperature": get_ir_temperature(roi),
        "tear_film_reflectivity": get_tear_film_reflectivity(roi),
        "pupil_dilation_rate": np.nan,
        "sclera_color_balance": get_sclera_color_balance(roi),
        "vein_pulsation_intensity": get_vein_pulsation_intensity(roi),
        "birefringence_index": get_birefringence_index(roi)
    }
    
    # Fill NaNs with model means
    pre = model.named_steps["preprocessor"]
    means = {n: float(s) for n, s in zip(FEATURES_ORDER, pre.named_steps["imputer"].statistics_)}
    
    for k in FEATURES_ORDER:
        if not np.isfinite(feats.get(k, np.nan)):
            feats[k] = means.get(k, 100.0)
    
    # Predict
    import pandas as pd
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