# server.py  (feature-based; uses your EyeGlucoseMonitor pipeline)
import io, os, logging
import numpy as np
import cv2
import joblib
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# --- import from prediction.py (your existing file) ---
from prediction import EyeGlucoseMonitor  # uses detector + 12 feature extractors

MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pkl")

app = FastAPI(title="EyeBloodGlucose (feature-based)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Shopify domain later
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def serve_index():
    # serve your one-file demo UI
    return FileResponse("index.html")

# Load the feature-based pipeline (once)
try:
    monitor = EyeGlucoseMonitor(MODEL_PATH)  # validates feature_names_in_ etc.
    logging.info(f"Loaded feature model: {MODEL_PATH}")
except Exception as e:
    logging.exception("Failed to initialize EyeGlucoseMonitor")
    raise RuntimeError(f"Could not load model {MODEL_PATH}: {e}")

def bytes_to_bgr(img_bytes: bytes):
    """Convert uploaded bytes -> OpenCV BGR frame."""
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1) load image
    try:
        raw = await file.read()
        frame = bytes_to_bgr(raw)
        if frame is None:
            # fall back through PIL if OpenCV can't decode
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            frame = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # 2) detect eyes + crop (uses your tolerant logic)
    det = monitor.detect(frame)
    if det.roi is None:
        # No usable eyes in this frame — send a 422 so the client can keep trying
        raise HTTPException(status_code=422, detail="No eyes detected; try adjusting lighting/position.")

    # 3) extract engineered features (12), impute & predict
    try:
        feats = monitor.extract_features(det.roi)          # dict of 12 features
        y = monitor.predict_once(feats)                    # mg/dL
        
        # Convert NaN values to None for JSON compatibility
        feats_clean = {k: (None if np.isnan(v) or np.isinf(v) else float(v)) 
                       for k, v in feats.items()}
        
        return {
            "estimate_mg_dl": float(round(y, 2)),
            "mode": monitor.eyes_mode,                     # "both" | "single"
            "features": feats_clean                        # optional: handy for debugging
        }
    except Exception as e:
        logging.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")