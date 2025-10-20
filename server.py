# server.py
# Minimal FastAPI API that matches training.py preprocessing
# POST /predict with an image -> returns {"estimate_mg_dl": float}

import io, os
import numpy as np
import cv2
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ==== MUST MATCH YOUR TRAINING SETTINGS ====
WIDTH  = int(os.getenv("IMG_WIDTH", 224))
HEIGHT = int(os.getenv("IMG_HEIGHT", 224))
USE_RGB = os.getenv("USE_RGB", "1") not in ("0", "false", "False")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")  # or "best_model.pkl"
# ===========================================

# pick the pkl that exists
if not os.path.exists(MODEL_PATH):
    if os.path.exists("best_model.pkl"):
        MODEL_PATH = "best_model.pkl"
    elif os.path.exists("model.pkl"):
        MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI(title="EyeBloodGlucose API (research)")

# Allow your store to call this API (adjust domain later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this to your store domain later
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

def preprocess_pil(img: Image.Image) -> np.ndarray:
    """Resize -> [0,1] -> flatten, matching training.py"""
    if USE_RGB:
        img = img.convert("RGB")
        arr = np.asarray(img.resize((WIDTH, HEIGHT)), dtype=np.float32) / 255.0
        feat = arr.reshape(-1)
    else:
        img = img.convert("L")
        arr = np.asarray(img.resize((WIDTH, HEIGHT)), dtype=np.float32) / 255.0
        feat = arr.reshape(-1)
    return feat

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    x = preprocess_pil(img)
    try:
        y = float(model.predict([x])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return {"estimate_mg_dl": round(y, 2)}
