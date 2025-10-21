import os
import io
import cv2
import joblib
import numpy as np
import pandas as pd
import logging
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List

logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------
# FeatureEngineer  (same as your training/prediction scripts)
# ------------------------------------------------------------------
FEATURES_ORDER = [
    'pupil_size', 'sclera_redness', 'vein_prominence', 'capture_duration', 'ir_intensity',
    'scleral_vein_density', 'ir_temperature', 'tear_film_reflectivity',
    'sclera_color_balance', 'vein_pulsation_intensity', 'birefringence_index',
    'lens_clarity_score', 'sclera_yellowness', 'vessel_tortuosity', 'image_quality_score'
]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, add_ratios=True, add_interactions=True, add_polynomials=True):
        self.add_ratios = add_ratios
        self.add_interactions = add_interactions
        self.add_polynomials = add_polynomials
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=FEATURES_ORDER)
        X_new = X_df.copy()

        if self.add_ratios:
            X_new['vein_to_redness_ratio'] = X_df['vein_prominence'] / (X_df['sclera_redness'] + 1e-6)
            X_new['density_to_prominence_ratio'] = X_df['scleral_vein_density'] / (X_df['vein_prominence'] + 1e-6)
            X_new['ir_to_reflectivity_ratio'] = X_df['ir_intensity'] / (X_df['tear_film_reflectivity'] + 1e-6)
            X_new['clarity_to_yellowness_ratio'] = X_df['lens_clarity_score'] / (X_df['sclera_yellowness'] + 1e-6)
            X_new['pupil_to_duration_ratio'] = X_df['pupil_size'] / (X_df['capture_duration'] + 1e-6)
            X_new['quality_to_tortuosity_ratio'] = X_df['image_quality_score'] / (X_df['vessel_tortuosity'] + 1e-6)

        if self.add_interactions:
            X_new['vascular_health_index'] = (
                X_df['vein_prominence'] * X_df['scleral_vein_density'] * X_df['vein_pulsation_intensity']
            ) ** (1/3)
            X_new['optical_clarity_index'] = (
                X_df['lens_clarity_score'] * X_df['tear_film_reflectivity'] * X_df['birefringence_index']
            ) ** (1/3)
            X_new['pupil_response_index'] = X_df['pupil_size'] * X_df['capture_duration']
            X_new['vessel_dynamics_index'] = X_df['vessel_tortuosity'] * X_df['vein_pulsation_intensity']

        if self.add_polynomials:
            key_features = ['pupil_size', 'sclera_redness', 'vein_prominence', 'ir_intensity', 'lens_clarity_score']
            for feat in key_features:
                X_new[f'{feat}_squared'] = X_df[feat] ** 2

        self.feature_names_ = list(X_new.columns)
        return X_new.values

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_ if self.feature_names_ else FEATURES_ORDER


# ------------------------------------------------------------------
# FastAPI app setup
# ------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best_model.pkl"
MODEL_URL = "https://github.com/jtb21091/EyeBloodGlucose/releases/download/v1.0.0/best_model.pkl"

def download_model():
    """Download model from GitHub Release if not already present."""
    if not os.path.exists(MODEL_PATH):
        logging.info(f"Model not found locally. Downloading from {MODEL_URL}...")
        resp = requests.get(MODEL_URL, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Model downloaded to {MODEL_PATH}")
    else:
        logging.info(f"Model found locally at {MODEL_PATH}")


# ------------------------------------------------------------------
# Load model (HOT-FIX for pickled __main__.FeatureEngineer)
# ------------------------------------------------------------------
download_model()

# >>> HOT-FIX <<<
import sys
setattr(sys.modules.setdefault("__main__", sys.modules["__main__"]), "FeatureEngineer", FeatureEngineer)
# >>> END HOT-FIX <<<

model = joblib.load(MODEL_PATH)
logging.info("Model loaded successfully")


# ------------------------------------------------------------------
# Prediction endpoint (same as before)
# ------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # Dummy example – replace with your actual preprocessing / feature extraction
    features = np.random.rand(1, len(FEATURES_ORDER))
    prediction = model.predict(features)[0]

    return {"predicted_glucose_level": float(prediction)}


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "OK", "message": "EyeBloodGlucose API running."}
