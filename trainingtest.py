import os, logging, warnings, json
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from scipy.stats import uniform, randint

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent

# Default candidates we’ll try in order if --csv is not provided
DEFAULT_CSV_CANDIDATES = [
    BASE_DIR / "_eye_glucose_data" / "labels.csv",  # your screenshot location
    BASE_DIR / "labels.csv",
]

DEFAULT_MODEL = BASE_DIR / "best_model.pkl"
DEFAULT_SCHEMA = BASE_DIR / "model_schema.json"

FEATURES_ORDER = [
    'pupil_size','sclera_redness','vein_prominence','pupil_response_time','ir_intensity',
    'scleral_vein_density','ir_temperature','tear_film_reflectivity','pupil_dilation_rate',
    'sclera_color_balance','vein_pulsation_intensity','birefringence_index'
]
TARGET_CANDIDATES = ["blood_glucose","blood_glucose_mg_dl","glucose_mg_dl"]

cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

def _autofind_labels_csv():
    """Return the first existing candidate; if none, search recursively for 'labels.csv'."""
    for p in DEFAULT_CSV_CANDIDATES:
        if p.exists():
            return p
    # last resort: search the repo for any labels.csv
    hits = list(BASE_DIR.rglob("labels.csv"))
    return hits[0] if hits else None

def _resolve_csv(csv_path):
    """
    Resolve CSV path so it works regardless of CWD.
    - If csv_path is provided: absolute => use as-is; relative => relative to script dir.
    - If not provided: pick first existing from DEFAULT_CSV_CANDIDATES, else rglob search.
    """
    if csv_path:
        p = Path(csv_path)
        return p if p.is_absolute() else (BASE_DIR / p)
    found = _autofind_labels_csv()
    return found

def prepare_data(csv_path=None, convert_mmol_to_mgdl=False):
    csv_path = _resolve_csv(csv_path)

    if not csv_path or not csv_path.exists():
        tried = "\n  ".join(str(p) for p in DEFAULT_CSV_CANDIDATES)
        raise FileNotFoundError(
            "CSV not found.\n"
            f"Tried:\n  {tried}\n")