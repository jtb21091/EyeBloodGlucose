# training.py
# Run examples:
#   (.venv) python training.py
#   (.venv) python training.py eye_glucose_data/labels.csv
#   (.venv) python training.py --csv eye_glucose_data/labels.csv --out best_model.pkl --mmol

import os, logging, warnings, json, argparse
import numpy as np, pandas as pd, joblib
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

# -----------------------
# Config
# -----------------------
FEATURES_ORDER = [
    "pupil_size","sclera_redness","vein_prominence","pupil_response_time","ir_intensity",
    "scleral_vein_density","ir_temperature","tear_film_reflectivity","pupil_dilation_rate",
    "sclera_color_balance","vein_pulsation_intensity","birefringence_index"
]
TARGET_CANDIDATES = ["blood_glucose","blood_glucose_mg_dl","glucose_mg_dl"]
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------
# Helpers
# -----------------------
def _cand_paths(here: str) -> list[str]:
    """Candidate locations for labels.csv relative to CWD and script dir."""
    return [
        # Plain name (CWD)
        "labels.csv",
        # Common subfolder (CWD)
        os.path.join("eye_glucose_data", "labels.csv"),
        # Next to this script
        os.path.join(here, "labels.csv"),
        # Subfolder next to this script
        os.path.join(here, "eye_glucose_data", "labels.csv"),
    ]

def resolve_csv_path(user_csv: str | None) -> str:
    """
    Resolve the actual path to labels.csv.
    - If user_csv is provided and exists, use it.
    - Otherwise search common locations.
    """
    if user_csv and os.path.exists(user_csv):
        return user_csv

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = _cand_paths(here)
    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        f"labels.csv not found. Tried:\n  " + "\n  ".join(os.path.abspath(c) for c in candidates) +
        ("\n(You can also pass --csv /path/to/labels.csv)")
    )

# -----------------------
# Data prep
# -----------------------
def prepare_data(csv_path: str | None = None, convert_mmol_to_mgdl: bool = False):
    csv_path = resolve_csv_path(csv_path)
    logging.info(f"Reading labels from: {os.path.abspath(csv_path)}")

    df = pd.read_csv(csv_path)

    # Choose a target column
    target = next((t for t in TARGET_CANDIDATES if t in df.columns), None)
    if target is None:
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            raise ValueError(
                "labels.csv must contain a numeric target column or one of "
                f"{TARGET_CANDIDATES}. Numeric columns found: {list(num.columns)}"
            )
        target = num.columns[-1]

    # Ensure all feature columns exist; missing become NaN -> imputed
    for f in FEATURES_ORDER:
        if f not in df.columns:
            df[f] = np.nan

    X = df[FEATURES_ORDER].astype(float)
    y = pd.to_numeric(df[target], errors="coerce")

    if convert_mmol_to_mgdl:
        y = y * 18.0

    logging.info(f"Target column: {target}")
    logging.info(
        "y stats (mg/dL expected): "
        f"min={np.nanmin(y):.3f} max={np.nanmax(y):.3f} mean={np.nanmean(y):.3f}"
    )
    logging.info(f"Training features (12): {FEATURES_ORDER}")
    return X, y

# -----------------------
# Model spaces
# -----------------------
def model_spaces():
    return {
        "RF": (RandomForestRegressor(random_state=42), {
            "regressor__n_estimators": randint(200, 500),
            "regressor__max_depth": randint(5, 30),
            "regressor__min_samples_split": randint(2, 20),
            "regressor__min_samples_leaf": randint(1, 8),
        }),
        "GB": (GradientBoostingRegressor(random_state=42), {
            "regressor__n_estimators": randint(150, 500),
            "regressor__learning_rate": uniform(0.01, 0.2),
            "regressor__max_depth": randint(2, 6),
            "regressor__subsample": uniform(0.7, 0.3),
        }),
        "MLP": (MLPRegressor(max_iter=2000, early_stopping=True, random_state=42), {
            "regressor__hidden_layer_sizes": [(128, 64), (64, 32)],
            "regressor__alpha": uniform(1e-5, 1e-2),
            "regressor__learning_rate_init": uniform(1e-4, 5e-3),
        }),
    }

# -----------------------
# Train
# -----------------------
def train(csv_path: str | None = None, out_path: str = "best_model.pkl", convert_mmol_to_mgdl: bool = False):
    X, y = prepare_data(csv_path, convert_mmol_to_mgdl)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # means reused at runtime
        ("scaler", StandardScaler()),
    ])

    best, best_name, best_mse = None, "", np.inf

    for name, (reg, space) in model_spaces().items():
        pipe = Pipeline([("preprocessor", pre), ("regressor", reg)])

        search = RandomizedSearchCV(
            pipe,
            param_distributions=space,
            n_iter=25,
            cv=cv5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=42,
        )
        search.fit(Xtr, ytr)

        pred = search.predict(Xva)
        mse = mean_squared_error(yva, pred)
        r2  = r2_score(yva, pred)
        mae = mean_absolute_error(yva, pred)
        logging.info(f"{name}: R2={r2:.4f} MSE={mse:.4f} MAE={mae:.4f}")

        if mse < best_mse:
            best, best_name, best_mse = search.best_estimator_, name, mse

    logging.info(f"Best: {best_name} (MSE={best_mse:.4f})")
    joblib.dump(best, out_path)
    logging.info(f"Saved model to {os.path.abspath(out_path)}")

    # Sanity: print feature schema and imputer stats length
    m = joblib.load(out_path)
    print("feature_names_in_:", getattr(m, "feature_names_in_", None))
    pre2 = m.named_steps["preprocessor"]
    stats = pre2.named_steps["imputer"].statistics_
    print("imputer statistics length:", len(stats))

    # Also persist schema for debugging
    with open("model_schema.json", "w") as f:
        json.dump({"features": list(getattr(m, "feature_names_in_", []))}, f, indent=2)
    logging.info(f"Wrote schema to {os.path.abspath('model_schema.json')}")

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train glucose regressor.")
    # allow either positional CSV or --csv
    p.add_argument("csv", nargs="?", default=None, help="Path to labels.csv (optional).")
    p.add_argument("--csv", dest="csv_kw", default=None, help="Path to labels.csv (keyword).")
    p.add_argument("--out", dest="out_path", default="best_model.pkl", help="Output model path.")
    p.add_argument("--mmol", action="store_true", help="Targets are in mmol/L; convert to mg/dL.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # prefer --csv over positional if both given
    chosen_csv = args.csv_kw if args.csv_kw is not None else args.csv
    train(csv_path=chosen_csv, out_path=args.out_path, convert_mmol_to_mgdl=args.mmol)
