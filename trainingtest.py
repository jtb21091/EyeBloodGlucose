import os, logging, warnings, json
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

FEATURES_ORDER = [
    'pupil_size','sclera_redness','vein_prominence','pupil_response_time','ir_intensity',
    'scleral_vein_density','ir_temperature','tear_film_reflectivity','pupil_dilation_rate',
    'sclera_color_balance','vein_pulsation_intensity','birefringence_index'
]
TARGET_CANDIDATES = ["blood_glucose","blood_glucose_mg_dl","glucose_mg_dl"]

cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

def prepare_data(csv_path="labels.csv", convert_mmol_to_mgdl=False):
    df = pd.read_csv(csv_path)

    # Choose target
    target = next((t for t in TARGET_CANDIDATES if t in df.columns), None)
    if target is None:
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            raise ValueError("labels.csv must contain a numeric target.")
        target = num.columns[-1]

    # Ensure the 12 columns exist (missing becomes NaN -> imputed)
    for f in FEATURES_ORDER:
        if f not in df.columns:
            df[f] = np.nan

    X = df[FEATURES_ORDER].astype(float)
    y = pd.to_numeric(df[target], errors="coerce")

    if convert_mmol_to_mgdl:
        y = y * 18.0

    logging.info(f"Target: {target}")
    logging.info(f"y stats (mg/dL expected): min={np.nanmin(y):.3f} max={np.nanmax(y):.3f} mean={np.nanmean(y):.3f}")
    logging.info(f"Training features (12): {FEATURES_ORDER}")
    return X, y

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
            "regressor__hidden_layer_sizes": [(128,64),(64,32)],
            "regressor__alpha": uniform(1e-5, 1e-2),
            "regressor__learning_rate_init": uniform(1e-4, 5e-3),
        }),
    }

def train(csv_path="labels.csv", out_path="best_model.pkl", convert_mmol_to_mgdl=False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
            pipe, param_distributions=space, n_iter=25, cv=cv5,
            scoring="neg_mean_squared_error", n_jobs=-1, random_state=42
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
    logging.info(f"Saved model to {out_path}")

    # Sanity: print feature schema and imputer stats length
    m = joblib.load(out_path)
    print("feature_names_in_:", getattr(m, "feature_names_in_", None))
    pre2 = m.named_steps["preprocessor"]
    stats = pre2.named_steps["imputer"].statistics_
    print("imputer statistics length:", len(stats))

    # Also persist schema for debugging
    with open("model_schema.json","w") as f:
        json.dump({"features": list(getattr(m,"feature_names_in_",[]))}, f, indent=2)

if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "labels.csv"
    train(csv)
