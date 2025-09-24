# training.py
# Run examples:
#   (.venv) python training.py
#   (.venv) python training.py eye_glucose_data/labels.csv
#   (.venv) python training.py --csv eye_glucose_data/labels.csv --out best_model.pkl --mmol
#   (.venv) python training.py --no-show

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
from scipy.stats import uniform, randint, norm
import matplotlib.pyplot as plt

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
# Metrics
# -----------------------
def compute_mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MARD (%) = mean( |pred - true| / true ) * 100
    Ignores non-positive true values (to avoid div-by-zero / nonsense).
    """
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    mask = y_true > 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]) * 100.0

def iso15197_within_spec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Returns a boolean array: prediction within ISO-like error spec commonly used
    for glucometers:
      - if true < 100 mg/dL: |error| <= 15 mg/dL
      - else:                |error| / true <= 0.15
    """
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    err = np.abs(y_pred - y_true)
    cond_low = y_true < 100
    within_low = err <= 15.0
    within_high = (err / np.maximum(y_true, 1e-9)) <= 0.15
    return np.where(cond_low, within_low, within_high)

def sigma_levels_from_yield(yield_rate: float) -> dict:
    """
    Convert yield (proportion within spec) to sigma levels.
    Short-term sigma = NORMSINV(yield)
    Long-term sigma (with 1.5 sigma shift) = NORMSINV(yield) + 1.5
    """
    yield_rate = float(np.clip(yield_rate, 1e-12, 1 - 1e-12))
    z = norm.ppf(yield_rate)
    return {
        "short_term_sigma": float(z),
        "long_term_sigma": float(z + 1.5),
    }

# -----------------------
# Plotting
# -----------------------
def plot_pred_vs_actual(y_true, y_pred, out_path="pred_vs_actual.png", show=True):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [
        np.floor(min(np.min(y_true), np.min(y_pred)) / 10.0) * 10.0,
        np.ceil(max(np.max(y_true), np.max(y_pred)) / 10.0) * 10.0
    ]
    plt.plot(lims, lims)  # y = x
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("True (mg/dL)")
    plt.ylabel("Predicted (mg/dL)")
    plt.title("Predicted vs True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def plot_residuals(y_true, y_pred, out_path="residuals_hist.png", show=True):
    residuals = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (Pred - True) mg/dL")
    plt.ylabel("Count")
    plt.title("Residuals Histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def plot_feature_importance(model, feature_names, out_path="feature_importance.png", show=True):
    # Works for tree-based models that expose feature_importances_
    importances = getattr(model.named_steps["regressor"], "feature_importances_", None)
    if importances is None:
        logging.info("Feature importance not available for this model; skipping plot.")
        return
    order = np.argsort(importances)[::-1]
    names_sorted = np.array(feature_names)[order]
    vals_sorted = np.array(importances)[order]
    plt.figure(figsize=(7,5))
    plt.bar(range(len(vals_sorted)), vals_sorted)
    plt.xticks(range(len(vals_sorted)), names_sorted, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

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
def train(csv_path: str | None = None,
          out_path: str = "best_model.pkl",
          convert_mmol_to_mgdl: bool = False,
          show_plots: bool = True):
    X, y = prepare_data(csv_path, convert_mmol_to_mgdl)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # means reused at runtime
        ("scaler", StandardScaler()),
    ])

    best, best_name, best_mse = None, "", np.inf
    best_preds = None

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
            best_preds = pred

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

    # ----- Extra metrics (on validation set) -----
    y_true = yva
    y_pred = best_preds

    mard = compute_mard(y_true, y_pred)
    within = iso15197_within_spec(y_true, y_pred)
    yield_rate = float(np.mean(within))
    sigma_dict = sigma_levels_from_yield(yield_rate)
    defects = int((~within).sum())
    total = int(within.size)
    dpmo = (1.0 - yield_rate) * 1_000_000.0

    logging.info(f"MARD: {mard:.2f}%")
    logging.info(f"ISO-like Yield (within spec): {yield_rate*100:.2f}%")
    logging.info(f"Defects: {defects} / {total}  |  DPMO: {dpmo:.0f}")
    logging.info(f"Sigma (short-term): {sigma_dict['short_term_sigma']:.2f}")
    logging.info(f"Sigma (long-term, +1.5 shift): {sigma_dict['long_term_sigma']:.2f}")

    # Save metrics to JSON
    metrics = {
        "model": best_name,
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mard_percent": float(mard),
        "yield_within_spec": float(yield_rate),
        "defects": defects,
        "total": total,
        "dpmo": float(dpmo),
        "sigma_short_term": float(sigma_dict["short_term_sigma"]),
        "sigma_long_term": float(sigma_dict["long_term_sigma"]),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Wrote metrics to {os.path.abspath('metrics.json')}")

    # ----- Plots -----
    plot_pred_vs_actual(y_true, y_pred, out_path="pred_vs_actual.png", show=show_plots)
    plot_residuals(y_true, y_pred, out_path="residuals_hist.png", show=show_plots)
    plot_feature_importance(m, FEATURES_ORDER, out_path="feature_importance.png", show=show_plots)

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
    p.add_argument("--no-show", action="store_true", help="Save plots without displaying them.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # prefer --csv over positional if both given
    chosen_csv = args.csv_kw if args.csv_kw is not None else args.csv
    train(
        csv_path=chosen_csv,
        out_path=args.out_path,
        convert_mmol_to_mgdl=args.mmol,
        show_plots=(not args.no_show),
    )
