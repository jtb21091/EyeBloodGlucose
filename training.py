# training.py - ENHANCED with Ensemble & Feature Engineering
# Run examples:
#   (.venv) python training.py
#   (.venv) python training.py eye_glucose_data/labels.csv
#   (.venv) python training.py --csv eye_glucose_data/labels.csv --out best_model.pkl --mmol
#   (.venv) python training.py --no-show --ensemble-type voting  # or 'stacking'

import os, logging, warnings, json, argparse, shutil, time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import numpy as np, pandas as pd, joblib
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    VotingRegressor,
    StackingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import (
    ElasticNet,
    Ridge,
    Lasso,
    BayesianRidge,
    HuberRegressor,
)
from sklearn.base import clone, BaseEstimator, TransformerMixin
from scipy.stats import uniform, randint, loguniform, norm
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
FEATURES_ORDER = [
    'pupil_size', 'sclera_redness', 'vein_prominence', 'capture_duration', 'ir_intensity',
    'scleral_vein_density', 'ir_temperature', 'tear_film_reflectivity',
    'sclera_color_balance', 'vein_pulsation_intensity', 'birefringence_index',
    'lens_clarity_score', 'sclera_yellowness', 'vessel_tortuosity', 'image_quality_score'
]

OLD_COLUMN_MAPPING = {
    'pupil_response_time': 'capture_duration',
    'pupil_dilation_rate': None
}

TARGET_CANDIDATES = ["blood_glucose","blood_glucose_mg_dl","glucose_mg_dl"]
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------
# Feature Engineering Transformer
# -----------------------
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

# -----------------------
# Helpers
# -----------------------
def _cand_paths(here: str) -> List[str]:
    """Preferred search order for dataset CSVs (cleaned first)."""
    candidates = [
        # Prefer strong cleaned version if present
        os.path.join("eye_glucose_data", "labels_clean_drop_any.csv"),
        os.path.join(here, "eye_glucose_data", "labels_clean_drop_any.csv"),
        # Then generic cleaned
        os.path.join("eye_glucose_data", "labels_clean.csv"),
        os.path.join(here, "eye_glucose_data", "labels_clean.csv"),
        # Fallback to raw
        "labels.csv",
        os.path.join("eye_glucose_data", "labels.csv"),
        os.path.join(here, "labels.csv"),
        os.path.join(here, "eye_glucose_data", "labels.csv"),
    ]
    # Deduplicate while preserving order
    seen = set(); out = []
    for c in candidates:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def resolve_csv_path(user_csv: Optional[str]) -> str:
    if user_csv and os.path.exists(user_csv):
        return user_csv
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = _cand_paths(here)
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "labels.csv not found. Tried:\n  " + "\n  ".join(os.path.abspath(c) for c in candidates) +
        "\n(You can also pass --csv /path/to/labels.csv)"
    )


def _resolve_raw_csv_fallback() -> Optional[str]:
    """Prefer the raw labels.csv for precleaning when user didn't specify a CSV."""
    here = os.path.dirname(os.path.abspath(__file__))
    raw_candidates = [
        os.path.join("eye_glucose_data", "labels.csv"),
        os.path.join(here, "eye_glucose_data", "labels.csv"),
        "labels.csv",
        os.path.join(here, "labels.csv"),
    ]
    for c in raw_candidates:
        if os.path.exists(c):
            return c
    return None

def _preclean_csv(input_csv: str, mode: str = "sixsigma", output: Optional[str] = None,
                  min_nonnull_ratio: float = 0.2, min_nonnull_count: int = 100,
                  ignore_cols: Optional[str] = None) -> str:
    """Run the cleaner and return path to cleaned CSV.

    mode: 'sixsigma' (±6σ only) or 'strong' (also drop any zero/max with guards).
    """
    from clean_labels_sixsigma import clean_dataframe
    df = pd.read_csv(input_csv)
    ignore = [c for c in (ignore_cols.split(',') if ignore_cols else []) if c.strip()]

    kwargs = dict(
        flatline_threshold=0.02,
        min_nonnull_ratio=min_nonnull_ratio,
        min_nonnull_count=min_nonnull_count,
        ignore_columns=ignore,
    )
    if mode == "strong":
        kwargs.update(drop_any_zero=True, drop_any_max=True)
    else:
        kwargs.update(drop_any_zero=False, drop_any_max=False)

    kept, removed, report = clean_dataframe(df, **kwargs)

    out_path = output or os.path.join(os.path.dirname(input_csv) or ".", f"labels_clean_{mode}.csv")
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False)
    logging.info(
        f"Preclean[{mode}] wrote {os.path.abspath(out_path)} | kept {report['rows_kept']}/{report['rows_total']} rows; removed {report['rows_removed']}"
    )
    return out_path

def _overwrite_canonical_labels(cleaned_csv: str) -> str:
    """Copy cleaned CSV over the canonical eye_glucose_data/labels.csv with a timestamped backup."""
    # Determine canonical path
    here = os.path.dirname(os.path.abspath(__file__))
    canonical = os.path.join(here, "eye_glucose_data", "labels.csv")
    Path(os.path.dirname(canonical)).mkdir(parents=True, exist_ok=True)

    # Backup existing if present
    if os.path.exists(canonical):
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = canonical + f".backup_{ts}"
        shutil.copyfile(canonical, backup)
        logging.info(f"Backed up existing labels.csv to {os.path.abspath(backup)}")

    # Overwrite canonical
    shutil.copyfile(cleaned_csv, canonical)
    logging.info(f"Wrote cleaned dataset to {os.path.abspath(canonical)}")
    return canonical

def _cleanup_clean_artifacts():
    """Remove intermediate cleaned/removed/report files to keep only one labels.csv."""
    here = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(here, "eye_glucose_data")
    patterns = [
        "labels_clean*.csv",
        "labels_removed*.csv",
        "cleaning_report*.json",
    ]
    import glob
    removed = 0
    for pat in patterns:
        for p in glob.glob(os.path.join(folder, pat)):
            # Don't remove the canonical labels.csv
            if os.path.basename(p) == "labels.csv":
                continue
            try:
                os.remove(p)
                removed += 1
            except OSError:
                pass
    if removed:
        logging.info(f"Cleaned up {removed} intermediate cleaning artifact files in {folder}")

# -----------------------
# Data prep
# -----------------------
def prepare_data(csv_path: Optional[str] = None, convert_mmol_to_mgdl: bool = False, log_target: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    csv_path = resolve_csv_path(csv_path)
    logging.info(f"Reading labels from: {os.path.abspath(csv_path)}")

    df = pd.read_csv(csv_path)

    target = next((t for t in TARGET_CANDIDATES if t in df.columns), None)
    if target is None:
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            raise ValueError("labels.csv must contain a numeric target column.")
        target = num.columns[-1]

    for old_name, new_name in OLD_COLUMN_MAPPING.items():
        if old_name in df.columns:
            if new_name is not None:
                df[new_name] = df[old_name]
                logging.info(f"Mapped old column '{old_name}' → '{new_name}'")
            else:
                logging.info(f"Dropping duplicate column '{old_name}'")

    for f in FEATURES_ORDER:
        if f not in df.columns:
            df[f] = np.nan
            logging.debug(f"Feature '{f}' not in CSV; will be imputed")

    X = df[FEATURES_ORDER].astype(float)
    y = pd.to_numeric(df[target], errors="coerce")

    if convert_mmol_to_mgdl:
        y = y * 18.0

    new_features = ['lens_clarity_score', 'sclera_yellowness', 'vessel_tortuosity', 'image_quality_score']
    new_feature_counts = {f: X[f].notna().sum() for f in new_features}
    
    logging.info(f"Target column: {target}")
    if log_target:
        # Only log-transform positive values; non-positive become NaN and will be dropped by scikit automatically
        y = pd.to_numeric(y, errors="coerce")
        y = y.where(y > 0)
        y = np.log(y)
        logging.info(
            "y stats (log mg/dL): "
            f"min={np.nanmin(y):.3f} max={np.nanmax(y):.3f} mean={np.nanmean(y):.3f}"
        )
    else:
        logging.info(
            "y stats (mg/dL expected): "
            f"min={np.nanmin(y):.3f} max={np.nanmax(y):.3f} mean={np.nanmean(y):.3f}"
        )
    logging.info(f"Training features (15): {FEATURES_ORDER}")
    logging.info(f"New features availability:")
    for feat, count in new_feature_counts.items():
        logging.info(f"  {feat}: {count}/{len(df)} samples ({count/len(df)*100:.1f}%)")
    
    return X, y

# -----------------------
# Metrics
# -----------------------
def compute_mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    mask = y_true > 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]) * 100.0

def iso15197_within_spec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    err = np.abs(y_pred - y_true)
    cond_low = y_true < 100
    within_low = err <= 15.0
    within_high = (err / np.maximum(y_true, 1e-9)) <= 0.15
    return np.where(cond_low, within_low, within_high)

def sigma_levels_from_yield(yield_rate: float) -> Dict[str, float]:
    yield_rate = float(np.clip(yield_rate, 1e-12, 1 - 1e-12))
    z = norm.ppf(yield_rate)
    return {
        "short_term_sigma": float(z),
        "long_term_sigma": float(z + 1.5),
    }

# -----------------------
# Plotting
# -----------------------
def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_path: str = "pred_vs_actual.png", show: bool = True):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [
        np.floor(min(np.min(y_true), np.min(y_pred)) / 10.0) * 10.0,
        np.ceil(max(np.max(y_true), np.max(y_pred)) / 10.0) * 10.0
    ]
    plt.plot(lims, lims, 'r--', label='Perfect prediction')
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("True (mg/dL)")
    plt.ylabel("Predicted (mg/dL)")
    plt.title("Predicted vs True (OOF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: str = "residuals.png", show: bool = True):
    resid = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, resid, alpha=0.5)
    plt.axhline(0.0, color='red', linestyle='--')
    plt.xlabel("Predicted (mg/dL)")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residuals vs Predicted (OOF)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def plot_feature_importance(model: Any, feature_names: List[str], out_path: str = "feature_importance.png", show: bool = True, top_n: int = 20):
    """Plot feature importance, handling ensemble models (Voting/Stacking) and singles.

    - For VotingRegressor, estimators_ is typically a list of fitted estimators (no names).
    - For StackingRegressor, estimators_ is also a list of fitted estimators.
    We aggregate available importances/coefficients across base estimators.
    """
    importances = None

    # Try to get importance from the model
    if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
        reg = model.named_steps['regressor']

        # Handle ensemble models
        if isinstance(reg, (VotingRegressor, StackingRegressor)) and hasattr(reg, 'estimators_'):
            # Build a normalized list of (name, estimator)
            base_ests = reg.estimators_
            if len(base_ests) > 0 and isinstance(base_ests[0], tuple):
                estimators = [(name, est) for name, est in base_ests]
            else:
                estimators = [(f"est_{i}", est) for i, est in enumerate(base_ests)]

            # Average importances from base estimators that have them
            agg = []
            for name, est in estimators:
                target_est = None
                if hasattr(est, 'named_steps') and 'regressor' in est.named_steps:
                    target_est = est.named_steps['regressor']
                else:
                    target_est = est

                if hasattr(target_est, 'feature_importances_'):
                    vals = np.asarray(getattr(target_est, 'feature_importances_'))
                    agg.append(vals)
                elif hasattr(target_est, 'coef_'):
                    coef = getattr(target_est, 'coef_')
                    if coef is not None:
                        vals = np.abs(np.ravel(coef))
                        agg.append(vals)

            if agg:
                # Ensure consistent length before averaging
                lens = [len(v) for v in agg]
                target_len = len(feature_names)
                aligned = [v for v in agg if len(v) == target_len]
                if aligned:
                    importances = np.mean(np.stack(aligned, axis=0), axis=0)
        elif hasattr(reg, 'feature_importances_'):
            importances = getattr(reg, 'feature_importances_')
        elif hasattr(reg, 'coef_'):
            coef = getattr(reg, 'coef_')
            importances = np.abs(np.ravel(coef))

    if importances is None:
        logging.info("Feature importance not available for this model; skipping plot.")
        return

    # Get feature names (might be engineered features)
    if hasattr(model, 'named_steps') and 'feature_engineer' in model.named_steps:
        feat_eng = model.named_steps['feature_engineer']
        if hasattr(feat_eng, 'feature_names_'):
            feature_names = feat_eng.feature_names_

    # Sort and plot top N
    order = np.argsort(importances)[::-1][:top_n]
    names_sorted = np.array(feature_names)[order]
    vals_sorted = np.array(importances)[order]

    # Color engineered and new features differently
    colors = []
    for n in names_sorted:
        if any(suffix in n for suffix in ['_ratio', '_index', '_squared', '_interaction']):
            colors.append('green')  # Engineered features
        elif n in ['lens_clarity_score', 'sclera_yellowness', 'vessel_tortuosity', 'image_quality_score']:
            colors.append('orange')  # New features
        else:
            colors.append('blue')  # Original features
        for n in names_sorted:
            if any(suffix in n for suffix in ['_ratio', '_index', '_squared', '_interaction']):
                colors.append('green')  # Engineered features
            elif n in ['lens_clarity_score', 'sclera_yellowness', 'vessel_tortuosity', 'image_quality_score']:
                colors.append('orange')  # New features
            else:
                colors.append('blue')  # Original features
    
        plt.figure(figsize=(12,8))
        plt.bar(range(len(vals_sorted)), vals_sorted, color=colors)
        plt.xticks(range(len(vals_sorted)), names_sorted, rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.title(f"Top {top_n} Feature Importances\n(Green=Engineered, Orange=New, Blue=Original)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        if show:
            plt.show()
        plt.close()
    
    # Get feature names (might be engineered features)
    if hasattr(model, 'named_steps') and 'feature_engineer' in model.named_steps:
        feat_eng = model.named_steps['feature_engineer']
        if hasattr(feat_eng, 'feature_names_'):
            feature_names = feat_eng.feature_names_
    
    # Sort and plot top N
    order = np.argsort(importances)[::-1][:top_n]
    names_sorted = np.array(feature_names)[order]
    vals_sorted = np.array(importances)[order]
    
    # Color engineered features differently
    colors = []
    for n in names_sorted:
        if any(suffix in n for suffix in ['_ratio', '_index', '_squared', '_interaction']):
            colors.append('green')  # Engineered features
        elif n in ['lens_clarity_score', 'sclera_yellowness', 'vessel_tortuosity', 'image_quality_score']:
            colors.append('orange')  # New features
        else:
            colors.append('blue')  # Original features
    
    plt.figure(figsize=(12,8))
    plt.bar(range(len(vals_sorted)), vals_sorted, color=colors)
    plt.xticks(range(len(vals_sorted)), names_sorted, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title(f"Top {top_n} Feature Importances\n(Green=Engineered, Orange=New, Blue=Original)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

# -----------------------
# Model spaces
# -----------------------
def model_spaces(include_optional: bool = True, fast: bool = False) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Returns dictionary of models to train.
    
    Args:
        include_optional: If True, include XGBoost, LightGBM, CatBoost if available
    """
    models = {
        # === TREE-BASED ENSEMBLES ===
        "RF": (RandomForestRegressor(random_state=42, n_jobs=-1), {
            "regressor__n_estimators": randint(200, 600),
            "regressor__max_depth": randint(5, 40),
            "regressor__min_samples_split": randint(2, 20),
            "regressor__min_samples_leaf": randint(1, 8),
        }),
        "ET": (ExtraTreesRegressor(random_state=42, n_jobs=-1), {
            "regressor__n_estimators": randint(300, 800),
            "regressor__max_depth": randint(5, 40),
            "regressor__min_samples_split": randint(2, 20),
            "regressor__min_samples_leaf": randint(1, 8),
        }),
        "GB": (GradientBoostingRegressor(random_state=42), {
            "regressor__n_estimators": randint(150, 600),
            "regressor__learning_rate": uniform(0.01, 0.25),
            "regressor__max_depth": randint(2, 6),
            "regressor__subsample": uniform(0.7, 0.3),
        }),
        "HGB": (HistGradientBoostingRegressor(random_state=42), {
            "regressor__max_depth": randint(3, 10),
            "regressor__learning_rate": loguniform(1e-3, 3e-1),
            "regressor__max_iter": randint(200, 1000),
            "regressor__l2_regularization": loguniform(1e-8, 1e-1),
        }),
        "ADA": (AdaBoostRegressor(random_state=42), {
            "regressor__n_estimators": randint(100, 600),
            "regressor__learning_rate": loguniform(1e-3, 1.0),
        }),
        
        # === NEURAL NETWORKS ===
        "MLP": (MLPRegressor(max_iter=3000, early_stopping=True, random_state=42), {
            "regressor__hidden_layer_sizes": [(256,128,64), (128,64), (64,32), (512,256,128)],
            "regressor__alpha": loguniform(1e-6, 1e-2),
            "regressor__learning_rate_init": loguniform(1e-4, 3e-3),
            "regressor__activation": ["relu", "tanh"],
        }),
        
        # === LINEAR REGRESSION MODELS ===
        "Ridge": (Ridge(random_state=42), {
            "regressor__alpha": loguniform(1e-4, 1e2),
            "regressor__solver": ["auto", "svd", "cholesky", "lsqr"],
        }),
        "Lasso": (Lasso(random_state=42, max_iter=5000), {
            "regressor__alpha": loguniform(1e-4, 1e1),
            "regressor__selection": ["cyclic", "random"],
        }),
        "EN": (ElasticNet(random_state=42, max_iter=5000), {
            "regressor__alpha": loguniform(1e-4, 1e1),
            "regressor__l1_ratio": uniform(0.0, 1.0),
        }),
        "BayesRidge": (BayesianRidge(), {
            "regressor__alpha_1": loguniform(1e-7, 1e-4),
            "regressor__alpha_2": loguniform(1e-7, 1e-4),
            "regressor__lambda_1": loguniform(1e-7, 1e-4),
            "regressor__lambda_2": loguniform(1e-7, 1e-4),
        }),
        "Huber": (HuberRegressor(max_iter=1000), {
            "regressor__epsilon": uniform(1.1, 0.5),
            "regressor__alpha": loguniform(1e-5, 1e-1),
        }),
        
        # === KERNEL & INSTANCE-BASED ===
        "SVR": (SVR(), {
            "regressor__C": loguniform(1e-1, 1e3),
            "regressor__epsilon": loguniform(1e-3, 1e0),
            "regressor__gamma": ["scale", "auto"],
            "regressor__kernel": ["rbf"],
        }),
        "KNN": (KNeighborsRegressor(), {
            "regressor__n_neighbors": randint(3, 35),
            "regressor__weights": ["uniform", "distance"],
            "regressor__p": [1, 2],
        }),
    }
    
    # === OPTIONAL: ADVANCED BOOSTING (require pip install) ===
    if include_optional and not fast:
        try:
            import xgboost as xgb
            models["XGB"] = (xgb.XGBRegressor(random_state=42, n_jobs=-1), {
                "regressor__n_estimators": randint(200, 800),
                "regressor__max_depth": randint(3, 10),
                "regressor__learning_rate": loguniform(1e-3, 3e-1),
                "regressor__subsample": uniform(0.6, 0.4),
                "regressor__colsample_bytree": uniform(0.6, 0.4),
                "regressor__reg_alpha": loguniform(1e-8, 1e-1),
                "regressor__reg_lambda": loguniform(1e-8, 1e-1),
            })
            logging.info("✓ XGBoost available")
        except ImportError:
            logging.info("✗ XGBoost not available (pip install xgboost)")
        
        try:
            import lightgbm as lgb
            models["LGBM"] = (lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1), {
                "regressor__n_estimators": randint(200, 800),
                "regressor__max_depth": randint(3, 15),
                "regressor__learning_rate": loguniform(1e-3, 3e-1),
                "regressor__num_leaves": randint(20, 150),
                "regressor__subsample": uniform(0.6, 0.4),
                "regressor__colsample_bytree": uniform(0.6, 0.4),
                "regressor__reg_alpha": loguniform(1e-8, 1e-1),
                "regressor__reg_lambda": loguniform(1e-8, 1e-1),
            })
            logging.info("✓ LightGBM available")
        except ImportError:
            logging.info("✗ LightGBM not available (pip install lightgbm)")
        
        try:
            import catboost as cb
            models["CatBoost"] = (cb.CatBoostRegressor(random_state=42, verbose=0), {
                "regressor__iterations": randint(200, 800),
                "regressor__depth": randint(4, 10),
                "regressor__learning_rate": loguniform(1e-3, 3e-1),
                "regressor__l2_leaf_reg": loguniform(1e-2, 1e1),
            })
            logging.info("✓ CatBoost available")
        except ImportError:
            logging.info("✗ CatBoost not available (pip install catboost)")
    
    # If fast mode, keep a lean set
    if fast:
        keep = ["RF", "ET", "HGB", "Ridge", "Huber", "SVR"]
        models = {k: v for k, v in models.items() if k in keep}
    return models

# -----------------------
# Create Ensemble
# -----------------------
def create_ensemble(top_models: List[Tuple[str, Any, np.ndarray]], ensemble_type: str = 'voting') -> Any:
    """
    Create an ensemble from top models.
    
    Args:
        top_models: List of (name, model, oof_predictions) tuples
        ensemble_type: 'voting' for simple averaging, 'stacking' for meta-learner
    
    Returns:
        Ensemble model (VotingRegressor or StackingRegressor)
    """
    estimators = [(name, model) for name, model, _ in top_models]
    
    if ensemble_type == 'voting':
        # Simple averaging ensemble
        ensemble = VotingRegressor(estimators=estimators, n_jobs=-1)
        logging.info(f"Created VotingRegressor with {len(estimators)} models")
    
    elif ensemble_type == 'stacking':
        # Stacking with Ridge meta-learner
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=-1
        )
        logging.info(f"Created StackingRegressor with {len(estimators)} base models and Ridge meta-learner")
    
    else:
        raise ValueError(f"Unknown ensemble_type: {ensemble_type}")
    
    return ensemble

# -----------------------
# Train with Ensemble
# -----------------------
def train(csv_path: Optional[str] = None,
          out_path: str = "best_model.pkl",
          convert_mmol_to_mgdl: bool = False,
          show_plots: bool = True,
          use_feature_engineering: bool = True,
          use_ensemble: bool = True,
          ensemble_type: str = 'voting',
        top_n_models: int = 3,
        log_target: bool = False,
        fast: bool = False,
          n_iter: int = 25):
    
    X, y = prepare_data(csv_path, convert_mmol_to_mgdl, log_target)

    # Build preprocessing pipeline with optional feature engineering
    pre_steps = []
    if use_feature_engineering:
        pre_steps.append(("feature_engineer", FeatureEngineer(
            add_ratios=True,
            add_interactions=True,
            add_polynomials=True
        )))
        logging.info("Feature engineering ENABLED: adding ratios, interactions, and polynomials")
    
    pre_steps.extend([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    pre = Pipeline(pre_steps)

    # Track all models
    all_results = []
    
    models_dict = model_spaces(fast=fast)
    logging.info(f"\n{'='*60}")
    logging.info(f"TRAINING {len(models_dict)} MODEL TYPES:")
    for model_name in models_dict.keys():
        logging.info(f"  • {model_name}")
    logging.info(f"{'='*60}\n")

    for name, (reg, space) in models_dict.items():
        logging.info(f"Training {name}...")
        pipe = Pipeline([("preprocessor", pre), ("regressor", reg)])

        # Randomized search
        search = RandomizedSearchCV(
            pipe,
            param_distributions=space,
            n_iter=n_iter if not fast else max(5, min(12, n_iter)),
            cv=cv5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=42,
            refit=True,
        )
        search.fit(X, y)

        # OOF predictions
        oof_pred = cross_val_predict(search.best_estimator_, X, y, cv=cv5, n_jobs=-1)
        if log_target:
            # Convert predictions back to mg/dL domain for reporting/selection
            oof_pred = np.exp(oof_pred)
            y_for_metrics = np.exp(y)
        else:
            y_for_metrics = y

        mse = mean_squared_error(y_for_metrics, oof_pred)
        r2  = r2_score(y_for_metrics, oof_pred)
        mae = mean_absolute_error(y_for_metrics, oof_pred)
        
        all_results.append({
            'name': name,
            'model': search.best_estimator_,
            'oof_pred': oof_pred,
            'mse': mse,
            'r2': r2,
            'mae': mae,
        })
        
        logging.info(f"{name}: OOF R2={r2:.4f} OOF MSE={mse:.4f} OOF MAE={mae:.4f}")

    # Sort by MSE (lower is better)
    all_results.sort(key=lambda x: x['mse'])
    
    # Log top models
    logging.info(f"\n{'='*60}")
    logging.info(f"TOP {top_n_models} MODELS:")
    for i, res in enumerate(all_results[:top_n_models], 1):
        logging.info(f"  {i}. {res['name']}: MSE={res['mse']:.4f}, R2={res['r2']:.4f}, MAE={res['mae']:.4f}")
    logging.info(f"{'='*60}\n")

    # Create ensemble or use best model
    if use_ensemble and len(all_results) >= top_n_models:
        top_models = [(r['name'], r['model'], r['oof_pred']) for r in all_results[:top_n_models]]
        
        # Create ensemble structure (without preprocessing - will add it)
        base_estimators = []
        for name, model, _ in top_models:
            # Extract just the regressor (preprocessing is shared)
            if hasattr(model, 'named_steps'):
                base_reg = model.named_steps['regressor']
            else:
                base_reg = model
            base_estimators.append((name, base_reg))
        
        # Create ensemble with shared preprocessing
        if ensemble_type == 'voting':
            ensemble_reg = VotingRegressor(estimators=base_estimators, n_jobs=-1)
        else:  # stacking
            ensemble_reg = StackingRegressor(
                estimators=base_estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=5,
                n_jobs=-1
            )
        
        final_model = Pipeline([("preprocessor", pre), ("regressor", ensemble_reg)])
        
        # Fit ensemble on full data
        logging.info(f"Fitting {ensemble_type} ensemble on full dataset...")
        final_model.fit(X, y)
        
        # Get OOF predictions for ensemble
        oof_pred = cross_val_predict(final_model, X, y, cv=cv5, n_jobs=-1)
        if log_target:
            oof_pred = np.exp(oof_pred)
            y_for_metrics = np.exp(y)
        else:
            y_for_metrics = y
        
        best_name = f"{ensemble_type.capitalize()}Ensemble_Top{top_n_models}"
        
    else:
        # Use single best model
        best_result = all_results[0]
        final_model = best_result['model']
        oof_pred = best_result['oof_pred']
        if log_target:
            oof_pred = np.exp(oof_pred)
            y_for_metrics = np.exp(y)
        else:
            y_for_metrics = y
        best_name = best_result['name']
        logging.info(f"Using single best model: {best_name}")

    # Save model
    joblib.dump(final_model, out_path)
    logging.info(f"Saved model to {os.path.abspath(out_path)}")

    # Get feature names for the final model
    if use_feature_engineering:
        # Transform once to get feature names
        X_transformed = pre.named_steps['feature_engineer'].fit_transform(X)
        feature_names = pre.named_steps['feature_engineer'].feature_names_
    else:
        feature_names = FEATURES_ORDER

    # Save schema
    with open("model_schema.json", "w") as f:
        json.dump({
            "features": list(feature_names),
            "original_features": FEATURES_ORDER,
            "model_type": best_name,
            "ensemble": use_ensemble,
            "feature_engineering": use_feature_engineering,
        }, f, indent=2)
    logging.info(f"Wrote schema to {os.path.abspath('model_schema.json')}")

    # Calculate metrics
    y_true = y_for_metrics
    y_pred = oof_pred

    mard = compute_mard(y_true, y_pred)
    within = iso15197_within_spec(y_true, y_pred)
    yield_rate = float(np.mean(within))
    defects = int((~within).sum())
    total = int(len(y_true))
    dpmo = (1.0 - yield_rate) * 1_000_000.0
    sigma = sigma_levels_from_yield(yield_rate)

    logging.info(f"\n{'='*60}")
    logging.info(f"FINAL MODEL METRICS:")
    logging.info(f"  Model: {best_name}")
    logging.info(f"  MARD: {mard:.2f}%")
    logging.info(f"  R²: {r2_score(y_true, y_pred):.4f}")
    logging.info(f"  MSE: {mean_squared_error(y_true, y_pred):.4f}")
    logging.info(f"  MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    logging.info(f"  ISO Yield: {yield_rate*100:.2f}%")
    logging.info(f"  Defects: {defects}/{total}  |  DPMO: {dpmo:.0f}")
    logging.info(f"  Sigma (ST): {sigma['short_term_sigma']:.2f}")
    logging.info(f"  Sigma (LT): {sigma['long_term_sigma']:.2f}")
    logging.info(f"{'='*60}\n")

    # Save metrics
    metrics = {
        "model": best_name,
        "ensemble": use_ensemble,
        "ensemble_type": ensemble_type if use_ensemble else None,
        "top_n_models": top_n_models if use_ensemble else 1,
        "feature_engineering": use_feature_engineering,
        "num_features": len(feature_names),
    "oof_r2": float(r2_score(y_true, y_pred)),
    "oof_mse": float(mean_squared_error(y_true, y_pred)),
    "oof_mae": float(mean_absolute_error(y_true, y_pred)),
        "mard_percent": float(mard),
        "yield_within_spec": float(yield_rate),
        "defects": defects,
        "total": total,
        "dpmo": float(dpmo),
        "sigma_short_term": float(sigma["short_term_sigma"]),
        "sigma_long_term": float(sigma["long_term_sigma"]),
        "top_models": [
            {"name": r['name'], "mse": float(r['mse']), "r2": float(r['r2'])}
            for r in all_results[:top_n_models]
        ]
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Wrote metrics to {os.path.abspath('metrics.json')}")

    # Plots
    Path("pngfiles").mkdir(parents=True, exist_ok=True)
    plot_pred_vs_actual(y_true, y_pred, out_path="pred_vs_actual.png", show=show_plots)
    try:
        shutil.copyfile("pred_vs_actual.png", os.path.join("pngfiles", "3.png"))
    except Exception as e:
        logging.warning(f"Could not mirror pred_vs_actual.png to pngfiles/3.png: {e}")

    plot_residuals(y_true, y_pred, out_path="residuals.png", show=show_plots)
    try:
        shutil.copyfile("residuals.png", os.path.join("pngfiles", "2.png"))
    except Exception as e:
        logging.warning(f"Could not mirror residuals.png to pngfiles/2.png: {e}")

    plot_feature_importance(final_model, feature_names, out_path="feature_importance.png", show=show_plots)
    try:
        shutil.copyfile("feature_importance.png", os.path.join("pngfiles", "4.png"))
    except Exception as e:
        logging.warning(f"Could not mirror feature_importance.png to pngfiles/4.png: {e}")

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train glucose regressor with ensemble & feature engineering.")
    p.add_argument("csv", nargs="?", default=None, help="Path to labels.csv (optional).")
    p.add_argument("--csv", dest="csv_kw", default=None, help="Path to labels.csv (keyword).")
    p.add_argument("--out", dest="out_path", default="best_model.pkl", help="Output model path.")
    p.add_argument("--mmol", action="store_true", help="Targets are in mmol/L; convert to mg/dL.")
    p.add_argument("--no-show", action="store_true", help="Save plots without displaying.")
    p.add_argument("--no-feature-engineering", action="store_true", help="Disable feature engineering.")
    p.add_argument("--no-ensemble", action="store_true", help="Use single best model instead of ensemble.")
    p.add_argument("--ensemble-type", choices=['voting', 'stacking'], default='voting',
                   help="Type of ensemble: 'voting' (averaging) or 'stacking' (meta-learner).")
    p.add_argument("--top-n", type=int, default=3, help="Number of top models to ensemble (default: 3).")
    p.add_argument("--log-target", action="store_true", help="Train on log(glucose); metrics reported in mg/dL.")
    p.add_argument("--fast", action="store_true", help="Faster run: fewer models and fewer search iterations.")
    p.add_argument("--n-iter", type=int, default=25, help="Search iterations per model (default 25).")
    # Integrated preclean options
    p.add_argument("--preclean", choices=["none", "sixsigma", "strong"], default="none",
                   help="Optionally clean the CSV before training. 'strong' also drops zero/max rows with guards.")
    p.add_argument("--min-nonnull-ratio", type=float, default=0.2,
                   help="Minimum non-null ratio required to apply zero/max rules in 'strong' mode.")
    p.add_argument("--min-nonnull-count", type=int, default=100,
                   help="Minimum non-null count required to apply zero/max rules in 'strong' mode.")
    p.add_argument("--ignore-cols", type=str, default="",
                   help="Comma-separated list of columns to ignore during preclean.")
    p.add_argument("--overwrite-labels", action="store_true",
                   help="After preclean, overwrite eye_glucose_data/labels.csv with cleaned data (backup created).")
    p.add_argument("--cleanup-clean-files", action="store_true",
                   help="Delete intermediate cleaned/removed/report files so only labels.csv remains.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    chosen_csv = args.csv_kw if args.csv_kw is not None else args.csv
    # Optional preclean
    if args.preclean != "none":
        base_csv = chosen_csv
        if base_csv is None:
            base_csv = _resolve_raw_csv_fallback()
            if base_csv is None:
                # Fall back to normal resolver if raw not found
                base_csv = resolve_csv_path(None)
        cleaned_csv = _preclean_csv(
            input_csv=base_csv,
            mode=("strong" if args.preclean == "strong" else "sixsigma"),
            output=None,
            min_nonnull_ratio=args.min_nonnull_ratio,
            min_nonnull_count=args.min_nonnull_count,
            ignore_cols=args.ignore_cols,
        )
        if args.overwrite_labels:
            canonical = _overwrite_canonical_labels(cleaned_csv)
            chosen_csv = canonical
            if args.cleanup_clean_files:
                _cleanup_clean_artifacts()
        else:
            chosen_csv = cleaned_csv
    train(
        csv_path=chosen_csv,
        out_path=args.out_path,
        convert_mmol_to_mgdl=args.mmol,
        show_plots=(not args.no_show),
        use_feature_engineering=(not args.no_feature_engineering),
        use_ensemble=(not args.no_ensemble),
        ensemble_type=args.ensemble_type,
        top_n_models=args.top_n,
        log_target=args.log_target,
        fast=args.fast,
        n_iter=args.n_iter,
    )
