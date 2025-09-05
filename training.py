import os
import logging
import warnings

import numpy as np
import pandas as pd
import joblib

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from scipy.stats import uniform, randint

# Optional boosters
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

# Global CV strategy
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

def compute_mard(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

def compute_sigma_level(y_true, y_pred, TEa=15):
    mard = compute_mard(y_true, y_pred)
    return float(TEa / (mard + 1e-8))

class EyeGlucoseModel:
    def __init__(self, csv_path="labels.csv"):
        self.best_model = None
        self.csv_path = csv_path

    def prepare_data(self):
        """
        Loads labels.csv if present; otherwise falls back to dummy data.
        - Uses 'blood_glucose' as target if that column exists; otherwise last numeric column.
        - Keeps only numeric features.
        """
        if os.path.exists(self.csv_path):
            logging.info(f"Loading dataset from: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            # Keep only numeric columns
            num = df.select_dtypes(include=[np.number]).copy()
            if num.empty or num.shape[1] < 2:
                raise ValueError("Dataset must contain at least one numeric feature and one numeric target.")
            if "blood_glucose" in num.columns:
                y = num["blood_glucose"].astype(float)
                X = num.drop(columns=["blood_glucose"])
            else:
                # Use last numeric column as target if not specified
                y = num.iloc[:, -1].astype(float)
                X = num.iloc[:, :-1]
            # Basic sanity logs
            logging.info(f"Rows: {len(df)}, numeric features: {X.shape[1]}, target name: {y.name}")
            return X, y
        else:
            logging.warning("labels.csv not found—using dummy synthetic data.")
            n = 200
            X = pd.DataFrame({"feature1": np.random.randn(n), "feature2": np.random.randn(n)})
            y = X["feature1"] * 10 + X["feature2"] * 5 + np.random.randn(n)
            return X, y

    def get_model_configurations(self):
        models = {
            "SVR": {
                "model": SVR(),
                "params": {
                    "C": uniform(0.1, 10.0),
                    "epsilon": uniform(0.01, 1.0),
                    "kernel": ["linear", "rbf", "poly"],
                },
            },
            "Random Forest": {
                "model": RandomForestRegressor(),
                "params": {
                    "n_estimators": randint(50, 200),
                    "max_depth": randint(3, 40),
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf": randint(1, 10),
                    "max_features": ["sqrt", "log2", None],
                },
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(),
                "params": {
                    "n_estimators": randint(50, 200),
                    "learning_rate": uniform(0.01, 0.2),
                    "max_depth": randint(2, 8),
                    "subsample": uniform(0.7, 0.3),
                    "max_features": ["sqrt", "log2", None],
                },
            },
            "Neural Network": {
                "model": MLPRegressor(
                    max_iter=2000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42,
                ),
                "params": {
                    "hidden_layer_sizes": [(64, 32), (128, 64)],
                    "activation": ["relu", "tanh"],
                    "alpha": uniform(0.0001, 0.01),
                    "learning_rate_init": uniform(0.0001, 0.01),
                },
            },
        }
        if XGBRegressor is not None:
            models["XGBoost"] = {
                "model": XGBRegressor(objective="reg:squarederror", verbosity=0, random_state=42),
                "params": {
                    "n_estimators": randint(100, 500),
                    "learning_rate": uniform(0.01, 0.3),
                    "max_depth": randint(3, 15),
                    "subsample": uniform(0.6, 0.4),
                    "colsample_bytree": uniform(0.6, 0.4),
                },
            }
        if LGBMRegressor is not None:
            models["LightGBM"] = {
                "model": LGBMRegressor(random_state=42, verbosity=-1),
                "params": {
                    "n_estimators": randint(50, 200),
                    "learning_rate": uniform(0.01, 0.2),
                    "max_depth": randint(-1, 12),
                    "num_leaves": randint(16, 256),
                    "min_child_samples": randint(5, 40),
                    "min_split_gain": uniform(0.0, 0.02),
                    "min_data_in_bin": randint(1, 20),
                    "subsample": uniform(0.7, 0.3),
                    "colsample_bytree": uniform(0.7, 0.3),
                    "reg_alpha": uniform(0.0, 0.2),
                    "reg_lambda": uniform(0.0, 0.2),
                    "force_row_wise": [True],
                },
            }
        if CatBoostRegressor is not None:
            models["CatBoost"] = {
                "model": CatBoostRegressor(random_state=42, silent=True),
                "params": {
                    "iterations": randint(100, 500),
                    "learning_rate": uniform(0.01, 0.3),
                    "depth": randint(3, 15),
                },
            }
        return models

    def plot_learning_curve(self, estimator, X_train, y_train, model_name):
        train_sizes, train_scores, val_scores = learning_curve(
            estimator,
            X_train,
            y_train,
            scoring="neg_mean_squared_error",
            cv=cv5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
        )
        train_loss = -np.mean(train_scores, axis=1)
        val_loss = -np.mean(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_loss, 'o-', label="Training Loss")
        plt.plot(train_sizes, val_loss, 'o-', label="Validation Loss")
        plt.xlabel("Training Set Size")
        plt.ylabel("Mean Squared Error")
        plt.title(f"Learning Curve for {model_name}")
        plt.legend(loc="best")
        plt.grid(True)
        plt.close()

    def save_metrics_to_csv(self, best_model_name, metrics, cgm_benchmarks,
                             best_model_details, best_model_params, filename="best_model_metrics.csv"):
        rows = [
            {"Metric": "Best Model Name", "Value": best_model_name, "CGM Benchmark": ""},
            {"Metric": "Best Model Details", "Value": best_model_details, "CGM Benchmark": ""},
        ]
        rows.append({"Metric": "Best Model Hyperparameters", "Value": "", "CGM Benchmark": ""})
        for param, value in best_model_params.items():
            rows.append({"Metric": f"  {param}", "Value": value, "CGM Benchmark": ""})

        rows.append({"Metric": "R²", "Value": metrics["R2"], "CGM Benchmark": cgm_benchmarks.get("R²", "")})
        rows.append({"Metric": "MSE", "Value": metrics["MSE"], "CGM Benchmark": cgm_benchmarks.get("MSE", "")})
        rows.append({"Metric": "MAE", "Value": metrics["MAE"], "CGM Benchmark": cgm_benchmarks.get("MAE", "")})
        rows.append({"Metric": "MARD", "Value": metrics["MARD"], "CGM Benchmark": cgm_benchmarks.get("MARD", "")})
        rows.append({"Metric": "Sigma Level", "Value": metrics["Sigma"], "CGM Benchmark": cgm_benchmarks.get("Sigma Level", "")})

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        logging.info(f"Best model metrics and details saved to: {filename}")

    def train_model(self):
        X, y = self.prepare_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        use_poly_features = False
        use_robust_scaler = False

        scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        preprocessor_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
        if use_poly_features:
            preprocessor_steps.insert(1, ("poly", PolynomialFeatures(degree=2, include_bias=False)))
        preprocessor = Pipeline(preprocessor_steps)

        models = self.get_model_configurations()
        n_iter_search = 2

        best_score = -np.inf
        best_estimator = None
        best_model_name = ""
        best_r2 = best_mse = best_mae = best_mard = best_sigma = None
        best_model_params = {}

        for name, config in models.items():
            logging.info(f"\nTraining {name}...")
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", config["model"]),
            ])
            param_grid = {f"regressor__{p}": v for p, v in config["params"].items()}

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=n_iter_search,
                cv=cv5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                random_state=42,
            )
            fit_params = {}
            search.fit(X_train, y_train, **fit_params)
            y_pred = search.predict(X_val)

            yv = np.asarray(y_val)
            r2 = r2_score(yv, y_pred)
            mse = mean_squared_error(yv, y_pred)
            mae = mean_absolute_error(yv, y_pred)
            mard_value = compute_mard(yv, y_pred)
            sigma_value = compute_sigma_level(yv, y_pred, TEa=15)

            logging.info(f"Best parameters for {name}: {search.best_params_}")
            logging.info(f"Metrics for {name}: R²={r2:.5f} | MSE={mse:.5f} | MAE={mae:.5f} | MARD={mard_value:.5f}% | Sigma={sigma_value:.5f}")

            # Parity plot (closed figure)
            plt.figure(figsize=(10, 6))
            plt.scatter(yv, y_pred, alpha=0.5)
            mn, mx = float(np.min(yv)), float(np.max(yv))
            plt.plot([mn, mx], [mn, mx], '--', lw=2)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"{name}: Actual vs. Predicted Values")
            plt.close()

            self.plot_learning_curve(search.best_estimator_, X_train, y_train, name)

            if (best_mse is None) or (mse < best_mse):
                best_score = r2
                best_estimator = search.best_estimator_
                best_model_name = name
                best_r2, best_mse, best_mae = r2, mse, mae
                best_mard, best_sigma = mard_value, sigma_value
                best_model_params = search.best_params_

        logging.info(f"\nBest Model: {best_model_name} with R² Score: {best_score:.5f}")
        self.best_model = best_estimator
        self.save_model()

        cgm_benchmarks = {"R²": 0.94, "MSE": 6.2, "MAE": 2.1, "MARD": 10.5, "Sigma Level": 3.0}
        best_metrics = {"R2": best_r2, "MSE": best_mse, "MAE": best_mae, "MARD": best_mard, "Sigma": best_sigma}
        best_model_details = str(best_estimator)
        self.save_metrics_to_csv(
            best_model_name, best_metrics, cgm_benchmarks, best_model_details, best_model_params,
            filename="best_model_metrics.csv",
        )

    def save_model(self, path="best_model.pkl"):
        if getattr(self, "best_model", None) is None:
            logging.warning("No model to save.")
            return
        joblib.dump(self.best_model, path)
        logging.info(f"Saved best model to {path}")

if __name__ == "__main__":
    # Minimal CLI: allow overriding the csv path:  python training.py /path/to/labels.csv
    import sys
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "labels.csv"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        model = EyeGlucoseModel(csv_path=csv_arg)
        model.train_model()
    except Exception as e:
        logging.exception("Fatal error during training")
        raise
