import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import logging

from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, KFold
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#########################################
# Helper functions to compute extra metrics
#########################################

def compute_mard(y_true, y_pred):
    epsilon = 1e-8  # small number to avoid division-by-zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def compute_sigma_level(y_true, y_pred, TEa=15):
    errors = y_pred - y_true
    bias = np.mean(errors)
    sd = np.std(errors)
    if sd == 0:
        return float('inf')
    return (TEa - np.abs(bias)) / sd

#########################################
# Main Model Training Class
#########################################

class EyeGlucoseModel:
    def __init__(self, labels_file="eye_glucose_data/labels.csv", model_file="eye_glucose_model.pkl"):
        self.labels_file = labels_file
        self.model_file = model_file
        self.best_model = None

    def remove_outliers(self, df):
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if "blood_glucose" in numeric_cols:
            numeric_cols.remove("blood_glucose")
        z_scores = np.abs((df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / df_clean[numeric_cols].std())
        outliers_z = (z_scores > 3).any(axis=1)
        df_clean = df_clean[~outliers_z].copy()
        removed_count = len(df) - len(df_clean)
        logging.info(f"Removed {removed_count} rows due to outliers (Z-score > 3) in at least one non-blood_glucose variable")
        return df_clean

    def prepare_data(self):
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Data file not found: {self.labels_file}")
        df = pd.read_csv(self.labels_file)
        df = self.remove_outliers(df)
        if len(df) < 5:
            raise ValueError(f"Not enough data to train (found {len(df)} rows). Need at least 5 rows.")
        
        if 'timestamp' in df.columns:
            df['time_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
        
        y = df["blood_glucose"].astype(float)
        reduced_features = [
            'pupil_size', 
            'sclera_redness', 
            'vein_prominence', 
            'pupil_response_time', 
            'ir_intensity',
            'scleral_vein_density', 
            'ir_temperature', 
            'tear_film_reflectivity', 
            'pupil_dilation_rate', 
            'sclera_color_balance', 
            'vein_pulsation_intensity', 
            'birefringence_index'
        ]
        X = df[reduced_features]
    
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            logging.warning(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
            X = X.drop(columns=non_numeric_cols)
            
        return X, y

    def get_model_configurations(self):
        """
        We restrict the search to models that produce more continuously responsive outputs.
        Tree-based models are removed. We tune ElasticNet (which worked before),
        SVR, and a Neural Network.
        """
        models = {
            "ElasticNet": {
                "model": ElasticNet(),
                "params": {
                    "alpha": uniform(0.0001, 0.1),  # smaller range to avoid over-regularization
                    "l1_ratio": uniform(0, 1)
                }
            },
            "SVR": {
                "model": SVR(),
                "params": {
                    "C": uniform(0.1, 10.0),
                    "epsilon": uniform(0.01, 1.0),
                    "kernel": ["linear", "rbf"]
                }
            },
            "Neural Network": {
                "model": MLPRegressor(max_iter=1000, early_stopping=True, random_state=42),
                "params": {
                    "hidden_layer_sizes": [(64, 32), (128, 64), (64, 64, 32)],
                    "activation": ["relu", "tanh"],
                    "alpha": uniform(0.0001, 0.01),
                    "learning_rate_init": uniform(0.0001, 0.01)
                }
            }
        }
        return models

    def plot_learning_curve(self, estimator, X_train, y_train, model_name):
        train_sizes, train_scores, val_scores = learning_curve(
            estimator,
            X_train,
            y_train,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_loss = -np.mean(train_scores, axis=1)
        val_loss = -np.mean(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_loss, 'o-', color="r", label="Training Loss")
        plt.plot(train_sizes, val_loss, 'o-', color="g", label="Validation Loss")
        plt.xlabel("Training Set Size")
        plt.ylabel("Mean Squared Error")
        plt.title(f"Learning Curve for {model_name}")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def save_metrics_to_file(self, best_model_name, metrics, cgm_benchmarks, filename="best_model_metrics.csv"):
        rows = [
            {"Metric": "R²", "Best Model Value": metrics["R2"], "CGM Benchmark": cgm_benchmarks["R²"]},
            {"Metric": "MSE", "Best Model Value": metrics["MSE"], "CGM Benchmark": cgm_benchmarks["MSE"]},
            {"Metric": "MAE", "Best Model Value": metrics["MAE"], "CGM Benchmark": cgm_benchmarks["MAE"]},
            {"Metric": "MARD", "Best Model Value": metrics["MARD"], "CGM Benchmark": cgm_benchmarks["MARD"]},
            {"Metric": "Sigma Level", "Best Model Value": metrics["Sigma"], "CGM Benchmark": cgm_benchmarks["Sigma Level"]}
        ]
        df_metrics = pd.DataFrame(rows)
        with open(filename, 'w') as f:
            f.write(f"Best Model: {best_model_name}\n")
        df_metrics.to_csv(filename, mode='a', index=False)
        logging.info(f"Best model metrics saved to: {filename}")

    def train_model(self):
        X, y = self.prepare_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Preprocessing Pipeline ---
        use_poly_features = False
        use_robust_scaler = False  # Change to True if needed.
        scaler = RobustScaler() if use_robust_scaler else StandardScaler()

        if use_poly_features:
            preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', scaler),
                ('poly', PolynomialFeatures(degree=2, include_bias=False))
            ])
        else:
            preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', scaler)
            ])

        best_score = float('-inf')
        best_model_name = None
        best_estimator = None
        best_r2 = None
        best_mse = None
        best_mae = None
        best_mard = None
        best_sigma = None

        models = self.get_model_configurations()
        n_iter_search = 100

        for name, config in models.items():
            logging.info(f"\nTraining {name}...")
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', config['model'])
            ])
            search = RandomizedSearchCV(
                pipeline,
                {f"regressor__{key}": value for key, value in config['params'].items()},
                n_iter=n_iter_search,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42
            )
            search.fit(X_train, y_train)
            y_pred = search.predict(X_val)
            
            r2 = r2_score(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            mard_value = compute_mard(y_val.values, y_pred)
            sigma_value = compute_sigma_level(y_val.values, y_pred, TEa=15)
            
            logging.info(f"Best parameters for {name}: {search.best_params_}")
            logging.info(f"Metrics for {name}:")
            logging.info(f"  R² Score: {r2:.5f}")
            logging.info(f"  MSE: {mse:.5f}")
            logging.info(f"  MAE: {mae:.5f}")
            logging.info(f"  MARD: {mard_value:.5f}%")
            logging.info(f"  Sigma Level: {sigma_value:.5f}")

            plt.figure(figsize=(10, 6))
            plt.scatter(y_val, y_pred, alpha=0.5)
            plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"{name}: Actual vs. Predicted Values")
            plt.show()
            self.plot_learning_curve(search.best_estimator_, X_train, y_train, name)
            
            if r2 > best_score:
                best_score = r2
                best_estimator = search.best_estimator_
                best_model_name = name
                best_r2 = r2
                best_mse = mse
                best_mae = mae
                best_mard = mard_value
                best_sigma = sigma_value

        logging.info(f"\nBest Model: {best_model_name} with R² Score: {best_score:.5f}")
        self.best_model = best_estimator
        self.save_model()

        cgm_benchmarks = {
            "R²": 0.94,
            "MSE": 6.2,
            "MAE": 2.1,
            "MARD": 10.5,
            "Sigma Level": 3.0
        }

        best_metrics = {
            "R2": best_r2,
            "MSE": best_mse,
            "MAE": best_mae,
            "MARD": best_mard,
            "Sigma": best_sigma
        }
        self.save_metrics_to_file(best_model_name, best_metrics, cgm_benchmarks, filename="best_model_metrics.csv")

    def save_model(self):
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        joblib.dump(self.best_model, self.model_file)
        logging.info(f"Best model saved as: {self.model_file}")

if __name__ == "__main__":
    model_trainer = EyeGlucoseModel()
    model_trainer.train_model()
