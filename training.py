import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import logging

from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, KFold
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint

# Additional model imports:
from sklearn.svm import SVR                   # Support Vector Regressor
from sklearn.neighbors import KNeighborsRegressor  # KNN Regressor
from sklearn.tree import DecisionTreeRegressor     # Decision Tree Regressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#########################################
# Helper functions to compute extra metrics
#########################################

def compute_mard(y_true, y_pred):
    """
    Compute the Mean Absolute Relative Difference (MARD)
    
    Args:
        y_true: Ground truth target values (array-like)
        y_pred: Predicted target values (array-like)
        
    Returns:
        MARD value in percent.
    """
    epsilon = 1e-8  # small number to avoid division-by-zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def compute_sigma_level(y_true, y_pred, TEa=15):
    """
    Compute a Sigma Level metric based on the quality-control formula:
    
        Sigma = (TEa - |bias|) / SD(error)
    
    where TEa is the Total Allowable Error.
    
    Args:
        y_true: Ground truth target values (array-like)
        y_pred: Predicted target values (array-like)
        TEa: Total allowable error (in the same units as y_true)
        
    Returns:
        Sigma level value (if SD(error)==0, returns float('inf')).
    """
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
        """
        Initialize the EyeGlucoseModel training class.
        """
        self.labels_file = labels_file
        self.model_file = model_file
        self.best_model = None

    def remove_outliers(self, df):
        """
        Remove outliers based on non-target numeric features (using a Z-score threshold).
        """
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove("blood_glucose")  # Exclude target from outlier detection
        z_scores = np.abs((df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / df_clean[numeric_cols].std())
        outliers_z = (z_scores > 3).any(axis=1)
        df_clean = df_clean[~outliers_z].copy()
        removed_count = len(df) - len(df_clean)
        logging.info(f"Removed {removed_count} rows due to outliers (Z-score > 3) in at least one non-blood_glucose variable")
        return df_clean

    def prepare_data(self):
        """
        Load data, remove outliers and prepare features.
        """
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Data file not found: {self.labels_file}")
        df = pd.read_csv(self.labels_file)
        df = self.remove_outliers(df)
        if len(df) < 5:
            raise ValueError(f"Not enough data to train (found {len(df)} rows). Need at least 5 rows.")
        
        # Feature engineering: add time_of_day if timestamp available
        if 'timestamp' in df.columns:
            df['time_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
        
        y = df["blood_glucose"].astype(float)
        
        # Define a list of reduced features (adjust as needed)
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
        Define models and hyperparameter search spaces.
        """
        models = {
            "ElasticNet": {
                "model": ElasticNet(),
                "params": {
                    "alpha": uniform(0.0001, 1.0),
                    "l1_ratio": uniform(0, 1)
                }
            },
            "Random Forest": {
                "model": RandomForestRegressor(),
                "params": {
                    "n_estimators": randint(100, 500),
                    "max_depth": randint(5, 50),
                    "min_samples_split": randint(2, 10),
                    "min_samples_leaf": randint(1, 4)
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(),
                "params": {
                    "n_estimators": randint(100, 500),
                    "learning_rate": uniform(0.01, 0.3),
                    "max_depth": randint(3, 15),
                    "subsample": uniform(0.6, 1.0)
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
            },
            # Additional models:
            "SVR": {
                "model": SVR(),
                "params": {
                    "C": uniform(0.1, 10.0),
                    "epsilon": uniform(0.01, 0.5),
                    "kernel": ["linear", "rbf"]
                }
            },
            "KNN Regressor": {
                "model": KNeighborsRegressor(),
                "params": {
                    "n_neighbors": randint(3, 20),
                    "weights": ["uniform", "distance"]
                }
            },
            "Decision Tree": {
                "model": DecisionTreeRegressor(),
                "params": {
                    "max_depth": randint(3, 15),
                    "min_samples_split": randint(2, 10),
                    "min_samples_leaf": randint(1, 5)
                }
            },
            "Stacking Regressor": {
                "model": StackingRegressor(
                    estimators=[
                        ('rf', RandomForestRegressor()),
                        ('gb', GradientBoostingRegressor())
                    ],
                    final_estimator=LinearRegression()
                ),
                "params": {
                    # (Optional tuning for the final estimator could be added)
                }
            }
        }
        return models

    def plot_learning_curve(self, estimator, X_train, y_train, model_name):
        """
        Plot the learning curve (training and validation loss).
        """
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
        """
        Save the best model's metrics and CGM benchmark comparisons to a CSV file.
        
        Args:
            best_model_name: Name of the best model.
            metrics: A dictionary containing the best model's metrics.
            cgm_benchmarks: A dictionary containing CGM benchmark values.
            filename: The output filename (CSV).
        """
        rows = [
            {"Metric": "R²", "Best Model Value": metrics["R2"], "CGM Benchmark": cgm_benchmarks["R²"]},
            {"Metric": "MSE", "Best Model Value": metrics["MSE"], "CGM Benchmark": cgm_benchmarks["MSE"]},
            {"Metric": "MAE", "Best Model Value": metrics["MAE"], "CGM Benchmark": cgm_benchmarks["MAE"]},
            {"Metric": "MARD", "Best Model Value": metrics["MARD"], "CGM Benchmark": cgm_benchmarks["MARD"]},
            {"Metric": "Sigma Level", "Best Model Value": metrics["Sigma"], "CGM Benchmark": cgm_benchmarks["Sigma Level"]}
        ]
        df_metrics = pd.DataFrame(rows)
        # Optionally, include the model name at the top of the file.
        with open(filename, 'w') as f:
            f.write(f"Best Model: {best_model_name}\n")
        df_metrics.to_csv(filename, mode='a', index=False)
        logging.info(f"Best model metrics saved to: {filename}")

    def train_model(self):
        """
        Train and evaluate each model via hyperparameter tuning.
        In addition to R²/MSE/MAE, this version computes MARD and Sigma Level.
        For the Neural Network model, we also demonstrate an epoch-by-epoch loop.
        """
        X, y = self.prepare_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the preprocessing pipeline (optionally including polynomial features)
        use_poly_features = False
        if use_poly_features:
            preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False))
            ])
        else:
            preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

        # These variables will track the best model and its metrics.
        best_score = float('-inf')
        best_model_name = None
        best_estimator = None
        best_r2 = None
        best_mse = None
        best_mae = None
        best_mard = None
        best_sigma = None

        models = self.get_model_configurations()
        n_iter_search = 50

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
                scoring='r2',
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

            # Plot actual vs. predicted and the learning curve
            plt.figure(figsize=(10, 6))
            plt.scatter(y_val, y_pred, alpha=0.5)
            plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"{name}: Actual vs. Predicted Values")
            plt.show()
            self.plot_learning_curve(search.best_estimator_, X_train, y_train, name)

            # For the Neural Network model, additionally show an epoch-by-epoch training loop.
            if name == "Neural Network":
                logging.info("Starting custom epoch loop for Neural Network to track MARD & Sigma Level...")
                # Extract best hyperparameters and reinitialize the regressor
                nn_params = {key.split("regressor__")[1]: value for key, value in search.best_params_.items()}
                nn_model = MLPRegressor(max_iter=1, warm_start=True, early_stopping=False, random_state=42, **nn_params)
                nn_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', nn_model)
                ])
                n_epochs = 50
                mard_epochs = []
                sigma_epochs = []
                for epoch in range(n_epochs):
                    nn_pipeline.fit(X_train, y_train)
                    y_pred_epoch = nn_pipeline.predict(X_val)
                    mard_epoch = compute_mard(y_val.values, y_pred_epoch)
                    sigma_epoch = compute_sigma_level(y_val.values, y_pred_epoch, TEa=15)
                    mard_epochs.append(mard_epoch)
                    sigma_epochs.append(sigma_epoch)
                    logging.info(f"Epoch {epoch+1}/{n_epochs} - MARD: {mard_epoch:.5f}% - Sigma Level: {sigma_epoch:.5f}")
                
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, n_epochs+1), mard_epochs, marker='o', label="MARD")
                plt.xlabel("Epoch")
                plt.ylabel("MARD (%)")
                plt.title("MARD over Epochs (Neural Network)")
                plt.legend()
                plt.grid(True)
                plt.show()

                plt.figure(figsize=(10, 6))
                plt.plot(range(1, n_epochs+1), sigma_epochs, marker='o', label="Sigma Level", color='orange')
                plt.xlabel("Epoch")
                plt.ylabel("Sigma Level")
                plt.title("Sigma Level over Epochs (Neural Network)")
                plt.legend()
                plt.grid(True)
                plt.show()
            
            # Update the best model if this one has a higher R² score.
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

        # Define some example CGM benchmark values (adjust as needed)
        cgm_benchmarks = {
            "R²": 0.94,         # For example, a CGM might be expected to achieve very high correlation.
            "MSE": 6.2,         # Example value: lower MSE is better.
            "MAE": 2.1,         # Example value.
            "MARD": 10.5,       # CGM devices sometimes target a MARD below 10%.
            "Sigma Level": 3.0  # A sigma level of 3 or more may be desired.
        }

        # Collect best model metrics into a dictionary
        best_metrics = {
            "R2": best_r2,
            "MSE": best_mse,
            "MAE": best_mae,
            "MARD": best_mard,
            "Sigma": best_sigma
        }
        # Save the best model's metrics to a CSV file.
        self.save_metrics_to_file(best_model_name, best_metrics, cgm_benchmarks, filename="best_model_metrics.csv")

    def save_model(self):
        """
        Save the best trained model (including the preprocessing pipeline) to disk.
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        joblib.dump(self.best_model, self.model_file)
        logging.info(f"Best model saved as: {self.model_file}")


if __name__ == "__main__":
    model_trainer = EyeGlucoseModel()
    model_trainer.train_model()
