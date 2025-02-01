import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, KFold
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EyeGlucoseModel:
    def __init__(self, labels_file="eye_glucose_data/labels.csv", model_file="eye_glucose_model.pkl"):
        """
        Initialize the EyeGlucoseModel training class.
        
        Args:
            labels_file: Path to the CSV file with data.
            model_file: Path to save the best model.
        """
        self.labels_file = labels_file
        self.model_file = model_file
        self.best_model = None

    def remove_outliers(self, df):
        """
        Remove outliers based on all variables except 'blood_glucose'.
        If a row contains an outlier in any other variable, the entire row is removed.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            DataFrame with outliers removed.
        """
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove("blood_glucose")  # Exclude blood_glucose from outlier detection
        
        # Identify outliers using Z-score
        z_scores = np.abs((df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / df_clean[numeric_cols].std())
        outliers_z = (z_scores > 3).any(axis=1)  # Flag rows where any variable is an outlier
        
        # Identify outliers using IQR
        Q1 = df_clean[numeric_cols].quantile(0.25)
        Q3 = df_clean[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = ((df_clean[numeric_cols] < (Q1 - 1.5 * IQR)) | (df_clean[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        
        # Combine both methods to determine outliers
        outliers_combined = outliers_z | outliers_iqr
        df_clean = df_clean[~outliers_combined].copy()
        
        removed_count = len(df) - len(df_clean)
        logging.info(f"Removed {removed_count} rows due to outliers in at least one non-blood_glucose variable")
        
        return df_clean

    def prepare_data(self):
        """
        Load and prepare data for training.
        
        Returns:
            A tuple (X, y) where X contains the features and y the target variable.
        """
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Data file not found: {self.labels_file}")
        
        df = pd.read_csv(self.labels_file)
        df = self.remove_outliers(df)
        
        if len(df) < 5:
            raise ValueError(f"Not enough data to train (found {len(df)} rows). Need at least 5 rows.")
        
        # Feature engineering: add time_of_day from timestamp if available
        if 'timestamp' in df.columns:
            df['time_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Split features and target
        y = df["blood_glucose"].astype(float)
        X = df.drop(columns=["blood_glucose"])
        
        # Remove non-numeric columns with a warning
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            logging.warning(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
            X = X.drop(columns=non_numeric_cols)
            
        return X, y

# Additional methods remain unchanged

    def get_model_configurations(self):
        """
        Define models and their hyperparameter search spaces.
        
        Returns:
            A dictionary with model configurations.
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
            }
        }
        return models

    def plot_learning_curve(self, estimator, X_train, y_train, model_name):
        """
        Plot the learning curve (training and validation loss) for the given estimator.
        
        Args:
            estimator: The estimator to evaluate.
            X_train: Training features.
            y_train: Training target.
            model_name: Name of the model (for plot title).
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

    def train_model(self):
        """
        Train and evaluate models using cross-validation and hyperparameter tuning.
        For each model, displays:
          - Scatter plot of actual vs. predicted values.
          - Learning curve (training and validation loss).
        
        After evaluating all models, the best one is saved.
        """
        X, y = self.prepare_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Set whether to include polynomial features (set to False if you prefer not to)
        use_poly_features = False

        # Build the preprocessing pipeline
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

        best_score = float('-inf')
        best_model_name = None
        best_estimator = None

        models = self.get_model_configurations()
        # Increase the number of iterations for a more thorough search.
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
            logging.info(f"Best parameters for {name}: {search.best_params_}")
            logging.info(f"Metrics for {name}:")
            logging.info(f"  R² Score: {r2:.5f}")
            logging.info(f"  MSE: {mse:.5f}")
            logging.info(f"  MAE: {mae:.5f}")
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

        logging.info(f"\nBest Model: {best_model_name} with R² Score: {best_score:.5f}")
        self.best_model = best_estimator
        self.save_model()

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
