import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#########################################
# Helper functions
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
# Custom Gradient Boosting Neural Network Regressor
#########################################

class GradientBoostingNNRegressor:
    def __init__(self, n_estimators=50, learning_rate=0.1, base_params=None, random_state=None):
        """
        A simple gradient boosting regressor that uses MLPRegressor as the base learner.
        
        Parameters:
            n_estimators (int): Number of boosting iterations.
            learning_rate (float): Shrinkage factor applied to each learner.
            base_params (dict): Hyperparameters to pass to each MLPRegressor.
            random_state (int): Seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_params = base_params if base_params is not None else {}
        self.random_state = random_state
        self.models = []
        self.initial_prediction = None

    def fit(self, X, y):
        # Initialize predictions with the mean of y
        self.initial_prediction = np.mean(y)
        current_prediction = np.full_like(y, self.initial_prediction, dtype=float)

        for i in range(self.n_estimators):
            # Compute the residuals
            residuals = y - current_prediction
            
            # Initialize and train a new neural network on the residuals
            mlp = MLPRegressor(random_state=self.random_state, max_iter=2000, **self.base_params)
            mlp.fit(X, residuals)
            self.models.append(mlp)
            
            # Update current predictions using the new model’s output
            update = mlp.predict(X)
            current_prediction += self.learning_rate * update
            
            logging.info(f"Trained estimator {i+1}/{self.n_estimators}")

        return self

    def predict(self, X):
        # Start with the initial prediction
        y_pred = np.full((X.shape[0],), self.initial_prediction, dtype=float)
        # Add contributions from each trained neural network
        for mlp in self.models:
            y_pred += self.learning_rate * mlp.predict(X)
        return y_pred


#########################################
# Main Model Training Class using Gradient Boosting Neural Network
#########################################

class EyeGlucoseModel:
    def __init__(self, labels_file="eye_glucose_data/labels.csv", model_file="eye_glucose_gbnn_model.pkl"):
        self.labels_file = labels_file
        self.model_file = model_file
        self.model = None

    def remove_outliers(self, df):
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if "blood_glucose" in numeric_cols:
            numeric_cols.remove("blood_glucose")
        z_scores = np.abs((df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / df_clean[numeric_cols].std())
        outliers_z = (z_scores > 6).any(axis=1)
        df_clean = df_clean[~outliers_z].copy()
        removed_count = len(df) - len(df_clean)
        logging.info(f"Removed {removed_count} rows due to outliers (Z-score > 6) in at least one non-blood_glucose variable")
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

    def train_model(self):
        X, y = self.prepare_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Preprocessing Pipeline ---
        use_robust_scaler = False  # Change to True if needed
        scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])

        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)

        # Define hyperparameters for the base neural network
        base_params = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "alpha": 0.0001,
            "learning_rate_init": 0.001
        }

        # Create and train the Gradient Boosting Neural Network
        gbnn = GradientBoostingNNRegressor(
            n_estimators=50,
            learning_rate=0.1,
            base_params=base_params,
            random_state=42
        )
        logging.info("Training Gradient Boosting Neural Network...")
        gbnn.fit(X_train_processed, y_train.values)
        y_pred = gbnn.predict(X_val_processed)

        # Evaluate performance
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mard_value = compute_mard(y_val.values, y_pred)
        sigma_value = compute_sigma_level(y_val.values, y_pred, TEa=15)

        logging.info("Gradient Boosting Neural Network Performance:")
        logging.info(f"  R² Score: {r2:.5f}")
        logging.info(f"  MSE: {mse:.5f}")
        logging.info(f"  MAE: {mae:.5f}")
        logging.info(f"  MARD: {mard_value:.5f}%")
        logging.info(f"  Sigma Level: {sigma_value:.5f}")

        # Plot Actual vs. Predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_val, y_pred, alpha=0.5)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        plt.xlabel("Actual Blood Glucose")
        plt.ylabel("Predicted Blood Glucose")
        plt.title("Gradient Boosting Neural Network: Actual vs. Predicted")
        plt.show()

        # Optionally, you can plot a learning curve here
        # self.plot_learning_curve(gbnn, X_train_processed, y_train, "Gradient Boosting NN")
        
        self.model = gbnn
        self.save_model()

    def save_model(self):
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        joblib.dump(self.model, self.model_file)
        logging.info(f"Model saved as: {self.model_file}")

if __name__ == "__main__":
    model_trainer = EyeGlucoseModel()
    model_trainer.train_model()
