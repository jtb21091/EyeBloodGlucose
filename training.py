import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

class EyeGlucoseModel:
    def __init__(self, labels_file="eye_glucose_data/labels.csv", model_file="eye_glucose_model.pkl"):
        self.labels_file = labels_file
        self.model_file = model_file
        self.best_model = None
        self.scaler = None
        self.imputer = None
        
    def remove_outliers(self, df, column="blood_glucose", n_std=3):
        """Remove outliers using both IQR and standard deviation methods."""
        # Z-score method
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df_no_outliers = df[z_scores < n_std].copy()
        
        # IQR method
        Q1 = df_no_outliers[column].quantile(0.25)
        Q3 = df_no_outliers[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_no_outliers[column] = df_no_outliers[column].clip(lower=lower_bound, upper=upper_bound)
        
        removed_count = len(df) - len(df_no_outliers)
        print(f"Removed {removed_count} outliers using combined Z-score and IQR methods")
        return df_no_outliers

    def prepare_data(self):
        """Load and prepare the data for training."""
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Data file not found: {self.labels_file}")
        
        df = pd.read_csv(self.labels_file)
        df = self.remove_outliers(df)
        
        if len(df) < 5:
            raise ValueError(f"Not enough data to train (found {len(df)} rows). Need at least 5 rows.")
        
        # Feature engineering (example - add your own based on domain knowledge)
        df['time_of_day'] = pd.to_datetime(df['timestamp']).dt.hour if 'timestamp' in df.columns else None
        
        # Split features and target
        y = df["blood_glucose"].astype(float)
        X = df.drop(columns=["blood_glucose"])
        
        # Remove non-numeric columns with warning
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            print(f"⚠️ Warning: Dropping non-numeric columns: {list(non_numeric_cols)}")
            X = X.drop(columns=non_numeric_cols)
            
        return X, y

    def get_model_configurations(self):
        """Define models with hyperparameter search spaces."""
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
                    "max_depth": randint(5, 30),
                    "min_samples_split": randint(2, 10),
                    "min_samples_leaf": randint(1, 4)
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(),
                "params": {
                    "n_estimators": randint(100, 500),
                    "learning_rate": uniform(0.01, 0.3),
                    "max_depth": randint(3, 10),
                    "subsample": uniform(0.6, 1.0)
                }
            },
            "Neural Network": {
                "model": MLPRegressor(max_iter=1000, early_stopping=True),
                "params": {
                    "hidden_layer_sizes": [(64, 32), (128, 64), (64, 64, 32)],
                    "activation": ["relu", "tanh"],
                    "alpha": uniform(0.0001, 0.01),
                    "learning_rate_init": uniform(0.0001, 0.01)
                }
            }
        }
        return models

    def train_model(self):
        """Train and evaluate models using cross-validation and hyperparameter tuning."""
        X, y = self.prepare_data()
        
        # Create training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create preprocessing pipeline
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        best_score = float('-inf')
        best_model_name = None
        models = self.get_model_configurations()
        
        for name, config in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline with preprocessing and model
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', config['model'])
            ])
            
            # Perform randomized search with cross-validation
            search = RandomizedSearchCV(
                pipeline,
                {f"regressor__{key}": value for key, value in config['params'].items()},
                n_iter=20,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                random_state=42
            )
            
            search.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = search.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            
            print(f"Best parameters: {search.best_params_}")
            print(f"Metrics for {name}:")
            print(f"R² Score: {r2:.5f}")
            print(f"MSE: {mse:.5f}")
            print(f"MAE: {mae:.5f}")
            
            # Plot actual vs predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(y_val, y_pred, alpha=0.5)
            plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"{name}: Actual vs Predicted Values")
            plt.show()
            
            if r2 > best_score:
                best_score = r2
                self.best_model = search.best_estimator_
                best_model_name = name
        
        print(f"\nBest Model: {best_model_name} with R² Score: {best_score:.5f}")
        self.save_model()
        
    def save_model(self):
        """Save the trained model and preprocessing pipeline."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        
        joblib.dump(self.best_model, self.model_file)
        print(f"Best model saved as: {self.model_file}")

if __name__ == "__main__":
    model_trainer = EyeGlucoseModel()
    model_trainer.train_model()