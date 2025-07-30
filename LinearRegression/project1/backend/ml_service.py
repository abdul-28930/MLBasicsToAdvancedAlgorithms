"""ML Service - Simple refactor of existing pipeline"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_diabetes
import numpy as np

class MLService:
    def __init__(self):
        # Load data and train model (same logic as original)
        diabetes = load_diabetes()
        self.feature_names = diabetes.feature_names
        X1 = diabetes.data
        Y1 = diabetes.target
        
        # Same pipeline as original
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(include_bias=False)),
            ("select", SelectKBest(score_func=f_regression)),
            ("lasso", Lasso(max_iter=10000)),
        ])
        
        # Same parameters as original
        param_grid = {
            "poly__degree": [1, 2, 3],
            "select__k": [5, 7, 10],
            "lasso__alpha": [0.0001, 0.001, 0.01],
        }
        
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2")
        grid.fit(X1, Y1)
        
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        
        # Calculate metrics (same as original)
        y_pred = self.model.predict(X1)
        self.metrics = {
            "r2": float(r2_score(Y1, y_pred)),
            "mae": float(mean_absolute_error(Y1, y_pred)),
            "mse": float(mean_squared_error(Y1, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(Y1, y_pred)))
        }
    
    def predict(self, features):
        """Simple prediction method"""
        prediction = self.model.predict([features])
        return float(prediction[0])
    
    def get_metrics(self):
        """Return model metrics"""
        return self.metrics 