"""advanced_lasso_pipeline.py
Simple, self-contained script that builds a preprocessing + Lasso pipeline,
performs grid-search CV, evaluates, and plots actual vs predicted.

NOTE: define or load your feature matrix `X1` and target vector `Y1` before
running. Keep the code minimal and readable as requested.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
# Load diabetes dataset
# -----------------------------------------------------------------------------
diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

X1 = diabetes.data
Y1 = diabetes.target

# -----------------------
# Define pipeline
# -----------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(include_bias=False)),
    ("select", SelectKBest(score_func=f_regression)),
    ("lasso", Lasso(max_iter=10000)),
])

# -----------------------
# GridSearch Parameters
# -----------------------
param_grid = {
    "poly__degree": [1, 2, 3],
    "select__k": [5, 7, 10],
    "lasso__alpha": [0.0001, 0.001, 0.01],
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2")
grid.fit(X1, Y1)

# -----------------------
# Best model & predictions
# -----------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X1)

# -----------------------
# Evaluation Metrics
# -----------------------
r2 = r2_score(Y1, y_pred)
mae = mean_absolute_error(Y1, y_pred)
mse = mean_squared_error(Y1, y_pred)
rmse = np.sqrt(mse)

print("Best Parameters:", grid.best_params_)
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# -----------------------
# Line Plot: Actual vs Predicted
# -----------------------
sorted_indices = np.argsort(Y1)
sorted_actual = np.array(Y1)[sorted_indices]
sorted_pred = y_pred[sorted_indices]

plt.figure(figsize=(10, 5))
plt.plot(range(len(sorted_actual)), sorted_actual, label="Actual", color="blue", linewidth=2)
plt.plot(range(len(sorted_pred)), sorted_pred, label="Predicted", color="red", linestyle="--")
plt.xlabel("Sample Index (sorted)")
plt.ylabel("Target (Y1)")
plt.title("Actual vs Predicted (Sorted)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 