import pandas as pd
import zipfile
import urllib.request
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Download and extract the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
zip_path = "student.zip"

# Download the file
urllib.request.urlretrieve(url, zip_path)

# Extract only 'student-mat.csv'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extract('student-mat.csv')

# Step 2: Load the CSV file (semicolon-separated)
df = pd.read_csv('student-mat.csv', sep=';')

# Step 3: Preview the data
print(df.head())



# Selected features for the model
selected_columns = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
    'Medu', 'Fedu', 'Mjob', 'Fjob', 'traveltime', 'studytime', 
    'failures', 'schoolsup', 'famsup', 'paid', 'activities', 
    'nursery', 'higher', 'internet', 'romantic', 'famrel', 
    'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
    'G1', 'G2', 'G3'
]

# Create a new dataframe with only selected columns
df_selected = df[selected_columns]

# Check the shape
print(f"Original dataset shape: {df.shape}")
print(f"Selected dataset shape: {df_selected.shape}")

# Let's first identify categorical vs numerical columns
categorical_columns = []
numerical_columns = []

for col in df_selected.columns:
    if df_selected[col].dtype == 'object':  # String/categorical columns
        categorical_columns.append(col)
    else:  # Numerical columns
        numerical_columns.append(col)

print("Categorical columns (need one-hot encoding):")
print(categorical_columns)
print("\nNumerical columns:")
print(numerical_columns)

# Step 1: One-Hot Encoding for categorical columns
print("Before one-hot encoding:", df_selected.shape)

# Apply one-hot encoding to categorical columns
df_encoded = pd.get_dummies(df_selected, columns=categorical_columns, drop_first=True)

print("After one-hot encoding:", df_encoded.shape)
print("\nFirst few columns after encoding:")
print(df_encoded.columns.tolist()[:10])  # Show first 10 columns
print("...")
print(df_encoded.columns.tolist()[-10:])  # Show last 10 columns

# Show a sample of the encoded data
print("\nSample of encoded data:")
print(df_encoded.head())


# Step 2: Separate features and target, then apply StandardScaler

# Separate target variable (G3) from features
y = df_encoded['G3']  # Target variable
X = df_encoded.drop('G3', axis=1)  # All features except G3

print("Target variable (G3) shape:", y.shape)
print("Features (X) shape:", X.shape)

# Identify numerical columns that need scaling (exclude G3 since it's now the target)
numerical_features_to_scale = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'
]

# Get one-hot encoded columns (these don't need scaling - they're already 0/1)
categorical_features_encoded = [col for col in X.columns if col not in numerical_features_to_scale]

print(f"\nNumerical features to scale ({len(numerical_features_to_scale)}): {numerical_features_to_scale}")
print(f"Categorical features (already scaled) ({len(categorical_features_encoded)}): {categorical_features_encoded[:5]}...")  # Show first 5

# Apply StandardScaler to numerical features only
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X[numerical_features_to_scale])

# Convert scaled numerical features back to DataFrame
X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, 
                                   columns=numerical_features_to_scale, 
                                   index=X.index)

# Keep categorical features unchanged
X_categorical = X[categorical_features_encoded]

# Combine scaled numerical + unscaled categorical features
X_final = pd.concat([X_numerical_scaled_df, X_categorical], axis=1)

print(f"\nFinal features shape: {X_final.shape}")
print("Sample of final processed data:")
print(X_final.head())

# Step 3: Train-Test Split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final,     # Features (processed)
    y,           # Target (G3)
    test_size=0.2,      # 20% for testing, 80% for training
    random_state=42,    # For reproducible results
    shuffle=True        # Shuffle the data before splitting
)

# Display the shapes of the splits
print("="*50)
print("TRAIN-TEST SPLIT RESULTS")
print("="*50)
print(f"Original dataset shape: {X_final.shape}")
print(f"Target variable shape: {y.shape}")
print()
print("After splitting:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")
print()
print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X_final)*100:.1f}%)")
print(f"Testing set:  {len(X_test)} samples ({len(X_test)/len(X_final)*100:.1f}%)")
print()
print("Sample of training features:")
print(X_train.head(3))
print()
print("Sample of training targets:")
print(y_train.head(3))



# Lasso Regression with Cross-Validation

# Create LassoCV model with different alpha values to test
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]  # Regularization strengths
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)  # 5-fold cross-validation

# Train the model
lasso_cv.fit(X_train, y_train)

# Make predictions
y_train_pred_lasso = lasso_cv.predict(X_train)
y_test_pred_lasso = lasso_cv.predict(X_test)

# Calculate evaluation metrics for training set
train_r2_lasso = r2_score(y_train, y_train_pred_lasso)
train_mae_lasso = mean_absolute_error(y_train, y_train_pred_lasso)
train_mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
train_rmse_lasso = np.sqrt(train_mse_lasso)

# Calculate evaluation metrics for test set
test_r2_lasso = r2_score(y_test, y_test_pred_lasso)
test_mae_lasso = mean_absolute_error(y_test, y_test_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)
test_rmse_lasso = np.sqrt(test_mse_lasso)

# Analyze feature selection
lasso_coefs = lasso_cv.coef_
features_selected = np.sum(lasso_coefs != 0)
features_eliminated = np.sum(lasso_coefs == 0)

# Display results
print("="*60)
print("LASSO REGRESSION (LassoCV) MODEL PERFORMANCE")
print("="*60)
print(f"Best Alpha (regularization): {lasso_cv.alpha_:.4f}")
print(f"Features selected: {features_selected}/{len(lasso_coefs)}")
print(f"Features eliminated: {features_eliminated}")
print("-"*60)
print(f"{'Metric':<15} {'Training Set':<15} {'Test Set':<15} {'Difference':<15}")
print("-"*60)
print(f"{'R² Score':<15} {train_r2_lasso:<15.4f} {test_r2_lasso:<15.4f} {abs(train_r2_lasso-test_r2_lasso):<15.4f}")
print(f"{'MAE':<15} {train_mae_lasso:<15.4f} {test_mae_lasso:<15.4f} {abs(train_mae_lasso-test_mae_lasso):<15.4f}")
print(f"{'MSE':<15} {train_mse_lasso:<15.4f} {test_mse_lasso:<15.4f} {abs(train_mse_lasso-test_mse_lasso):<15.4f}")
print(f"{'RMSE':<15} {train_rmse_lasso:<15.4f} {test_rmse_lasso:<15.4f} {abs(train_rmse_lasso-test_rmse_lasso):<15.4f}")
print("="*60)



# Feature selection analysis
print(f"\n🎯 FEATURE SELECTION RESULTS:")
print(f"• {features_eliminated} features eliminated (coefficient = 0)")
print(f"• {features_selected} features kept for prediction")
print(f"• Feature reduction: {features_eliminated/len(lasso_coefs)*100:.1f}%")

# Show which features were selected
feature_names = X_train.columns
selected_features = feature_names[lasso_coefs != 0]
eliminated_features = feature_names[lasso_coefs == 0]

print(f"\n📊 SELECTED FEATURES ({len(selected_features)}):")
for i, (feature, coef) in enumerate(zip(selected_features, lasso_coefs[lasso_coefs != 0])):
    if i < 10:  # Show first 10
        print(f"  {feature}: {coef:.4f}")
    elif i == 10:
        print(f"  ... and {len(selected_features)-10} more")
        break

if len(eliminated_features) > 0:
    print(f"\n❌ ELIMINATED FEATURES ({len(eliminated_features)}):")
    print(f"  {list(eliminated_features[:10])}")  # Show first 10
    if len(eliminated_features) > 10:
        print(f"  ... and {len(eliminated_features)-10} more")

# Brief interpretation
print("\n📊 METRIC INTERPRETATION:")
print(f"• R² Score: {test_r2_lasso:.1%} of variance explained")
print(f"• MAE: On average, predictions are off by {test_mae_lasso:.2f} grade points")
print("✅ Model performance summary complete!")

# Visualizations
plt.figure(figsize=(15, 6))

# 1. Line Chart: Actual vs Predicted (Test Set)
plt.subplot(1, 2, 1)
sorted_indices = np.argsort(y_test)
sorted_actual = y_test.iloc[sorted_indices]
sorted_predicted_lasso = y_test_pred_lasso[sorted_indices]

plt.plot(range(len(sorted_actual)), sorted_actual, 'b-', label='Actual', linewidth=2, alpha=0.8)
plt.plot(range(len(sorted_predicted_lasso)), sorted_predicted_lasso, 'orange', linestyle='--', label='Lasso Predicted', linewidth=2, alpha=0.8)
plt.title('Lasso Regression: Actual vs Predicted\nSorted by Actual Values', fontsize=12, fontweight='bold')
plt.xlabel('Sample Index (sorted)')
plt.ylabel('Final Grade (G3)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Scatter Plot: Actual vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred_lasso, alpha=0.6, color='orange')
min_val = min(y_test.min(), y_test_pred_lasso.min())
max_val = max(y_test.max(), y_test_pred_lasso.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.title('Lasso: Actual vs Predicted Scatter Plot', fontsize=12, fontweight='bold')
plt.xlabel('Actual Grade (G3)')
plt.ylabel('Predicted Grade (G3)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



