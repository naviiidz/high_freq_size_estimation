import pandas as pd
import numpy as np
import h5py
from scipy.signal import hilbert
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Read ground truth labels (bounding box data)
csv_name = 'ground_truth_labels.csv'
pd_input = pd.read_csv(csv_name)

# Extract bounding box data
x_coords = pd_input['x']
y_coords = pd_input['y']
widths = pd_input['width']
heights = pd_input['height']
d = pd_input['diameter']  # Particle sizes (target variable)

# Load RF data (signals)
directory = '20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_glass_3200x3000x10um/40MHz_glass_3200x3000x10um_TXT/40MHz_glass_3200x3000x10umVZ000R00000/40MHz_glass_3200x3000x10um.mat'
with h5py.File(directory, 'r') as file:
    RF = np.array(file['RFdata']['RF']).transpose()  # Transpose as per your original code
    RFtime = np.array(file['RFdata']['RFtime']).transpose()

# Function to extract features from the signal
def extract_features(signal):
    features = {}
    mean_val = np.mean(signal)
    if mean_val == 0:
        return None
    features['peak_to_peak'] = np.ptp(signal)
    return features

# Collect signals and corresponding particle sizes (labels)
collected_features = []
labels_for_signals = []

# Extract features for the top 5 signals within each bounding box
for x, y, width, height, diameter in zip(x_coords, y_coords, widths, heights, d):
    signal_strengths = []
    
    # Extract signals from RF data within the bounding boxes
    for i in range(int(height)):
        for j in range(int(width)):
            signal = RF[:, x + j, y + i]
            strength = np.ptp(signal)  # Use peak-to-peak amplitude as signal strength
            signal_strengths.append((strength, signal))
    
    # Sort signals by strength and select the top 5
    signal_strengths.sort(key=lambda x: x[0], reverse=True)  # Sort in descending order
    top_signals = signal_strengths[:3]  # Get the top 5 strongest signals
    
    # Extract features for the selected top signals
    for strength, signal in top_signals:
        features = extract_features(signal)
        if features is not None:  # Ensure features were extracted
            collected_features.append(features)
            labels_for_signals.append(diameter)

# Convert collected features and labels to numpy arrays
X = pd.DataFrame(collected_features)  # Features DataFrame
y = np.array(labels_for_signals)  # Labels array

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class Distribution:", class_distribution)

# Determine the size of the class with the minimum instances
min_class_size = min(counts)
print(f"Minimum class size: {min_class_size}")

# Collect instances for each class
X_balanced = []
y_balanced = []

for diameter in unique:
    # Create a mask for the current class
    mask = (y == diameter)
    X_class = X[mask]
    y_class = y[mask]

    # Randomly select instances from the current class
    X_selected, y_selected = resample(
        X_class,
        y_class,
        replace=False,  # No replacement for random sampling
        n_samples=min_class_size,  # Match the minimum class size
        random_state=35
    )

    # Append selected data to the balanced lists
    X_balanced.append(X_selected)
    y_balanced.append(y_selected)

# Convert the list of balanced data back to arrays
X_balanced = np.vstack(X_balanced)
y_balanced = np.concatenate(y_balanced)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=505)

# Train SVR model
svr_model = SVR(kernel='rbf')  # Using Radial Basis Function kernel for non-linearity
svr_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = svr_model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Absolute Error (SVR): {mae:.2f}")

# Plot true vs predicted sizes
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, color='blue')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
plt.xlabel('True Particle Size')
plt.ylabel('Predicted Particle Size')
plt.title('SVR: True vs Predicted Sizes (All Features)')
plt.grid()
plt.show()

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_rf = rf_model.predict(X_val)
mae_rf = mean_absolute_error(y_val, y_pred_rf)
print(f"Mean Absolute Error (Random Forest): {mae_rf:.2f}")

# Plot true vs predicted sizes
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_rf, color='green')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
plt.xlabel('True Particle Size')
plt.ylabel('Predicted Particle Size')
plt.title('Random Forest: True vs Predicted Sizes')
plt.grid()
plt.show()

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_gb = gb_model.predict(X_val)
mae_gb = mean_absolute_error(y_val, y_pred_gb)
print(f"Mean Absolute Error (Gradient Boosting): {mae_gb:.2f}")

# Plot true vs predicted sizes
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_gb, color='orange')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
plt.xlabel('True Particle Size')
plt.ylabel('Predicted Particle Size')
plt.title('Gradient Boosting: True vs Predicted Sizes')
plt.grid()
plt.show()

# Train KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_knn = knn_model.predict(X_val)
mae_knn = mean_absolute_error(y_val, y_pred_knn)
print(f"Mean Absolute Error (KNN): {mae_knn:.2f}")

# Plot true vs predicted sizes
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_knn, color='purple')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
plt.xlabel('True Particle Size')
plt.ylabel('Predicted Particle Size')
plt.title('KNN: True vs Predicted Sizes')
plt.grid()
plt.show()

from sklearn.model_selection import GridSearchCV

# Define a parameter grid for SVR
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_svr = grid_search.best_estimator_

# Predictions and evaluation
y_pred_best_svr = best_svr.predict(X_val)
mae_best_svr = mean_absolute_error(y_val, y_pred_best_svr)
print(f"Mean Absolute Error (Best SVR): {mae_best_svr:.2f}")

# Plot true vs predicted sizes
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_best_svr, color='cyan')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
plt.xlabel('True Particle Size')
plt.ylabel('Predicted Particle Size')
plt.title('Best SVR: True vs Predicted Sizes')
plt.grid()
plt.show()
