'''

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
import matplotlib.pyplot as plt
import re

# Define bins and labels for mapping
bin_ranges = [(20, 28), (38, 57), (67, 90), (300, 330)]
bin_labels = list(range(len(bin_ranges)))

def map_to_bins(size):
    """ Map sizes to their respective bin labels. """
    for i, (low, high) in enumerate(bin_ranges):
        if low <= size <= high:
            return i
    return -1  # For unexpected values

# Load dataset
data = pd.read_csv('train_test.csv')

# Remove entries with 'Material Type' as 'pmma'
data = data[data['Material Type'] == 'pmma']

# Extract signals and size labels
signals = data['Signal'].values
sizes = data['Radius'].values

# Convert signals from strings to numpy arrays
signals_array = [np.fromstring(signal, sep=',') for signal in signals]

# Clean up and convert sizes to integers
sizes_cleaned = [int(re.search(r'\d+', str(size).split()[0]).group()) for size in sizes]

# Map sizes to bins
size_bins = [map_to_bins(size) for size in sizes_cleaned]

# Drop entries with unmapped sizes
valid_indices = [i for i, val in enumerate(size_bins) if val != -1]
signals_array = [signals_array[i] for i in valid_indices]
size_bins = [size_bins[i] for i in valid_indices]

# Pad signals to the same length
max_length = max(len(signal) for signal in signals_array)
signals_array = [np.pad(signal, (0, max_length - len(signal))) for signal in signals_array]

# Use original signal as features
signal_features = np.array(signals_array)

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=50)
fold = 1
all_mse = []
all_accuracy = []

for train_index, test_index in kf.split(signal_features):
    print(f"\nStarting Fold {fold}")
    X_train, X_test = signal_features[train_index], signal_features[test_index]
    y_train, y_test = np.array(size_bins)[train_index], np.array(size_bins)[test_index]

    # Build neural network model
    model = Sequential([
        Input(shape=(max_length,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(bin_labels), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Predict on test data
    y_pred_probs = model.predict(X_test)
    y_pred_bins = np.argmax(y_pred_probs, axis=1)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred_bins)
    accuracy = accuracy_score(y_test, y_pred_bins)
    all_mse.append(mse)
    all_accuracy.append(accuracy)
    print(f"Fold {fold} - Mean Squared Error: {mse}, Accuracy: {accuracy * 100:.2f}%")

    fold += 1

# Report overall performance
print(f"\nOverall Mean Squared Error: {np.mean(all_mse)}")
print(f"Overall Accuracy: {np.mean(all_accuracy) * 100:.2f}%")
print(f"Standard deviation: {np.std(all_accuracy)*100:.2f}%")
