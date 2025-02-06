import numpy as np
import h5py
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model = load_model('size_estimation_model.h5')

# Load the test image (assuming it's stored in the same format as your training data)
test_file_path = '20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_glass_3200x3000x10um/40MHz_glass_3200x3000x10um_TXT/40MHz_glass_3200x3000x10umVZ000R00000/40MHz_glass_3200x3000x10um.mat'
with h5py.File(test_file_path, 'r') as file:
    RF_test = np.array(file['RFdata']['RF']).transpose()

# Preprocess the test image to match the input shape (similar to your training data)
signal_length = RF_test.shape[0]  # Assuming each test image has the same signal length as training data
test_signals = []  # List to store extracted signals from the test image
coords = []  # To track x, y coordinates for predictions

# Extract signals and their coordinates from the RF data (similar to your training set)
for x_inst in range(RF_test.shape[1]):
    for y_inst in range(RF_test.shape[2]):
        test_signals.append(RF_test[:, x_inst, y_inst])
        coords.append((x_inst, y_inst))

# Convert the test signals into a numpy array and reshape them to match the model's input shape
X_test = np.array(test_signals).reshape((len(test_signals), signal_length, 1))

# Predict the test signals using the loaded model
predictions = model.predict(X_test)

# Initialize an empty array to store the predictions for the heatmap
prediction_map = np.zeros((RF_test.shape[1], RF_test.shape[2]))

# Assign the predicted values to the corresponding pixels (based on the coordinates)
for (x, y), pred in zip(coords, predictions):
    prediction_map[x, y] = pred[0]  # Assuming the prediction is a scalar

# Plot the prediction map as a heatmap with annotations (numbers in each pixel)
plt.figure(figsize=(12, 10))
ax = sns.heatmap(prediction_map, cmap='viridis', cbar=True, annot=True, fmt=".2f", annot_kws={"size": 8}, linewidths=0.5)

# Customize the plot
plt.title('Predicted Diameters as Pixel Intensities with Annotations')
plt.xlabel('X axis')
plt.ylabel('Y axis')

plt.tight_layout()
plt.show()
