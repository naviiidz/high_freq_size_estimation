import pandas as pd

import numpy as np
import tensorflow as tf
import h5py
from scipy.signal import hilbert
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
from roi_detection import fastSam
from scipy.fft import fft
from sklearn.model_selection import train_test_split


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall

def network_structure(signal_length):
    # Define the model
    model = Sequential()

    # C11: Convolutional Layer with 16 filters, 8 kernel size, ReLU activation
    model.add(Conv1D(filters=16, kernel_size=8, activation='relu', input_shape=(signal_length, 1)))  # Define the input shape here
    # Output shape: (None, 996, 16)

    # Continue with the rest of the model configuration as before...
    model.add(Conv1D(filters=16, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=4, activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=512, kernel_size=2, activation='relu'))
    model.add(Conv1D(filters=512, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    
    # Output layer for regression
    model.add(Dense(1, activation='linear'))  # Use 1 unit for regression output

    # Compile the model with Mean Squared Error loss for regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Model summary
    model.summary()
    return model




csv_name='ground_truth_labels.csv'

pd_input = pd.read_csv(csv_name)

def envelope_detection(signal):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope



# Define a function to calculate the Intersection over Union (IoU)
# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    
    intersection_area = inter_width * inter_height
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0

# Function to remove all overlapping bounding boxes
def remove_overlapping_boxes(x_coords, y_coords, widths, heights, iou_threshold=0.0):
    boxes = np.column_stack([x_coords, y_coords, widths, heights])
    keep_boxes = []
    
    # Loop through all boxes and check for overlaps
    for i, box1 in enumerate(boxes):
        has_overlap = False
        for j, box2 in enumerate(boxes):
            if i != j and calculate_iou(box1, box2) > iou_threshold:
                has_overlap = True
                break
        if not has_overlap:
            keep_boxes.append(box1)
    
    keep_boxes = np.array(keep_boxes)
    return keep_boxes[:, 0], keep_boxes[:, 1], keep_boxes[:, 2], keep_boxes[:, 3] if len(keep_boxes) > 0 else ([], [], [], [])

# Non-Maximum Suppression (NMS) to remove overlapping boxes
def non_max_suppression(x_coords, y_coords, widths, heights, iou_threshold=0.3):
    boxes = np.column_stack([x_coords, y_coords, widths, heights])
    keep_boxes = []
    
    while len(boxes) > 0:
        # Always select the first box and remove it from the list
        current_box = boxes[0]
        keep_boxes.append(current_box)
        boxes = boxes[1:]
        
        # Remove boxes that overlap too much with the current box
        boxes = np.array([box for box in boxes if calculate_iou(current_box, box) < iou_threshold])
    
    keep_boxes = np.array(keep_boxes)
    return keep_boxes[:, 0], keep_boxes[:, 1], keep_boxes[:, 2], keep_boxes[:, 3]





# Extract bounding box data
x_coords = pd_input['x']
y_coords = pd_input['y']
widths = pd_input['width']
heights = pd_input['height']
d = pd_input['diameter']
type = pd_input['type']


directory = '20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_glass_3200x3000x10um/40MHz_glass_3200x3000x10um_TXT/40MHz_glass_3200x3000x10umVZ000R00000/40MHz_glass_3200x3000x10um.mat'
with h5py.File(directory, 'r') as file:
    RF = np.array(file['RFdata']['RF']).transpose()
    RFtime = np.array(file['RFdata']['RFtime']).transpose()
    header = np.array(file['RFdata']['header']).transpose()

# Apply NMS to the bounding boxes
x_coords_nms, y_coords_nms, widths_nms, heights_nms = non_max_suppression(x_coords, y_coords, widths, heights, iou_threshold=0.05)


# Compute envelopes for all signals
signal_array = np.zeros(RF.shape)

t = RFtime[0, :]



all_features = []


prev_x=RF[:, 0, 0]
for x_inst in range(RF.shape[1]):
    for y_inst in range(RF.shape[2]):
        x = RF[:, x_inst, y_inst]
        mean_val = np.mean(x)
        if mean_val==0:
            envelope = envelope_detection(prev_x)
        else:
            envelope = envelope_detection(x)
            signal_array[:, x_inst, y_inst] += envelope




aggregated_signals = np.zeros(RF[100].shape)
temp=[]
for instance in signal_array[:]:
    aggregated_signals += instance
    temp.append(instance)
aggregated_signals=np.amax(temp,axis=0)



plt.figure(figsize=(12, 10))
plt.xlim(0, aggregated_signals.shape[0])
plt.ylim(0, aggregated_signals.shape[1])
sns.heatmap(aggregated_signals, cmap='viridis', cbar=True)
plt.title('Particles')
plt.xlabel('X axis')
plt.ylabel('Y axis')

labels_for_signals=[]
collected_signals=[]
# Overlay bounding boxes on the heatmap
for x, y, width, height, diameter in zip(x_coords_nms, y_coords_nms, widths_nms, heights_nms, d):
    rect = plt.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)

    # Collect signals from the signal_array within the bounding box
    for i in range(int(height)):
        # Extract signals in a single column and append them to collected_signals
        for j in range(int(width)):
            collected_signals.append(RF[:,x + j, y + i])  # Append each signal individually
            labels_for_signals.append(diameter)  # Append the corresponding label


plt.tight_layout()
plt.show()


# Example training code
# Ensure collected_signals is a numpy array and reshaped
import numpy as np

# Convert lists to numpy arrays
X = np.array(collected_signals)
y = np.array(labels_for_signals)
# Reshape to (n_samples, signal_length, 1)
X = X.reshape((X.shape[0], X.shape[1], 1))


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=505)
print(y_train)

# Create and train the model
signal_length = X.shape[1]
model = network_structure(signal_length)

# Train the model (adjust epochs and batch_size as needed)
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
model.save('size_estimation_model.h5')




model = network_structure(signal_length)

history = model.fit(
    X_train,            # Training data
    y_train,            # Training labels
    epochs=50,          # Number of epochs
    batch_size=32,      # Batch size
    validation_data=(X_val, y_val),  # Validation data
    shuffle=True        # Shuffle training data at the beginning of each epoch
)


plt.figure()
# Plot training & validation Mean Absolute Error values
plt.subplot(1,2,1)
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
if 'val_mean_absolute_error' in history.history:
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()

# Plot training & validation loss values
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
