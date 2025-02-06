import numpy as np
import h5py
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert, butter, lfilter
from scipy.fft import fft
from scipy.stats import skew, kurtosis

# Function definitions remain the same...

def envelope_detection(signal):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

def extract_features(signal):
    features = {}
    mean_val = np.mean(signal)
    if mean_val == 0:
        features['mean'] = 0
        features['var'] = 0
        features['skew'] = 0
        features['kurtosis'] = 0
        features['max_power'] = 0
        features['mean_power'] = 0
        features['max_amplitude'] = 0
        features['min_amplitude'] = 0
        features['peak_to_peak'] = 0
    else:
        features['mean'] = mean_val
        features['var'] = np.var(signal)
        features['skew'] = skew(signal)
        features['kurtosis'] = kurtosis(signal)
        power_spectrum = np.abs(fft(signal))**2
        features['max_power'] = np.max(power_spectrum)
        features['mean_power'] = np.mean(power_spectrum)
        features['max_amplitude'] = np.max(signal)
        features['min_amplitude'] = np.min(signal)
        features['peak_to_peak'] = np.ptp(signal)
    return features

# Load the data
directory = '20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_glass_3200x3000x10um/40MHz_glass_3200x3000x10um_TXT/40MHz_glass_3200x3000x10umVZ000R00000/40MHz_glass_3200x3000x10um.mat'
with h5py.File(directory, 'r') as file:
    RF = np.array(file['RFdata']['RF']).transpose()
    RFtime = np.array(file['RFdata']['RFtime']).transpose()

# Initialize arrays and variables
signal_array = np.zeros(RF.shape)
all_features = []
signal_samples = []  # List to store the signals of interest

# Loop over RF data
for x_inst in range(RF.shape[1]):
    for y_inst in range(RF.shape[2]):
        x = RF[:, x_inst, y_inst]
        envelope = envelope_detection(x)
        features = extract_features(envelope)
        if features is not None:
            features['x_index'] = x_inst
            features['y_index'] = y_inst
            all_features.append(features)
            signal_array[:, x_inst, y_inst] += envelope
            
            # Save the signal
            signal_samples.append({
                'x_index': x_inst,
                'y_index': y_inst,
                'signal': list(envelope)  # Save the envelope signal as a list
            })

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
plt.tight_layout()

# Convert to a DataFrame for easier handling
import pandas as pd

features_df = pd.DataFrame(all_features)

# Plotting features as heatmaps using subplots
feature_names = ['var', 'skew', 'kurtosis', 'max_power', 'mean_power', 'max_amplitude', 'min_amplitude', 'peak_to_peak']
titles = ['Variance', 'Skew', 'Kurtosis', 'Max Power', 'Mean Power', 'Max Amplitude', 'Min Amplitude', 'Peak to Peak']

num_features = len(feature_names)

# Create subplots for each feature
fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Adjust the number of rows and columns as needed
axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

for i, feature in enumerate(feature_names):
    # Reshape the feature data to match the grid of particles
    feature_data = features_df[feature].values.reshape(RF.shape[1], RF.shape[2])
    
    # Plotting heatmap for each feature
    sns.heatmap(feature_data, ax=axes[i], cmap='viridis', cbar=True)
    axes[i].set_title(titles[i])

    axes[i].set_xlabel('X axis')
    axes[i].set_ylabel('Y axis')
    # Turn off the numbers on the x and y axes for each subplot
    axes[i].set_xticks([])  # Turn off x-axis ticks
    axes[i].set_yticks([])  # Turn off y-axis ticks
plt.tight_layout()
plt.show()








# Normalize aggregated_envelopes to the range [0, 255]
aggregated_envelopes_norm = cv2.normalize(aggregated_signals, None, 0, 255, cv2.NORM_MINMAX)

# Convert to uint8
aggregated_signals_uint8 = np.uint8(aggregated_envelopes_norm)

# Display the normalized image
cv2.imshow("Original Image", aggregated_signals_uint8)


# Normalize the image to the range [0, 255]
aggregated_signals_norm = cv2.normalize(aggregated_signals, None, 0, 255, cv2.NORM_MINMAX)
aggregated_signals_uint8 = np.uint8(aggregated_signals_norm)

# Thresholding with lower and higher thresholds
_, lower_threshold_image = cv2.threshold(aggregated_signals_uint8, 50, 100, cv2.THRESH_BINARY)  # Lower threshold
_, higher_threshold_image = cv2.threshold(aggregated_signals_uint8, 100, 255, cv2.THRESH_BINARY)  # Higher threshold

# Find contours for lower threshold image (small boxes)
contours_small, _ = cv2.findContours(lower_threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter small contours: Remove contours with bounding boxes smaller than 10x10
filtered_small_contours = []
for contour in contours_small:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 15 and h <= 15:  # Keep contours that are at least 10x10
        filtered_small_contours.append(contour)
    
# Find contours for higher threshold image (large boxes)
contours_large, _ = cv2.findContours(higher_threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter small contours: Remove contours with bounding boxes smaller than 10x10
filtered_contours_large = []
for contour in contours_large:
    x, y, w, h = cv2.boundingRect(contour)
    if w >= 5 and h >= 5:  # Keep contours that are at least 10x10
        filtered_contours_large.append(contour)

# Merge the small and large contours
merged_contours = filtered_small_contours+filtered_contours_large

# Draw bounding boxes for merged contours on a copy of the original image
image_with_merged_boxes = cv2.cvtColor(aggregated_signals_uint8, cv2.COLOR_GRAY2BGR)  # Convert to color

# Draw bounding boxes around the merged contours
for contour in merged_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_with_merged_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for merged boxes

# Display the image with merged bounding boxes
cv2.imshow("Image with Merged Bounding Boxes", image_with_merged_boxes)

# Save the resulting image
cv2.imwrite('output_image_with_merged_boxes.png', image_with_merged_boxes)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Initialize a list to store labels (diameters)
labels = []

# Draw and display each contour
for contour in merged_contours:
    # Create a copy of the original image for drawing
    image_with_single_box = cv2.cvtColor(aggregated_signals_uint8, cv2.COLOR_GRAY2BGR)  # Convert to color
    x, y, w, h = cv2.boundingRect(contour)

    # Draw the bounding box for the current contour
    cv2.rectangle(image_with_single_box, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for bounding box

    # Display the image with the current bounding box
    cv2.imshow("Image with Single Bounding Box", image_with_single_box)
    cv2.waitKey(0)  # Display each contour for 1 second
    cv2.destroyAllWindows()

    # Prompt the user for the ground truth diameter
    diameter = input(f"Enter the ground truth diameter for the contour at ({x}, {y}): ")
    if float(diameter) != 0:
        labels.append({
            'diameter': float(diameter),
            'type': 'glass',
            'x': x,
            'y': y,
            'width': w,
            'height': h
        })

# Convert the list of signals into a DataFrame
signal_df = pd.DataFrame(signal_samples)

# Save the DataFrame to a CSV file
signal_df.to_csv('extracted_signals.csv', index=False)

print("Signals have been saved to 'extracted_signals.csv'.")

# Close all windows after displaying all contours
cv2.destroyAllWindows()

# Print the labels for verification
print("Ground Truth Labels (Diameters):", labels)

# Convert the list of labels into a pandas DataFrame
labels_df = pd.DataFrame(labels)

# Save the DataFrame to a CSV file
labels_df.to_csv('ground_truth_labels.csv', index=False)

print("Labels have been saved to 'ground_truth_labels.csv'.")
