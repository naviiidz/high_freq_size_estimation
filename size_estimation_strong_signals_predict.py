import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, Masking
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint



import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder




import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from mpl_toolkits.mplot3d import Axes3D



# Convert the signals from time-domain to frequency-domain using FFT
def signal_to_frequency(signal):
    fft_result = np.fft.fft(signal)
    magnitude_spectrum = np.abs(fft_result)  # Use magnitude for frequency-domain representation
    return magnitude_spectrum



size_labels=[20, 45, 70, 300]

default_material="pe"
x_drop_pixels=0
y_drop_pixels=0


# Set global font size
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

def envelope_detection(signal):
    """Compute the envelope of a signal using the Hilbert transform."""
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

def find_prominent_peaks(signal, window_size=3, threshold=0):
    """Find local peaks (mountain-like maxima) in a 2D signal array using a sliding window approach.
    The central point must be greater than all surrounding points by a certain threshold.
    """
    peaks_indices = []
    
    # Iterate over the signal dimensions, avoiding edges
    for x in range(window_size//2, signal.shape[0] - window_size//2):
        for y in range(window_size//2, signal.shape[1] - window_size//2):
            
            # Define the sliding window region
            window = signal[x - window_size//2 : x + window_size//2 + 1, 
                            y - window_size//2 : y + window_size//2 + 1]
            
            # Extract the central value and compare it with its neighbors
            central_value = signal[x, y]
            surrounding_values = np.delete(window.flatten(), len(window.flatten())//2)
            
            # Check if the central value is a peak (greater than surrounding values by threshold)
            if np.all(central_value > surrounding_values + threshold):
                peaks_indices.append((x, y))
    
    return peaks_indices


def extract_features(signal):
    """Extract various statistical features from a signal."""
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
        fft_vals = fft(signal)
        power_spectrum = np.abs(fft_vals)**2
        features['max_power'] = np.max(power_spectrum)
        features['mean_power'] = np.mean(power_spectrum)
        features['max_amplitude'] = np.max(signal)
        features['min_amplitude'] = np.min(signal)
        features['peak_to_peak'] = np.ptp(signal)
    return features

# Load data
# Old samples
# PMMA 40MHz 50um 
#directory = '/home/navid/research/high_freq_mp_detection/20231004_PS PMMA_spheres_40-80-200_MHz/PMMA50um_40MHz_2200x2000x12_5um/PMMA50um_40MHz_2200x2000x12_5um_TXT/PMMA50um_40MHz_2200x2000x12_5umVZ000R00000/PMMA50um_40MHz_2200x2000x12_5um.mat'
# # PMMA 40MHz 50um 
directory = "/home/navid/research/high_freq_mp_detection/20231004_PS PMMA_spheres_40-80-200_MHz/PMMA50um_40MHz_2200x2000x12_5umB/PMMA50um_40MHz_2200x2000x12_5umB_TXT/PMMA50um_40MHz_2200x2000x12_5umBVZ000R00000/PMMA50um_40MHz_2200x2000x12_5umB.mat"
# # PS 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20231004_PS PMMA_spheres_40-80-200_MHz/PS50um_40MHz_1800x1500x12_5um/PS50um_40MHz_1800x1400x12_5um_TXT/PS50um_40MHz_1800x1400x12_5umVZ000R00000/PS50um_40MHz_1800x1400x12_5um.mat'
# # PS 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20231004_PS PMMA_spheres_40-80-200_MHz/PS50um_40MHz_2200x2000x12_5um/PS50um_40MHz_2200x2000x12_5um_TXT/PS50um_40MHz_2200x2000x12_5umVZ000R00000/PS50um_40MHz_2200x2000x12_5um.mat'
# # Glass 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_glass_300x300x10um/40MHz_glass_300x300x10um_TXT/40MHz_glass_300x300x10umVZ000R00000/40MHz_glass_300x300x10um.mat'
# # Glass 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_glass_1000x1000x20um/40MHz_glass_1000x1000x20um_TXT/40MHz_glass_1000x1000x20umVZ000R00000/40MHz_glass_1000x1000x20um.mat'
# # PE 40MHz 50um single microsphere
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_polye_300x300x10um/40MHz_polye_300x300x10um_TXT/40MHz_polye_300x300x10umVZ000R00000/40MHz_polye_300x300x10um.mat'
# # PE 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_polye_1200x1200x20um/40MHz_polye_1200x1200x20um_TXT/40MHz_polye_1200x1200x20umVZ000R00000/40MHz_polye_1200x1200x20um.mat'
# # Steel 40MHz 50um single particle
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_steel_300x300x10um/40MHz_steel_300x300x10um_TXT/40MHz_steel_300x300x10umVZ000R00000/40MHz_steel_300x300x10um.mat'
# # Steel 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_steel_1000x1000x20um/40MHz_steel_1000x1000x20um_TXT/40MHz_steel_1000x1000x20umVZ000R00000/40MHz_steel_1000x1000x20um.mat'


# New samples
# Glass 40MHz 1025
#directory = '20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_glass_3200x3000x10um/40MHz_glass_3200x3000x10um_TXT/40MHz_glass_3200x3000x10umVZ000R00000/40MHz_glass_3200x3000x10um.mat'
# PE 40MHz 1025
#directory = '/home/navid/research/high_freq_size_estimation/20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_ps_3200x3000x10um/40MHz_ps_3200x3000x10um_TXT/40MHz_ps_3200x3000x10umVZ000R00000/40MHz_ps_3200x3000x10um.mat'
# Steel 2 40MHz
#directory = "20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_steel2_3200x3000x10um/40MHz_steel2_3200x3000x10um_TXT/40MHz_steel2_3200x3000x10umVZ000R00000/40MHz_steel2_3200x3000x10um.mat"
# Steel 40MHz
#directory="/home/navid/research/high_freq_size_estimation/20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_steel_1800x1500x10um/40MHz_steel_1800x1500x10um_TXT/40MHz_steel_1800x1500x10umVZ000R00000/40MHz_steel_1800x1500x10um.mat"
# PE 40MHz
#directory="/home/navid/research/high_freq_size_estimation/20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_ps_3200x3000x10um/40MHz_ps_3200x3000x10um_TXT/40MHz_ps_3200x3000x10umVZ000R00000/40MHz_ps_3200x3000x10um.mat"

# Mixed samples
# directory="/home/navid/research/high_freq_size_estimation/20241115_Mixed_microspheres_40_80_MHz/40MHz_mixedspheres_4200x4000x10um_10dB/40MHz_mixedspheres_4200x4000x10um_10dB_TXT/40MHz_mixedspheres_4200x4000x10um_10dBVZ000R00000/40MHz_mixedspheres_4200x4000x10um_10dB.mat"
# directory="/home/navid/research/high_freq_size_estimation/20241115_Mixed_microspheres_40_80_MHz/40MHz_mixedspheres_4200x4000x10um_10dB_area2/40MHz_mixedspheres_4200x4000x10um_10dB_area2_TXT/40MHz_mixedspheres_4200x4000x10um_10dB_area2VZ000R00000/40MHz_mixedspheres_4200x4000x10um_10dB_area2.mat"



with h5py.File(directory, 'r') as file:
    RF = np.array(file['RFdata']['RF']).transpose()
    RFtime = np.array(file['RFdata']['RFtime']).transpose()

# Compute envelopes for all signals
signal_array_size=(RF.shape[0], RF.shape[1]-x_drop_pixels, RF.shape[2]-y_drop_pixels)
signal_array = np.zeros(signal_array_size)
all_features = []

for x_inst in range(signal_array_size[1]):
    for y_inst in range(signal_array_size[2]):
        x = RF[:, x_inst, y_inst]
        envelope = envelope_detection(x)
        features = extract_features(envelope)
        if features is not None:
            features['x_index'] = x_inst
            features['y_index'] = y_inst
            all_features.append(features)
            signal_array[:, x_inst, y_inst] += envelope

# Aggregate signals
aggregated_signals = np.max(signal_array, axis=0)

# # 3D Visualization
X = np.arange(aggregated_signals.shape[0])
Y = np.arange(aggregated_signals.shape[1])
X, Y = np.meshgrid(X, Y)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, aggregated_signals.T, cmap='viridis')

ax.set_title('3D Visualization of Aggregated Signals')
ax.set_xlabel('X Index')
ax.set_ylabel('Y Index')
ax.set_zlabel('Signal Amplitude')

plt.tight_layout()
plt.show()

from scipy.ndimage import gaussian_filter, maximum_filter
import numpy as np



def smooth_signal(signal, sigma=1):
    """Smooth the signal using a Gaussian filter."""
    smoothed_signal = gaussian_filter(signal, sigma=sigma)
    return smoothed_signal
def fill_valleys_shift_peaks(signal, mean, std_dev):
    """Shift the smaller values upwards by their distance to mean + std_dev."""
    threshold = mean + std_dev
    adjusted_signal = np.where(signal < threshold, signal + np.abs(threshold - signal), signal)
    return adjusted_signal

# Calculate mean and standard deviation of the aggregated signals
mean_amplitude = np.mean(aggregated_signals)
std_dev_amplitude = np.std(aggregated_signals)

# Apply the new valley filling and peak shifting method
shifted_signals = fill_valleys_shift_peaks(aggregated_signals, mean_amplitude, std_dev_amplitude)

# Optional: Apply smoothing after shifting
sigma_value = 3  # Adjust sigma for more smoothing (higher value = smoother)
smoothed_shifted_signals = smooth_signal(shifted_signals, sigma=sigma_value)

# 3D Visualization of Shifted Signals
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, smoothed_shifted_signals.T, cmap='viridis')

ax.set_title('3D Visualization of Shifted Signals (Peaks Shifted Above)')
ax.set_xlabel('X Index')
ax.set_ylabel('Y Index')
ax.set_zlabel('Signal Amplitude')

plt.tight_layout()
plt.show()

# Optional: Print out mean and standard deviation for verification
print(f"Mean Amplitude: {mean_amplitude}, Standard Deviation: {std_dev_amplitude}")





def find_prominent_maxima(signal, window_size=5, prominence_threshold=0.1):
    """Find prominent maxima in the smoothed signal using a sliding window and prominence threshold."""
    # Apply a maximum filter to find local maxima
    max_filter = maximum_filter(signal, size=window_size)
    local_maxima = (signal == max_filter)
    
    # Apply prominence threshold to filter small maxima
    prominent_maxima = np.where(signal > prominence_threshold, local_maxima, False)
    
    # Get the coordinates and values of prominent maxima
    maxima_coords = np.argwhere(prominent_maxima)
    maxima_values = signal[prominent_maxima]
    
    return maxima_coords, maxima_values

# Apply the Gaussian filter to smooth the signal
sigma_value = 3  # Adjust sigma for more smoothing (higher value = smoother)
smoothed_signals = smooth_signal(shifted_signals, sigma=sigma_value)

# Find prominent maxima in the smoothed signal
window_size = 3  # Increase window size to avoid small local maxima
prominence_threshold = np.mean(shifted_signals) + np.std(shifted_signals)  # Set prominence based on signal statistics
maxima_coords, maxima_values = find_prominent_maxima(shifted_signals, window_size=window_size, prominence_threshold=prominence_threshold)

# 3D Visualization of Smoothed Signals with Prominent Maxima
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, shifted_signals.T, cmap='viridis')

# Plot maxima points on the surface
ax.scatter(maxima_coords[:, 0], maxima_coords[:, 1], shifted_signals[maxima_coords[:, 0], maxima_coords[:, 1]], color='red', s=50, label='Prominent Maxima')

ax.set_title(f'3D Visualization of Smoothed Signals with Prominent Maxima (Sigma = {sigma_value})')
ax.set_xlabel('X Index')
ax.set_ylabel('Y Index')
ax.set_zlabel('Signal Amplitude')
ax.legend()

plt.tight_layout()
plt.show()






import numpy as np





# Initialize list to store particle data
maxima_labels = []


# Loop through the prominent maxima
for coord, value in zip(maxima_coords, maxima_values):
    print(f"Maxima at (X: {coord[0]}, Y: {coord[1]}) -> Amplitude: {value}")
    
    # Plot the heatmap and mark the maxima
    plt.figure(figsize=(6, 6))
    sns.heatmap(aggregated_signals, cmap='viridis', cbar=True)
    
    # Mark the maxima with a red star
    plt.plot(coord[1], coord[0], 'r*', markersize=15)  # coord[1] is Y, coord[0] is X
    
    plt.title(f"Maxima at (X: {coord[0]}, Y: {coord[1]})")
    plt.xlabel("Y Index")
    plt.ylabel("X Index")
    
    plt.show()

    signal_at_maxima=RF[:, coord[0], coord[1]]
    # Store the signal, radius, and material type in particle_info
    particle_info = [signal_at_maxima, [coord[0], coord[1]]]
    maxima_labels.append(particle_info)



# for loop to irerate through each signal and predict material
    
# based on material, predict size and load the material's corresponding model for size estimation
    
# visualize the results on the plot with a bounding box and text



















# # Load the new CSV data
# material="glass"
# # Define size bins and labels
# bin_ranges = [(38, 53), (75, 90), (300, 335)]
# bin_labels = list(range(len(bin_ranges)))

# model_save_path = material+"_size_model"+'.keras'
# print(os.path.exists(model_save_path))  # Should return True if the file exists
# new_csv_file = 'ABChomeABCnavidABCresearchABChigh_freq_mp_detectionABC20240417_50um steel glass plastic spheres 40-80 MHzABC40MHz_glass_1000x1000x20umABC40MHz_glass_1000x1000x20um_TXTABC40MHz_glass_1000x1000x20umVZ000R00000ABC40MHz_glass_1000x1000x20um.csv'
# new_data = pd.read_csv(new_csv_file)

# # Extract signals and sizes
# new_signals = new_data['Signal'].values
# new_sizes = new_data['Radius'].values

# # Convert signals from strings to numpy arrays
# new_signals_array = [np.fromstring(signal, sep=',') for signal in new_signals]

# # Assign bin labels to each size (use the same function defined earlier)
# def get_bin_label(size):
#     for i, (lower, upper) in enumerate(bin_ranges):
#         if lower <= int(size[1:-1]) <= upper:
#             return i
#     return 0

# # Assign bin labels to each size
# new_size_labels = [get_bin_label(size) for size in new_sizes]

# # Apply FFT to all new signals
# new_signals_freq_array = [signal_to_frequency(signal) for signal in new_signals_array]

# # Pad new signals for uniform length (using the max_length from training)
# max_length = max(len(signal) for signal in new_signals_freq_array)
# padded_new_signals = pad_sequences(new_signals_freq_array, maxlen=max_length, dtype='float32', padding='post', truncating='post')
# padded_new_signals = np.expand_dims(padded_new_signals, axis=2)

# # Load the best model
# best_model = load_model(model_save_path)

# # Generate predictions
# new_predictions = best_model.predict(padded_new_signals)
# predicted_labels = np.argmax(new_predictions, axis=1)

# # Encode the bin labels for evaluation
# label_encoder = LabelEncoder()
# label_encoder.fit(new_size_labels)  # Fit on the new size labels to get correct mappings
# encoded_true_labels = label_encoder.transform(new_size_labels)

# # Evaluate the predictions
# from sklearn.metrics import confusion_matrix, classification_report

# # Determine unique classes
# unique_classes = np.unique(encoded_true_labels)
# num_classes = len(unique_classes)

# # Create target names based on the unique classes
# target_names = [f"{bin_ranges[i][0]}-{bin_ranges[i][1]}" for i in unique_classes]

# conf_matrix = confusion_matrix(encoded_true_labels, predicted_labels)
# print("Confusion Matrix:\n", conf_matrix)

# # Print the classification report with the specified labels
# print(classification_report(encoded_true_labels, predicted_labels, target_names=target_names, zero_division=0))

# # Optionally, visualize the confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=target_names, yticklabels=target_names)
# plt.xlabel('Predicted Bin')
# plt.ylabel('True Bin')
# plt.title('Confusion Matrix for New Data')
# plt.show()