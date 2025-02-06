'''
Navid.Zarrabi@torontomu.ca
October 25, 2024
Goal: Getting strongest signals from microspheres and their neighbors, and generating labels for them including their location, type, and size
Notes: 
1. Use local maxima with sliding window
2. The code is semi-automatic, it uses some prior info and suggests a material and size, user has to supervise the label using optical images 
'''

import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from scipy.ndimage import gaussian_filter, maximum_filter
import numpy as np
from sklearn.cluster import KMeans

import csv

size_labels=[50]

default_material="ps"
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


# Load data
# Old samples
# PMMA 40MHz 50um 
#directory = '/home/navid/research/high_freq_mp_detection/20231004_PS PMMA_spheres_40-80-200_MHz/PMMA50um_40MHz_2200x2000x12_5um/PMMA50um_40MHz_2200x2000x12_5um_TXT/PMMA50um_40MHz_2200x2000x12_5umVZ000R00000/PMMA50um_40MHz_2200x2000x12_5um.mat'
# # PMMA 40MHz 50um 
#directory = "/home/navid/research/high_freq_mp_detection/20231004_PS PMMA_spheres_40-80-200_MHz/PMMA50um_40MHz_2200x2000x12_5umB/PMMA50um_40MHz_2200x2000x12_5umB_TXT/PMMA50um_40MHz_2200x2000x12_5umBVZ000R00000/PMMA50um_40MHz_2200x2000x12_5umB.mat"
# # PS 40MHz 50um
directory = '/home/navid/research/high_freq_mp_detection/20231004_PS PMMA_spheres_40-80-200_MHz/PS50um_40MHz_1800x1500x12_5um/PS50um_40MHz_1800x1400x12_5um_TXT/PS50um_40MHz_1800x1400x12_5umVZ000R00000/PS50um_40MHz_1800x1400x12_5um.mat'
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
# directory = '20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_glass_3200x3000x10um/40MHz_glass_3200x3000x10um_TXT/40MHz_glass_3200x3000x10umVZ000R00000/40MHz_glass_3200x3000x10um.mat'
# PE 40MHz 1025
#directory = '/home/navid/research/high_freq_size_estimation/20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_ps_3200x3000x10um/40MHz_ps_3200x3000x10um_TXT/40MHz_ps_3200x3000x10umVZ000R00000/40MHz_ps_3200x3000x10um.mat'
# Steel 2 40MHz
#directory = "20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_steel2_3200x3000x10um/40MHz_steel2_3200x3000x10um_TXT/40MHz_steel2_3200x3000x10umVZ000R00000/40MHz_steel2_3200x3000x10um.mat"
# Steel 40MHz
#directory="/home/navid/research/high_freq_size_estimation/20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_steel_1800x1500x10um/40MHz_steel_1800x1500x10um_TXT/40MHz_steel_1800x1500x10umVZ000R00000/40MHz_steel_1800x1500x10um.mat"
# PE 40MHz
#directory="/home/navid/research/high_freq_size_estimation/20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_ps_3200x3000x10um/40MHz_ps_3200x3000x10um_TXT/40MHz_ps_3200x3000x10umVZ000R00000/40MHz_ps_3200x3000x10um.mat"


# Mixed samples
#directory="/home/navid/research/high_freq_size_estimation/20241115_Mixed_microspheres_40_80_MHz/40MHz_mixedspheres_4200x4000x10um_10dB/40MHz_mixedspheres_4200x4000x10um_10dB_TXT/40MHz_mixedspheres_4200x4000x10um_10dBVZ000R00000/40MHz_mixedspheres_4200x4000x10um_10dB.mat"
directory="/home/navid/research/high_freq_size_estimation/20241115_Mixed_microspheres_40_80_MHz/40MHz_mixedspheres_4200x4000x10um_10dB_area2/40MHz_mixedspheres_4200x4000x10um_10dB_area2_TXT/40MHz_mixedspheres_4200x4000x10um_10dB_area2VZ000R00000/40MHz_mixedspheres_4200x4000x10um_10dB_area2.mat"


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
    
    # Sort maxima based on x-coordinate
    sorted_indices = np.argsort(maxima_coords[:, 0])  # Sort by the first column (x-coordinate)
    maxima_coords = maxima_coords[sorted_indices]
    maxima_values = maxima_values[sorted_indices]
    
    return maxima_coords, maxima_values

def smooth_signal(signal, sigma=1):
    """Smooth the signal using a Gaussian filter."""
    smoothed_signal = gaussian_filter(signal, sigma=sigma)
    return smoothed_signal

def fill_valleys_shift_peaks(signal, mean, std_dev):
    """Shift the smaller values upwards by their distance to mean + std_dev."""
    threshold = mean + std_dev
    adjusted_signal = np.where(signal < threshold, signal + np.abs(threshold - signal), signal)
    return adjusted_signal

def train_and_predict_amplitude_classifier(amplitudes, n_classes, new_amplitudes):
    """
    Train a model to classify amplitude values using K-Means clustering, 
    and then predict new amplitude values.

    :param amplitudes: list or np.array of original amplitude values (training data)
    :param n_classes: Number of classes to classify the amplitudes into
    :param new_amplitudes: list or np.array of new amplitude values to classify
    :return: List of predicted class labels for the new_amplitudes
    """
    # Reshape the data as KMeans expects 2D data
    amplitudes = np.array(amplitudes).reshape(-1, 1)
    
    # Step 1: Train the model using KMeans
    kmeans = KMeans(n_clusters=n_classes, random_state=0)
    kmeans.fit(amplitudes)
    
    # Step 2: Predict the class labels for the new amplitudes
    new_amplitudes = np.array(new_amplitudes).reshape(-1, 1)
    predictions = kmeans.predict(new_amplitudes)
    
    return predictions



def calculate_thresholds(amplitudes, n_classes):
    """
    Calculate threshold values based on quantiles, ensuring each bin has approximately 
    the same number of amplitude values.
    
    :param amplitudes: list or np.array of original amplitude values (training data)
    :param n_classes: Number of classes (quantiles) to divide the data into
    :return: List of threshold values
    """
    # Calculate the quantile-based thresholds
    thresholds = np.quantile(amplitudes, np.linspace(0, 1, n_classes + 1))
    
    return thresholds

def predict_amplitude_labels(thresholds, amplitude, guide):
    """
    Assign class labels to new amplitude values based on predefined thresholds (quantiles).
    
    :param thresholds: List of calculated thresholds from calculate_thresholds
    :param new_amplitudes: List of new amplitude values to classify
    :return: List of predicted class labels for new_amplitudes
    """
    labels = []
    # Find the appropriate bin (class) based on thresholds
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= amplitude <= thresholds[i + 1]:
            labels.append(guide[i])  # Assign class label based on the bin
            break

    return labels


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
        #envelope = envelope_detection(x)
        features = extract_features(x)
        if features is not None:
            features['x_index'] = x_inst
            features['y_index'] = y_inst
            all_features.append(features)
            signal_array[:, x_inst, y_inst] += x

# generating an image of max amplitudes
max_amplitude_array = np.max(signal_array, axis=0)

# # 3D Visualization
X = np.arange(max_amplitude_array.shape[0])
Y = np.arange(max_amplitude_array.shape[1])
X, Y = np.meshgrid(X, Y)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, max_amplitude_array.T, cmap='viridis')

ax.set_title('3D Visualization of Aggregated Signals')
ax.set_xlabel('X Index')
ax.set_ylabel('Y Index')
ax.set_zlabel('Signal Amplitude')

plt.tight_layout()
plt.show()

# Calculate mean and standard deviation of the aggregated signals
mean_amplitude = np.mean(max_amplitude_array)
std_dev_amplitude = np.std(max_amplitude_array)

# Apply the new valley filling and peak shifting method
shifted_signals = fill_valleys_shift_peaks(max_amplitude_array, mean_amplitude, std_dev_amplitude)

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


n_size_clusters=len(size_labels)

thresholds = calculate_thresholds(maxima_values, n_size_clusters)


# Initialize list to store particle data
maxima_labels = []


# Loop through the prominent maxima
for coord, value in zip(maxima_coords, maxima_values):
    print(f"Maxima at (X: {coord[0]}, Y: {coord[1]}) -> Amplitude: {value}")
    
    # Plot the heatmap and mark the maxima
    plt.figure(figsize=(6, 6))
    sns.heatmap(max_amplitude_array, cmap='viridis', cbar=True)
    
    # Mark the maxima with a red star
    plt.plot(coord[1], coord[0], 'r*', markersize=15)  # coord[1] is Y, coord[0] is X
    
    plt.title(f"Maxima at (X: {coord[0]}, Y: {coord[1]})")
    plt.xlabel("Y Index")
    plt.ylabel("X Index")
    
    plt.show()
    
    # Query for radius and material type
    # Replace these placeholder functions with the actual logic or arrays
    predictions = predict_amplitude_labels(thresholds, value, size_labels)

    prompt = input(f"Radius is {predictions} or what? ")  # Function or array access to get radius at the coordinate
    
    if prompt=='0':
        pass
    elif prompt=='end':
        break
    else:
        if prompt=="":
            radius=predictions
        else:
            while True:
                try:
                    radius = int(prompt)
                    break  # Exit the loop if input is valid
                except ValueError:
                    print("Invalid input. Please enter an integer. Enter 0 if you want to exclude this particle.")
        while True:
            prompt = input(f"Material is {default_material} or what? ")  # Function or array to get material type at the coordinate
            if prompt in ['pe', 'steel', 'glass', '']:
                break
            else:
                print("Invalid input. Supported inputs are pe, steel, glass")
        
        if prompt=="":
            material_type=default_material
        else:
            material_type=prompt
        # Extract the signal at the maxima's coordinates from signal_array
        #signal_at_maxima = signal_array[:, coord[0], coord[1]]
        
        for i in range(-2,3):
            for j in range(-2,3):
                if (coord[0]+i<=signal_array.shape[1]-1 and coord[1]+j<=signal_array.shape[2]-1):
                    signal_at_maxima=RF[:, coord[0]+i, coord[1]+j]
                    # Store the signal, radius, and material type in particle_info
                    particle_info = [signal_at_maxima, [coord[0]+i, coord[1]+j], radius, material_type]
                    maxima_labels.append(particle_info)
# Optional: Print out collected data for verification
for particle in maxima_labels:
    print(f"\nMaxima detected with Radius: {particle[2]}, Material Type: {particle[3]}, Coordinates: {particle[1]}")
    print(f"  Signal: {particle[0]}")



# Define the filename
csv_filename = 'ps50_40_sample1.csv'

# Write to CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers
    writer.writerow(["Signal", "Radius", "Material Type", "coordinates"])
    
    # Write data
    for particle_info in maxima_labels:
        # Convert the signal array to a string (if it's a numpy array or list)
        signal_as_string = ','.join(map(str, particle_info[0]))
        writer.writerow([signal_as_string, particle_info[2], particle_info[3], particle_info[1]])

print(f"Data successfully saved to {csv_filename}")
