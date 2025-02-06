'''
Navid.Zarrabi@torontomu.ca
October 25, 2024
Goal: Use strong signals to train and test CNN model for material identification of microspheres
Notes: 
1. 5-fold cross-validation applied
2. Every 25 signals are grouped together to avoid having similar signals in train and test instances
'''

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Masking
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report


from tensorflow.keras.layers import (
Input, Conv1D, MaxPool1D, Dropout, GlobalMaxPool1D,
Dense, Masking, GlobalAveragePooling1D, GlobalMaxPooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization


apply_augmentation=True

# Convert the signals from time-domain to frequency-domain using FFT
def signal_to_frequency(signal):
    fft_result = np.fft.fft(signal)
    magnitude_spectrum = np.abs(fft_result)  # Use magnitude for frequency-domain representation
    return magnitude_spectrum

def add_noise(signal, noise_level=0.05):
    """Add Gaussian noise to a signal."""
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

def time_shift(signal, shift_factor=10):
    """Shift the signal left or right."""
    shift = np.random.randint(-shift_factor, shift_factor)
    return np.roll(signal, shift)

def time_stretch(signal, stretch_factor_range=(0.8, 1.2)):
    """Stretch or compress the signal in time."""
    stretch_factor = np.random.uniform(*stretch_factor_range)
    indices = np.round(np.arange(0, len(signal), stretch_factor)).astype(int)
    indices = indices[indices < len(signal)]
    return signal[indices]

def frequency_mask(signal, max_mask_percentage=0.2):
    """Mask frequencies in the signal."""
    mask_percentage = np.random.uniform(0, max_mask_percentage)
    mask_length = int(len(signal) * mask_percentage)
    start = np.random.randint(0, len(signal) - mask_length)
    signal[start:start + mask_length] = 0
    return signal

def get_model(structure):
    if structure=="1dcnn_uspaper":
        model=Sequential([
        Masking(mask_value=0., input_shape=(max_length, 1)),
        Conv1D(16, 8, activation='relu', kernel_regularizer=l2(0.01)),
        Conv1D(16, 8, activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling1D(4),
        Dropout(0.1),
        Conv1D(64, 4, activation='relu', kernel_regularizer=l2(0.01)),
        Conv1D(64, 4, activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling1D(2),
        Dropout(0.1),
        GlobalAveragePooling1D(),
        Dropout(0.1),
        Dense(len(label_encoder.classes_), activation='softmax')
        ])
    elif structure=="medium_model":
        model = Sequential([
            Conv1D(16, kernel_size=9, activation='relu', padding="valid", input_shape=(max_length, 1)),
            BatchNormalization(),
            Conv1D(32, kernel_size=9, activation='relu', padding="valid"),
            BatchNormalization(),
            MaxPooling1D(pool_size=16),
            Dropout(rate=0.1),

            Conv1D(64, kernel_size=3, activation='relu', padding="valid"),
            BatchNormalization(),
            Conv1D(64, kernel_size=3, activation='relu', padding="valid"),
            BatchNormalization(),
            MaxPooling1D(pool_size=4),
            Dropout(rate=0.1),

            Conv1D(128, kernel_size=3, activation='relu', padding="valid"),
            BatchNormalization(),
            GlobalMaxPooling1D(),
            Dropout(rate=0.2),

            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
    return model


# Load the CSV file
csv_file = 'train_test.csv'
data = pd.read_csv(csv_file)



#removing pe label
#data = data[data['Material Type']!= 'pmma']


# Extract signals and material types
signals = data['Signal'].values
material_types = data['Material Type'].values



# Convert signals from strings to numpy arrays
signals_array = [np.fromstring(signal, sep=',') for signal in signals]

# Apply FFT to all signals
signals_freq_array = [signal_to_frequency(signal) for signal in signals_array]

label_encoder = LabelEncoder()

# Collect predictions and true labels from all folds
all_predicted_labels = []
all_true_labels = []

# Store results for each fold
fold_accuracies = []


# Split the data into training and testing sets
train_signals_array, test_signals_array, train_material_types, test_material_types = train_test_split(
    signals_freq_array, material_types, test_size=0.2, random_state=50, stratify=material_types
)
# Balance the training set
labels = list(set(train_material_types))




label_count = [sum(np.array(train_material_types) == label) for label in labels]
num = min(label_count)

balanced_train_signals = []
balanced_train_material_types = []
for label in labels:
    label_indices = np.where(np.array(train_material_types) == label)[0]
    selected_indices = np.random.choice(label_indices, num, replace=False)
    balanced_train_signals.extend([train_signals_array[i] for i in selected_indices])
    balanced_train_material_types.extend([train_material_types[i] for i in selected_indices])

# Pad the training and testing signals
max_length = max(
    max(len(signal) for signal in balanced_train_signals),
    max(len(signal) for signal in test_signals_array)
)

# Apply augmentations to balanced training signals
if apply_augmentation:
    augmented_train_signals = []
    for signal in balanced_train_signals:
        augmented_signal = signal  # Start with the original signal
        if np.random.rand() < 0.5:
            augmented_signal = add_noise(augmented_signal)
        if np.random.rand() < 0.5:
            augmented_signal = time_shift(augmented_signal)
        if np.random.rand() < 0.5:
            augmented_signal = time_stretch(augmented_signal)
        if np.random.rand() < 0.5:
            augmented_signal = frequency_mask(augmented_signal)
        augmented_train_signals.append(augmented_signal)

    # Combine original and augmented signals
    final_train_signals = balanced_train_signals + augmented_train_signals
    final_train_labels = balanced_train_material_types * 2  # Double labels for original + augmented

else:
    final_train_signals = balanced_train_signals
    final_train_labels = balanced_train_material_types

padded_train_signals = pad_sequences(final_train_signals, maxlen=max_length, dtype='float32', padding='post', truncating='post')
padded_test_signals = pad_sequences(test_signals_array, maxlen=max_length, dtype='float32', padding='post', truncating='post')
padded_train_signals = np.expand_dims(padded_train_signals, axis=2)
padded_test_signals = np.expand_dims(padded_test_signals, axis=2)

# Encode labels
train_labels = label_encoder.fit_transform(final_train_labels)
categorical_train_labels = to_categorical(train_labels)
test_labels = label_encoder.transform(test_material_types)
categorical_test_labels = to_categorical(test_labels)

# Save the classes of the label encoder
np.save('label_encoder_classes.npy', label_encoder.classes_)
print("Label encoder classes saved as 'label_encoder_classes.npy'")

# Build and compile the model
model = get_model("medium_model")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(
    padded_train_signals, 
    categorical_train_labels, 
    epochs=50, 
    batch_size=16, 
    validation_data=(padded_test_signals, categorical_test_labels),
    callbacks=[early_stopping],
    verbose=0
)

# Save the model
model_filename = 'final_model.h5'
model.save(model_filename)
print(f"Final model saved as {model_filename}")

# Generate predictions for the testing set
test_predictions = model.predict(padded_test_signals)
predicted_labels = np.argmax(test_predictions, axis=1)


# Evaluate the model on the testing set
test_loss, test_accuracy = model.evaluate(padded_test_signals, categorical_test_labels, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate and plot the confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Test set size distribution visualization
test_size_distribution = pd.Series([label_encoder.inverse_transform([label])[0] for label in test_labels]).value_counts()
plt.figure(figsize=(8, 6))
test_size_distribution.plot(kind='bar')
plt.xlabel('Material Type')
plt.ylabel('Count')
plt.title('Test Set Size Distribution of Each Material Type')
plt.show()



# Plot training and validation accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()

# Generate the classification report with more floating points
report = classification_report(
    test_labels, 
    predicted_labels, 
    target_names=label_encoder.classes_, 
    digits=6  # Set number of floating-point digits
)

# Print the classification report
print("Classification Report:")
print(report)