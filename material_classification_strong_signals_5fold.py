'''
Navid.Zarrabi@torontomu.ca
October 25
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

from tensorflow.keras.layers import (
Input, Conv1D, MaxPool1D, Dropout, GlobalMaxPool1D,
Dense, Masking, GlobalAveragePooling1D, GlobalMaxPooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization


# Convert the signals from time-domain to frequency-domain using FFT
def signal_to_frequency(signal):
    fft_result = np.fft.fft(signal)
    magnitude_spectrum = np.abs(fft_result)  # Use magnitude for frequency-domain representation
    return magnitude_spectrum


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
            Conv1D(16, kernel_size=9, activation='relu', padding="valid"),
            BatchNormalization(),
            MaxPooling1D(pool_size=16),
            Dropout(rate=0.1),

            Conv1D(32, kernel_size=3, activation='relu', padding="valid"),
            BatchNormalization(),
            Conv1D(32, kernel_size=3, activation='relu', padding="valid"),
            BatchNormalization(),
            MaxPooling1D(pool_size=4),
            Dropout(rate=0.1),

            Conv1D(32, kernel_size=3, activation='relu', padding="valid"),
            BatchNormalization(),
            Conv1D(32, kernel_size=3, activation='relu', padding="valid"),
            BatchNormalization(),
            MaxPooling1D(pool_size=4),
            Dropout(rate=0.1),

            Conv1D(256, kernel_size=3, activation='relu', padding="valid"),
            BatchNormalization(),
            Conv1D(256, kernel_size=3, activation='relu', padding="valid"),
            BatchNormalization(),
            GlobalMaxPooling1D(),
            Dropout(rate=0.2),

            Dense(64, activation='relu'),
            Dense(1028, activation='relu'),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
    return model

# Load the CSV file
csv_file = 'train_test.csv'
data = pd.read_csv(csv_file)

# Extract signals and material types
signals = data['Signal'].values
material_types = data['Material Type'].values

# Convert signals from strings to numpy arrays
signals_array = [np.fromstring(signal, sep=',') for signal in signals]

# Apply FFT to all signals
signals_freq_array = [signal_to_frequency(signal) for signal in signals_array]

# Define group IDs for each set of 25 signals
groups = np.arange(len(signals_array)) // 25

# Set up GroupKFold cross-validation
gkf = GroupKFold(n_splits=5)
label_encoder = LabelEncoder()

# Collect predictions and true labels from all folds
all_predicted_labels = []
all_true_labels = []

# Store results for each fold
fold_accuracies = []


# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(gkf.split(signals_freq_array, material_types, groups=groups)):
    print(f"\nFold {fold + 1}")

    # Extract training and validation signals and labels based on the indices
    train_signals_array = [signals_freq_array[i] for i in train_idx]
    val_signals_array = [signals_freq_array[i] for i in val_idx]
    train_material_types = [material_types[i] for i in train_idx]
    val_material_types = [material_types[i] for i in val_idx]

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

    # Pad the training and validation signals
    max_length = max(
        max(len(signal) for signal in balanced_train_signals),
        max(len(signal) for signal in val_signals_array)
    )
    padded_train_signals = pad_sequences(balanced_train_signals, maxlen=max_length, dtype='float32', padding='post', truncating='post')
    padded_val_signals = pad_sequences(val_signals_array, maxlen=max_length, dtype='float32', padding='post', truncating='post')
    padded_train_signals = np.expand_dims(padded_train_signals, axis=2)
    padded_val_signals = np.expand_dims(padded_val_signals, axis=2)

    # Encode labels
    train_labels = label_encoder.fit_transform(balanced_train_material_types)
    categorical_train_labels = to_categorical(train_labels)
    val_labels = label_encoder.transform(val_material_types)
    categorical_val_labels = to_categorical(val_labels)

    # Build and compile the model
    model = get_model("medium_model")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(
        padded_train_signals, 
        categorical_train_labels, 
        epochs=200, 
        batch_size=16, 
        validation_data=(padded_val_signals, categorical_val_labels),
        callbacks=[early_stopping],
        verbose=0
    )

    # Generate predictions for the validation set of this fold
    val_predictions = model.predict(padded_val_signals)
    predicted_labels = np.argmax(val_predictions, axis=1)

    # Collect predictions and true labels
    all_predicted_labels.extend(predicted_labels)
    all_true_labels.extend(val_labels)

    # Evaluate model on validation set
    val_loss, val_accuracy = model.evaluate(padded_val_signals, categorical_val_labels, verbose=0)
    fold_accuracies.append(val_accuracy)
    print(f"Fold {fold + 1} Validation Accuracy: {val_accuracy:.4f}")

# Generate and plot the confusion matrix for all folds
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Test set size distribution visualization
test_size_distribution = pd.Series([label_encoder.inverse_transform([label])[0] for label in all_true_labels]).value_counts()
plt.figure(figsize=(8, 6))
test_size_distribution.plot(kind='bar')
plt.xlabel('Material Type')
plt.ylabel('Count')
plt.title('Test Set Size Distribution of Each Material Type')
plt.show()