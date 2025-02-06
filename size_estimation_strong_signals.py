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

csv_file = 'pe_signals.csv'
material="pmma"
# Define size bins and labels
bin_ranges = [(15, 25), (40, 50), (65, 75), (290, 310)]
bin_labels = list(range(len(bin_ranges)))

# Function to assign a bin label based on size
def get_bin_label(size):
    for i, (lower, upper) in enumerate(bin_ranges):
        if '[' in size:
            if lower <= int(size[1:-1]) <= upper:
                return i 
        else:
            if lower <= int(size) <= upper:
                return i         
    return 0

# Convert signals from time-domain to frequency-domain using FFT
def signal_to_frequency(signal):
    fft_result = np.fft.fft(signal)
    magnitude_spectrum = np.abs(fft_result)
    return magnitude_spectrum

# Load CSV data
data = pd.read_csv(csv_file)

# Extract signals and sizes
signals = data['Signal'].values
sizes = data['Radius'].values

# Convert signals from strings to numpy arrays
signals_array = [np.fromstring(signal, sep=',') for signal in signals]

# Assign bin labels to each size
size_labels = [get_bin_label(size) for size in sizes]

# Apply FFT to all signals
signals_freq_array = [signal_to_frequency(signal) for signal in signals_array]

# Group signals into sets of 25 for consistent fold splits
groups = np.arange(len(signals_array)) // 25

# Set up GroupKFold cross-validation
gkf = GroupKFold(n_splits=5)

# Initialize LabelEncoder before the loop 
label_encoder = LabelEncoder()
label_encoder.fit(size_labels)  # Fit on all size labels to ensure it knows all classes

# Collect all predictions and true labels across folds
all_predicted_labels = []
all_true_labels = []
fold_accuracies = []

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(gkf.split(signals_freq_array, size_labels, groups=groups)):
    print(f"\nFold {fold + 1}")

    # Split train/validation sets
    train_signals = [signals_freq_array[i] for i in train_idx]
    val_signals = [signals_freq_array[i] for i in val_idx]
    train_sizes = [size_labels[i] for i in train_idx]
    val_sizes = [size_labels[i] for i in val_idx]

    # Balance the training data
    unique_labels = list(set(train_sizes))
    label_counts = [train_sizes.count(label) for label in unique_labels]
    min_count = min(label_counts)
    
    balanced_train_signals = []
    balanced_train_sizes = []
    for label in unique_labels:
        label_indices = [i for i, l in enumerate(train_sizes) if l == label]
        selected_indices = np.random.choice(label_indices, min_count, replace=False)
        balanced_train_signals.extend([train_signals[i] for i in selected_indices])
        balanced_train_sizes.extend([train_sizes[i] for i in selected_indices])

    # Pad signals for uniform length
    max_length = max(
        max(len(signal) for signal in balanced_train_signals),
        max(len(signal) for signal in val_signals)
    )
    padded_train_signals = pad_sequences(balanced_train_signals, maxlen=max_length, dtype='float32', padding='post', truncating='post')
    padded_val_signals = pad_sequences(val_signals, maxlen=max_length, dtype='float32', padding='post', truncating='post')
    padded_train_signals = np.expand_dims(padded_train_signals, axis=2)
    padded_val_signals = np.expand_dims(padded_val_signals, axis=2)

    # Encode bin labels 
    train_labels = label_encoder.transform(balanced_train_sizes) 
    categorical_train_labels = to_categorical(train_labels, num_classes=len(bin_labels)) # Specify num_classes
    val_labels = label_encoder.transform(val_sizes)
    categorical_val_labels = to_categorical(val_labels, num_classes=len(bin_labels))  # Specify num_classes

    # Build and compile the model
    model = Sequential([
        Masking(mask_value=0., input_shape=(max_length, 1)),
        Conv1D(16, 8, activation='relu', kernel_regularizer=l2(0.01)),
        Conv1D(16, 8, activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling1D(2),
        Dropout(0.4),
        Conv1D(64, 4, activation='relu', kernel_regularizer=l2(0.01)),
        Conv1D(64, 4, activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling1D(2),
        Dropout(0.4),
        GlobalAveragePooling1D(),
        Dropout(0.3),
        Dense(len(bin_labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Define the path where the model will be saved
    model_save_path = material+"_size_model"+'.keras'

    model_checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # Train the model
    model.fit(
        padded_train_signals, 
        categorical_train_labels, 
        epochs=200, 
        batch_size=16, 
        validation_data=(padded_val_signals, categorical_val_labels),
        callbacks=[early_stopping, model_checkpoint],  # Add model_checkpoint here
        verbose=0
    )

    # Generate predictions
    val_predictions = model.predict(padded_val_signals)
    predicted_labels = np.argmax(val_predictions, axis=1)

    # Collect predictions and true labels
    all_predicted_labels.extend(predicted_labels)
    all_true_labels.extend(val_labels)

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(padded_val_signals, categorical_val_labels, verbose=0)
    fold_accuracies.append(val_accuracy)
    print(f"Fold {fold + 1} Validation Accuracy: {val_accuracy:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f"{low}-{high}" for low, high in bin_ranges], 
            yticklabels=[f"{low}-{high}" for low, high in bin_ranges])
plt.xlabel('Predicted Bin')
plt.ylabel('True Bin')
plt.title('Confusion Matrix for Size Bins')
plt.show()

# Print overall classification report
print(classification_report(all_true_labels, all_predicted_labels, target_names=[f"{low}-{high}" for low, high in bin_ranges]))

# Display accuracy across folds
print(f"Mean Cross-Validation Accuracy: {np.mean(fold_accuracies):.4f}")