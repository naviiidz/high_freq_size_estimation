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
from tensorflow.keras.models import load_model



# Load the best model
model_save_path = 'best_model.keras'
best_model = load_model(model_save_path)

# Generate predictions
new_predictions = best_model.predict(padded_new_signals)
predicted_labels = np.argmax(new_predictions, axis=1)

# Encode the bin labels for evaluation
label_encoder = LabelEncoder()
label_encoder.fit(new_size_labels)  # Fit on the new size labels to get correct mappings
encoded_true_labels = label_encoder.transform(new_size_labels)

# Evaluate the predictions
from sklearn.metrics import confusion_matrix, classification_report

# Determine unique classes
unique_classes = np.unique(encoded_true_labels)
num_classes = len(unique_classes)

# Create target names based on the unique classes
target_names = [f"{bin_ranges[i][0]}-{bin_ranges[i][1]}" for i in unique_classes]

conf_matrix = confusion_matrix(encoded_true_labels, predicted_labels)
print("Confusion Matrix:\n", conf_matrix)

# Print the classification report with the specified labels
print(classification_report(encoded_true_labels, predicted_labels, target_names=target_names, zero_division=0))

# Optionally, visualize the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Bin')
plt.ylabel('True Bin')
plt.title('Confusion Matrix for New Data')
plt.show()