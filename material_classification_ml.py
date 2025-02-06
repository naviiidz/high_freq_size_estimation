import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Random seed
rand_state = np.random.randint(0,1000)

def calculate_feature_correlation(features):
    """
    Computes the correlation matrix of the given feature set.

    Parameters:
    - features: np.ndarray
        The dataset containing the features (after preprocessing).

    Returns:
    - corr_matrix: np.ndarray
        The computed correlation matrix.
    """
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(features, rowvar=False)

    # Display the correlation matrix
    print("Feature Correlation Matrix:")
    print(corr_matrix)

    return corr_matrix

def show_feature_correlation(data, features, target_column=None, method='pearson'):
    """
    Computes and visualizes the correlation matrix of features.

    Parameters:
    - data: pd.DataFrame
        The dataset containing features and target.
    - features: list of str
        List of feature column names to include in the correlation matrix.
    - target_column: str, optional
        Name of the target column to include in the correlation analysis (default: None).
    - method: str
        Correlation method to use: 'pearson', 'spearman', or 'kendall' (default: 'pearson').

    Returns:
    - corr_matrix: pd.DataFrame
        The computed correlation matrix.
    """
    if target_column:
        # Include the target column in the correlation analysis
        features = features + [target_column]

    # Compute correlation matrix
    corr_matrix = data[features].corr(method=method)

    # Plot the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title(f'Feature Correlation ({method.title()} Method)')
    plt.show()

    return corr_matrix


# Function to convert signals to the frequency domain
def signal_to_frequency(signal):
    fft_result = np.fft.fft(signal)
    magnitude_spectrum = np.abs(fft_result)
    return magnitude_spectrum

# Load dataset
data = pd.read_csv('train_test.csv')

# Remove entries with 'Material Type' as 'pmma'
data = data[data['Material Type'] != 'pmma']

# Extract signals and material types
signals = data['Signal'].values
material_types = data['Material Type'].values

# Convert signals from strings to numpy arrays
signals_array = [np.fromstring(signal, sep=',') for signal in signals]

# Apply FFT to all signals
signals_freq_array = [signal_to_frequency(signal) for signal in signals_array]

# Encode material types into numeric labels
label_encoder = LabelEncoder()
encoded_materials = label_encoder.fit_transform(material_types)

# Convert FFT-transformed signals into a feature set
features = np.array([[np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
                      kurtosis(signal), skew(signal)] 
                     for signal in signals_freq_array])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, encoded_materials, test_size=0.3, random_state=rand_state)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate and display the correlation of the features
feature_correlation_matrix = calculate_feature_correlation(X_train_scaled)


# Define and initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=rand_state),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(random_state=rand_state),
    'Gradient Boosting': GradientBoostingClassifier(random_state=rand_state),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 100), random_state=rand_state, max_iter=100),
    'Decision Tree': DecisionTreeClassifier(random_state=rand_state),
    'SVM (RBF Kernel)': SVC(kernel='rbf', random_state=rand_state),
    'PCA + Logistic Regression': None,  # Handled separately
    'Voting Classifier': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=rand_state)),
        ('lr', LogisticRegression(random_state=rand_state)),
        ('gbm', GradientBoostingClassifier(random_state=rand_state)),
    ], voting='soft')
}

#
# Evaluate models and calculate per-class accuracy
results = {name: [] for name in models.keys()}
for name, model in models.items():
    if name == 'PCA + Logistic Regression':
        # PCA + Logistic Regression
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        model = LogisticRegression(random_state=rand_state)
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    # Calculate overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    
    # Confusion matrix for per-class accuracy
    cm = confusion_matrix(y_test, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # Save results
    results[name].append((accuracy, report, class_accuracies))

# Display results with per-class accuracy
for model_name, scores in results.items():
    accuracies = [score[0] for score in scores]
    reports = [score[1] for score in scores]
    class_accuracies_list = [score[2] for score in scores]

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    precision_list = [report['weighted avg']['precision'] for report in reports]
    recall_list = [report['weighted avg']['recall'] for report in reports]
    f1_list = [report['weighted avg']['f1-score'] for report in reports]

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)

    print(f"\n{model_name}:")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    # Per-class accuracy
    avg_class_accuracies = np.mean(class_accuracies_list, axis=0)
    print("Class-wise Accuracy:")
    for idx, acc in enumerate(avg_class_accuracies):
        class_name = label_encoder.inverse_transform([idx])[0]
        print(f"  {class_name}: {acc:.4f}")