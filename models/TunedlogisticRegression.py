import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/mhizg/Downloads/Processed_features.csv")

# Split the data into features (X) and target (y)
X = df.iloc[:, 0:9].values  # Features (columns 0 to 8)
y = df.iloc[:, 9].values    # Target (column 9)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features (recommended for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to evaluate a model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"=== {model_name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    return accuracy

# Function to visualize confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.show()

# Train and evaluate the base Logistic Regression model
print("\n=== Base Logistic Regression ===")
base_log_reg = LogisticRegression(random_state=0)
base_log_reg.fit(X_train, y_train)
base_log_reg_accuracy = evaluate_model(base_log_reg, X_test, y_test, "Base Logistic Regression")
plot_confusion_matrix(y_test, base_log_reg.predict(X_test), "Base Logistic Regression")

# Hyperparameter tuning for Logistic Regression
print("\n=== Tuning Logistic Regression ===")
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],               # Regularization type
    'solver': ['liblinear', 'saga'],       # Solvers that support l1 and l2
    'max_iter': [100, 200, 500],          # Maximum iterations
    'class_weight': [None, 'balanced']     # Handle class imbalance
}

# Perform Grid Search
grid_search = GridSearchCV(LogisticRegression(random_state=0), param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters for Logistic Regression:", best_params)

# Train the model with the best parameters
best_log_reg = LogisticRegression(**best_params, random_state=0)
best_log_reg.fit(X_train, y_train)

# Evaluate the tuned Logistic Regression model
tuned_log_reg_accuracy = evaluate_model(best_log_reg, X_test, y_test, "Tuned Logistic Regression")
plot_confusion_matrix(y_test, best_log_reg.predict(X_test), "Tuned Logistic Regression")

# Compare base and tuned Logistic Regression models
results = {
    'Model': ['Base Logistic Regression', 'Tuned Logistic Regression'],
    'Accuracy': [base_log_reg_accuracy, tuned_log_reg_accuracy]
}

results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df)

# Save the best model
joblib.dump(best_log_reg, 'C:/Users/mhizg/Downloads/Best_LogisticRegression.pkl')
print("\nBest Logistic Regression model saved as Best_LogisticRegression.pkl")