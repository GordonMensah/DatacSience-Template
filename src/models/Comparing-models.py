import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
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

# Standardize the features (recommended for Logistic Regression and SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to evaluate a model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)

# Train and evaluate Logistic Regression
print("=== Logistic Regression ===")
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
log_reg_accuracy = evaluate_model(log_reg, X_test, y_test)

# Train and evaluate Random Forest
print("\n=== Random Forest ===")
rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(X_train, y_train)
rf_accuracy = evaluate_model(rf_classifier, X_test, y_test)

# Train and evaluate SVM
print("\n=== Support Vector Machine (SVM) ===")
svm_classifier = SVC(random_state=0, probability=True)
svm_classifier.fit(X_train, y_train)
svm_accuracy = evaluate_model(svm_classifier, X_test, y_test)

# Train and evaluate CatBoost
print("\n=== CatBoost ===")
catboost_classifier = CatBoostClassifier(random_state=0, verbose=0)
catboost_classifier.fit(X_train, y_train)
catboost_accuracy = evaluate_model(catboost_classifier, X_test, y_test)

# Compare model performance
results = {
    'Model': ['Logistic Regression', 'Random Forest', 'SVM', 'CatBoost'],
    'Accuracy': [log_reg_accuracy, rf_accuracy, svm_accuracy, catboost_accuracy]
}

results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df)

# Save the best model (e.g., CatBoost)
joblib.dump(catboost_classifier, 'C:/Users/mhizg/Downloads/Best_Model.pkl')
print("\nBest model saved as Best_Model.pkl")

# Visualize confusion matrix for the best model (e.g., CatBoost)
cm = confusion_matrix(y_test, catboost_classifier.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (CatBoost)')
plt.show()