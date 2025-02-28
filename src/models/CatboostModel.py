import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Standardize the features (recommended for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to plot feature importance (for CatBoost)
def plot_feature_importance(model, feature_names, model_name):
    feature_importance = model.get_feature_importance()
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'Feature Importance ({model_name})')
    plt.show()

# === CatBoost ===
print("\n=== CatBoost ===")

# Hyperparameter tuning for CatBoost
param_grid_catboost = {
    'iterations': [100, 200, 300],       # Number of boosting iterations
    'learning_rate': [0.01, 0.1, 0.2],   # Learning rate
    'depth': [4, 6, 8],                  # Depth of trees
    'l2_leaf_reg': [1, 3, 5],            # L2 regularization
    'border_count': [32, 64, 128],       # Number of splits for numerical features
    'random_strength': [0.1, 0.5, 1]     # Randomness in splits
}

# Perform Grid Search
catboost = CatBoostClassifier(random_state=0, verbose=0)
grid_search_catboost = GridSearchCV(catboost, param_grid_catboost, scoring='accuracy', cv=5, n_jobs=-1)
grid_search_catboost.fit(X_train, y_train)

# Get the best parameters
best_params_catboost = grid_search_catboost.best_params_
print("Best Parameters for CatBoost:", best_params_catboost)

# Train the model with the best parameters
best_catboost = CatBoostClassifier(**best_params_catboost, random_state=0, verbose=0)
best_catboost.fit(X_train, y_train)

# Evaluate the tuned CatBoost model
catboost_accuracy = evaluate_model(best_catboost, X_test, y_test, "Tuned CatBoost")
plot_confusion_matrix(y_test, best_catboost.predict(X_test), "Tuned CatBoost")

# Plot feature importance for CatBoost
plot_feature_importance(best_catboost, df.columns[0:9], "CatBoost")

# Save the best CatBoost model
joblib.dump(best_catboost, 'C:/Users/mhizg/Downloads/Best_CatBoost.pkl')
print("Best CatBoost model saved as Best_CatBoost.pkl")

# === Model Comparison ===
results = {
    'Model': ['CatBoost'],
    'Accuracy': [catboost_accuracy]
}

results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df)