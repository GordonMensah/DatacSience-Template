import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# Standardize the features (optional but recommended for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Save the trained model
joblib.dump(classifier, 'C:/Users/mhizg/Downloads/LogisticRegression.pkl')
print("Model has been trained and saved as LogisticRegression.pkl")

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Get predicted probabilities
y_pred_prob = classifier.predict_proba(X_test)

# Save predictions to a CSV file
df_pred_prob = pd.DataFrame(y_pred_prob, columns=['prob_0', 'prob_1'])
df_test_df = pd.DataFrame(y_test, columns=['Actual Outcome'])
df_x_test = pd.DataFrame(X_test, columns=[f'Feature_{i}' for i in range(X_test.shape[1])])

# Combine all data into a single DataFrame
dfx = pd.concat([df_x_test, df_test_df, df_pred_prob], axis=1)
dfx.to_csv('C:/Users/mhizg/Downloads/Modeloutput_10Percent.csv', index=False)
print("Predictions saved to Modeloutput_10Percent.csv")
