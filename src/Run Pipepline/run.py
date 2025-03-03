import joblib
import pandas as pd

# Load the saved pipeline and label encoders
pipeline = joblib.load('C:/Users/mhizg/Downloads/catboost_pipeline.pkl')
label_encoders = joblib.load('C:/Users/mhizg/Downloads/label_encoders.pkl')  # Optional, if categorical encoding was used

# Load the new data
new_data = pd.read_excel("C:/Users/mhizg/Downloads/a2_Dataset_90Percent.xlsx")

# Ensure the new data has the same columns as the training data
# Drop any extra columns or add missing columns with default values
required_columns = pipeline.named_steps['preprocessor'].transformers_[0][2] + \
                   pipeline.named_steps['preprocessor'].transformers_[1][2]
new_data = new_data[required_columns]

# Encode categorical features in new data (if label encoders were used)
for col, le in label_encoders.items():
    if col in new_data.columns:
        new_data[col] = le.transform(new_data[col].astype(str))

# Make predictions
predictions = pipeline.predict(new_data)
predicted_probabilities = pipeline.predict_proba(new_data)  # Optional: Get probabilities

# Add predictions to the new data
new_data['Predicted_Class'] = predictions
new_data['Predicted_Probability_0'] = predicted_probabilities[:, 0]  # Probability of class 0
new_data['Predicted_Probability_1'] = predicted_probabilities[:, 1]  # Probability of class 1

# Save the new data with predictions
new_data.to_csv('C:/Users/mhizg/Downloads/New_Data_With_Predictions.csv', index=False)
print("Predictions saved to New_Data_With_Predictions.csv")

# Display the first few rows of the new data with predictions
print(new_data.head())