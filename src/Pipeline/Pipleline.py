import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('C:/Users/mhizg/Downloads/a1_Dataset_10Percent.xlsx')

# Data Preprocessing
## Fill missing values
df['DemAffl'] = df['DemAffl'].fillna(df['DemAffl'].mode()[0])
df['DemAge'] = df['DemAge'].fillna(df['DemAge'].mode()[0])
df['DemClusterGroup'] = df['DemClusterGroup'].fillna(df['DemClusterGroup'].mode()[0])
df['DemGender'] = df['DemGender'].fillna(df['DemGender'].mode()[0])
df['DemReg'] = df['DemReg'].fillna(df['DemReg'].mode()[0])
df['DemTVReg'] = df['DemTVReg'].fillna(df['DemTVReg'].mode()[0])
df['LoyalTime'] = df['LoyalTime'].fillna(df['LoyalTime'].mean())

## Drop the 'ID' column
df.drop('ID', axis=1, inplace=True)

## Encode categorical variables
categorical_cols = ['DemGender', 'DemReg', 'DemTVReg', 'LoyalClass', 'DemClusterGroup']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Save the encoders for future use

# Split the data into features (X) and target (y)
X_fresh = df.iloc[:, 0:9].values  # Features (columns 0 to 8)
import joblib
classifier= joblib.load('C:/Users/mhizg/Downloads/Best_CatBoost.pkl')
y_pred=classifier.predict(X_fresh)
print(y_pred)

#Predictions
predictions=classifier.predict_proba(X_fresh)
predictions

#writing model output file
df_prediction_prob =pd.DataFrame(predictions, columns=['prob_0','prob_1'])
df=pd.concat([df,df_prediction_prob], axis=1)
df.to_csv('Buyproba')
df.head()