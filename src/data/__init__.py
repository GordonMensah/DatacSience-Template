## Importing the necessary libraries
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

## Function to load the dataset
df = pd.read_excel('C:/Users/mhizg/Downloads/a1_Dataset_10Percent.xlsx')
print(df.head())

# Data Preprocessing
df.shape
df.head()

## Checking for missing values
df.isnull().sum()
df.info()

## filling missing values
df['DemAffl'] = df['DemAffl'].fillna(df['DemAffl'].mode()[0])
df['DemAge'] = df['DemAge'].fillna(df['DemAge'].mode()[0])
df['DemClusterGroup'] = df['DemClusterGroup'].fillna(df['DemClusterGroup'].mode()[0])
df['DemGender'] = df['DemGender'].fillna(df['DemGender'].mode()[0])
df['DemReg'] = df['DemReg'].fillna(df['DemReg'].mode()[0])
df['DemTVReg'] = df['DemTVReg'].fillna(df['DemTVReg'].mode()[0])
df['LoyalTime'] = df['LoyalTime'].fillna(df['LoyalTime'].mean())

## Checking for missing values
df.isnull().sum()

## Recheck df
df.head()   

## Dropping the 'ID' column
df.drop('ID', axis=1, inplace=True)


df.to_csv('C:/Users/mhizg/Downloads/Processed_df', index=False)
