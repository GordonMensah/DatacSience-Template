## Importing the necessary libraries
import pandas as pd
import numpy as np

df=pd.read_csv("C:/Users/mhizg/Downloads/Processed_df")

## Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['DemGender'] = label.fit_transform(df['DemGender'].astype(str))
integer_mapping1 = {l: i for i, l in enumerate(label.classes_)}
print(integer_mapping1)

df['DemReg'] = label.fit_transform(df['DemReg'].astype(str))
integer_mapping2 = {l: i for i, l in enumerate(label.classes_)}
print(integer_mapping2)

df['DemTVReg'] = label.fit_transform(df['DemTVReg'].astype(str))
integer_mapping3 = {l: i for i, l in enumerate(label.classes_)}
print(integer_mapping3)

df['LoyalClass'] = label.fit_transform(df['LoyalClass'].astype(str))
integer_mapping4 = {l: i for i, l in enumerate(label.classes_)}
print(integer_mapping4)

df['DemClusterGroup'] = label.fit_transform(df['DemClusterGroup'].astype(str))
integer_mapping5 = {l: i for i, l in enumerate(label.classes_)}
print(integer_mapping5)

#save mapping to csv
df1 = pd.DataFrame(integer_mapping1.items(), columns=['DemGender', 'Encoded'])
df2 = pd.DataFrame(integer_mapping2.items(), columns=['DemReg', 'Encoded'])
df3 = pd.DataFrame(integer_mapping3.items(), columns=['DemTVReg', 'Encoded'])
df4 = pd.DataFrame(integer_mapping4.items(), columns=['LoyalClass', 'Encoded'])
df5 = pd.DataFrame(integer_mapping5.items(), columns=['DemClusterGroup', 'Encoded'])
df1.to_csv('DemGender.csv', index=False)
df2.to_csv('DemReg.csv', index=False)
df3.to_csv('DemTVReg.csv', index=False)
df4.to_csv('LoyalClass.csv', index=False)
df5.to_csv('DemClusterGroup.csv', index=False)

#merge the dataframes
integer_mapping= pd.concat([df1, df2, df3, df4, df5], axis=1)
integer_mapping.to_csv('C:/Users/mhizg/Downloads/integer_mapping.csv', index=False)

df.head()

## Check for multicollinearity
import seaborn as sns
import matplotlib.pyplot as plt
corr = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)


## save the processed data
df.to_csv('C:/Users/mhizg/Downloads/Processed_features.csv', index=False)