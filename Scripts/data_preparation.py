import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data_path = "./Data/AA_Churn_Data.csv"
df = pd.read_csv(data_path) 

print("Missing values per column:\n", df.isnull().sum())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Verify no missing values remain
print("Missing values after cleaning:\n", df.isnull().sum())

# binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
# df[binary_columns] = df[binary_columns].replace({'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0, '1': 1, '0': 0})

df = pd.get_dummies(df, columns=[
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
], drop_first=True)

print(df.head())

print(df.describe())

sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.savefig('./outputs/visualizations/churn_distribution.png')
plt.show()
