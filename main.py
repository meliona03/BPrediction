import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the dataset
dataframe = pd.read_csv('data.csv')
print(dataframe.head(5))

df = dataframe.copy()
df.dropna(inplace=True, axis=1)
print(df.isnull().sum())
df.drop('id', axis=1, inplace=True)

# Convert diagnosis to 1 (M) or 0 (B)
df['diagnosis'] = [1 if value == 'M' else 0 for value in df['diagnosis']]

# Plot the diagnosis distribution
df['diagnosis'].value_counts().plot(kind='bar')
plt.title("Diagnosis Distribution")
plt.xlabel("Diagnosis (0: Benign, 1: Malignant)")
plt.ylabel("Count")
plt.show()

# Prepare features and target
x = df.drop(df.columns[1], axis=1)  # Drop the first column (index 0)
y = df['diagnosis']

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split the data into training and testing sets
trainX, testX, trainY, testY = train_test_split(x_scaled, y, test_size=0.3, random_state=43)

# Create Logistic Regression model
model = LogisticRegression()
model.fit(trainX, trainY)

