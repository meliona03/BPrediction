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

# Predictions for training and testing data
y_train_pred = model.predict(trainX)
y_test_pred = model.predict(testX)

# Calculate accuracy
train_accuracy = accuracy_score(trainY, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

test_accuracy = accuracy_score(testY, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Predictions for training and testing data
y_train_pred = model.predict(trainX)
y_test_pred = model.predict(testX)

# Calculate accuracy
train_accuracy = accuracy_score(trainY, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

test_accuracy = accuracy_score(testY, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot actual vs predicted for training data
plt.figure(figsize=(12, 6))
plt.scatter(range(len(trainY)), trainY, color='green', label='Actual Labels', alpha=0.5)
plt.scatter(range(len(trainY)), y_train_pred, color='blue', label='Predicted Labels', alpha=0.5)
plt.title('Training Data: Actual vs Predicted Labels')
plt.xlabel('Sample Index')
plt.ylabel('Label (0 or 1)')
plt.legend()
plt.grid(True)
plt.show()

# Plot actual vs predicted for test data
plt.figure(figsize=(12, 6))
plt.scatter(range(len(testY)), testY, color='green', label='Actual Labels', alpha=0.5)
plt.scatter(range(len(testY)), y_test_pred, color='blue', label='Predicted Labels', alpha=0.5)
plt.title('Test Data: Actual vs Predicted Labels')
plt.xlabel('Sample Index')
plt.ylabel('Label (0 or 1)')
plt.legend()
plt.grid(True)
plt.show()

