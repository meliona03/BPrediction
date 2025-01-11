# Logistic regression.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import numpy as np
import joblib

# Load the dataset
dataframe = pd.read_csv('data1.csv')
print(dataframe.head(5))

# Data Preprocessing
df = dataframe.copy()
df.dropna(inplace=True, axis=1)
print(df.isnull().sum())
df.drop('id', axis=1, inplace=True)

# Encode diagnosis as 1 (Malignant) and 0 (Benign)
df['diagnosis'] = [1 if value == 'M' else 0 for value in df['diagnosis']]

# Display the count of malignant and benign cases
df['diagnosis'].value_counts().plot(kind='bar')
plt.title('Diagnosis Distribution')
plt.show()

# Features and labels
x = df.drop(df.columns[0], axis=1)  # Drop diagnosis column to keep features
y = df['diagnosis']

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train/test split
trainX, testX, trainY, testY = train_test_split(x_scaled, y, test_size=0.3, random_state=43)
print(f"Feature columns used during training: {list(x.columns)}")

# Logistic Regression Model (from sklearn)
model = LogisticRegression(random_state=43)
model.fit(trainX, trainY)  # Train the logistic regression model

# Save the model and scaler
joblib.dump(model, 'cancer_model.pkl')  # Save the trained model
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

# Plot training loss curve (dummy since sklearn does not have an epoch loss curve)
print("Model trained with Logistic Regression")

# Prediction and Accuracy
y_train_pred = model.predict(trainX)
y_test_pred = model.predict(testX)

train_accuracy = accuracy_score(trainY, y_train_pred)
test_accuracy = accuracy_score(testY, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

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

# Plot actual vs predicted for testing data
plt.figure(figsize=(12, 6))
plt.scatter(range(len(testY)), testY, color='green', label='Actual Labels', alpha=0.5)
plt.scatter(range(len(testY)), y_test_pred, color='blue', label='Predicted Labels', alpha=0.5)
plt.title('Test Data: Actual vs Predicted Labels')
plt.xlabel('Sample Index')
plt.ylabel('Label (0 or 1)')
plt.legend()
plt.grid(True)
plt.show()

y_test_pred = model.predict(testX)

# Compute confusion matrix
conf_matrix = confusion_matrix(testY, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Test Data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Print classification report for precision, recall, and F1-score
print("Classification Report:")
print(classification_report(testY, y_test_pred))
