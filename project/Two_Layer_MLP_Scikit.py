# Load and preprocess the dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset from a CSV file
# Shuffle the data to ensure randomness
# Reset index to maintain continuity
np.random.seed(42)
df = pd.read_csv("BankNote_Authentication.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X = X.to_numpy()
y = y.to_numpy()

# Split the dataset into training and testing sets
# Use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define and train the MLPClassifier
# Using a single hidden layer with 6 neurons and tanh activation
mlp = MLPClassifier(hidden_layer_sizes=(6,), activation='tanh', solver='sgd', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Evaluate the model
# Predict the test set results
predictions = mlp.predict(X_test)
# Calculate accuracy and print evaluation metrics
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
