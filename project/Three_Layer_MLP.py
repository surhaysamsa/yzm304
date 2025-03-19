# Load and preprocess the dataset
import pandas as pd
import numpy as np

# Load the dataset from a CSV file
# Shuffle the data to ensure randomness
# Reset index to maintain continuity
np.random.seed(42)
df = pd.read_csv("BankNote_Authentication.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)

# Manually split the data into training and testing sets
# 80% of the data is used for training and 20% for testing
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define a class for the 3-layer neural network model
class ThreeLayerNeuralNetwork:
    def __init__(self, n_x, n_h1, n_h2, n_y):
        # Initialize the network parameters
        self.parameters = self.initialize_parameters(n_x, n_h1, n_h2, n_y)

    def initialize_parameters(self, n_x, n_h1, n_h2, n_y):
        # Initialize weights and biases for each layer
        W1 = np.random.randn(n_h1, n_x) * 0.01
        b1 = np.zeros((n_h1, 1))
        W2 = np.random.randn(n_h2, n_h1) * 0.01
        b2 = np.zeros((n_h2, 1))
        W3 = np.random.randn(n_y, n_h2) * 0.01
        b3 = np.zeros((n_y, 1))
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    def forward_propagation(self, X, activation='tanh'):
        # Perform forward propagation through the network
        W1, b1, W2, b2, W3, b3 = self.parameters.values()
        Z1 = np.dot(W1, X.T) + b1 
        A1 = np.tanh(Z1) if activation == 'tanh' else np.maximum(0, Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = np.tanh(Z2) if activation == 'tanh' else np.maximum(0, Z2)
        Z3 = np.dot(W3, A2) + b3
        A3 = 1 / (1 + np.exp(-Z3))
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
        return A3, cache

    def compute_cost(self, A3, Y):
        # Calculate the cross-entropy cost
        m = A3.shape[1]
        cost = - (np.dot(np.log(A3), Y) + np.dot(np.log(1 - A3), (1 - Y))) / m
        cost = float(np.squeeze(cost))
        return cost

    def backpropagation(self, X, Y, cache):
        # Compute gradients for backpropagation
        m = X.shape[0]
        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]
        W3 = self.parameters["W3"]
        A1 = cache["A1"]
        A2 = cache["A2"]
        A3 = cache["A3"]
        dZ3 = A3.T - Y
        dW3 = np.dot(dZ3.T, A2.T) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        dZ2 = np.dot(dZ3, W3) * (1 - np.power(A2, 2)).T
        dW2 = np.dot(dZ2.T, A1.T) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, W2) * (1 - np.power(A1, 2)).T
        dW1 = np.dot(dZ1.T, X) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        grads = {"dW1": dW1, "dW2": dW2, "dW3": dW3, "db1": db1, "db2": db2, "db3": db3}
        return grads

    def update_parameters(self, grads, learning_rate=0.01):
        # Update the network parameters using the computed gradients
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        W3 = self.parameters["W3"]
        b3 = self.parameters["b3"]
        W1 -= learning_rate * grads["dW1"]
        b1 -= learning_rate * grads["db1"].T
        W2 -= learning_rate * grads["dW2"]
        b2 -= learning_rate * grads["db2"].T
        W3 -= learning_rate * grads["dW3"]
        b3 -= learning_rate * grads["db3"].T
        self.parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    def nn_model(self, X, Y, n_steps=1000, print_cost=True):
        # Train the neural network model
        for i in range(n_steps):
            A3, cache = self.forward_propagation(X)
            cost = self.compute_cost(A3, Y)
            grads = self.backpropagation(X, Y, cache)
            self.update_parameters(grads)
            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X):
        # Make predictions using the trained model
        A3, _ = self.forward_propagation(X)
        return A3 > 0.5

# Example usage
n_x = X_train.shape[1]
n_h1 = 6
n_h2 = 6
n_y = 1
model = ThreeLayerNeuralNetwork(n_x, n_h1, n_h2, n_y)
model.nn_model(X_train, y_train, n_steps=1000)
predictions = model.predict(X_test)

# Calculate accuracy manually
correct_predictions = np.sum(predictions.flatten() == y_test.flatten())
accuracy = correct_predictions / len(y_test)
print("Accuracy:", accuracy)

# Calculate confusion matrix manually
true_positive = np.sum((predictions == 1) & (y_test == 1))
false_positive = np.sum((predictions == 1) & (y_test == 0))
true_negative = np.sum((predictions == 0) & (y_test == 0))
false_negative = np.sum((predictions == 0) & (y_test == 1))
confusion_matrix = np.array([[true_positive, false_positive], [false_negative, true_negative]])
print("Confusion Matrix:", confusion_matrix)
