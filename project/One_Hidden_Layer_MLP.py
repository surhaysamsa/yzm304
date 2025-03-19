# Load and preprocess the dataset
import pandas as pd
# Load the dataset from a CSV file
# Shuffle the data to ensure randomness
# Reset index to maintain continuity
df = pd.read_csv("BankNote_Authentication.csv")
df = df.sample(frac = 1, random_state = 42).reset_index(drop= True)

# Separate features and labels
X, y = df.iloc[:, :-1], df.iloc[:, -1]
# Convert to numpy arrays for compatibility with numerical operations
X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# Define the neural network structure
import numpy as np
# Initialize model parameters with small random values
# This helps in breaking symmetry and ensures convergence
def initialize_parameters(n_x, n_h, n_y = 1):
    np.random.seed(42)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2
    }
    return parameters

# Perform forward propagation to compute predictions
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X.T) + b1 
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2 
    A2 = sigmoid(Z2) 
    
    cache = {
        "Z1" : Z1,
        "A1" : A1,
        "Z2" : Z2,
        "A2" : A2
    }
    
    return A2, cache

def sigmoid(Z):
    # Compute the sigmoid activation function
    return 1 / (1 + np.exp(-1 * Z))

## Compute Cost
def compute_cost(A2, Y):
    # Calculate the cross-entropy cost
    m = A2.shape[1]
    cost = - (np.dot(np.log(A2), Y) + np.dot(np.log(1 - A2), (1 - Y))) / m
    cost = float(np.squeeze(cost))
    return cost

## Backpropagation
def backpropagation(X, Y, cache, parameters):
    # Compute gradients of the cost with respect to model parameters
    m = X.shape[0]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    # (1, 1097) (1097, 5) 
    dZ2 = A2.T - Y # (1097, 1) 
    dW2 = np.dot(dZ2.T, A1.T) / m # (1, 1097) (1097, 5) -> (1, 5)
    db2 = np.sum(dZ2, axis = 0, keepdims = True) / m 
    dZ1 = np.dot(dZ2, W2) * (1 - np.power(A1, 2)).T # (1097, 1) (1, 5) ... -> (1097, 5) 
    dW1 = np.dot(dZ1.T, X) / m # (5, 1097) 
    db1 = np.sum(dZ1, axis = 0, keepdims = True) / m # (1, 5)
    
    grads = {
        "dW1" : dW1,
        "dW2" : dW2,
        "db1" : db1,
        "db2" : db2
    }
    
    return grads

## Update Propagation
def update_parameters(parameters, grads, learning_rate = 0.01):
    # Update model parameters using gradient descent
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1.T
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2
    }
    
    return parameters

## Integration All Items
def nn_model(X, Y, n_x, n_h, n_y, n_steps = 1000, print_cost = True):
    # Train the neural network model
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(0, n_steps):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backpropagation(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 1000 == 0:
            print("cost %i %f" %(i,cost))
    
    return parameters

## Model Training and Evaluation
parameters = nn_model(X_train, y_train, X_train.shape[1], n_h = 6, n_y = 1, n_steps = 1000)

def predict(parameters, X):
    # Make predictions using the trained model
    A2, cache = forward_propagation(X, parameters)
    predicts = A2 > 0.5
    return predicts

predicts = predict(parameters, X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
y_true = y_test.flatten()  
y_pred = predicts.flatten() 
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary') 
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_true, y_pred)) 

parameters_n_h = [i for i in range(3, 11)]
parameters_n_steps = [i for i in range(100, 1100, 100)]
for n_h in parameters_n_h:
    for n_step  in parameters_n_steps:
        parameters = nn_model(X_train, y_train, X_train.shape[1], n_h = n_h, n_y = 1, n_steps = n_step, print_cost = False)
        predicts = predict(parameters, X_test)
        acc = accuracy_score(y_test.flatten(), predicts.flatten())
        print(f"n_h: {n_h}, n_step: {n_step}, acc: {acc}")