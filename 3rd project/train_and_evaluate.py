import numpy as np
import matplotlib.pyplot as plt
from data import train_data, test_data
from utils import preprocess_data, preprocess_test_data
from rnn_scratch import RNN
from ml_library_model import TorchRNN
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

# Preprocess data
X_train, y_train, vectorizer = preprocess_data(train_data)
X_test, y_test = preprocess_test_data(test_data, vectorizer)

# RNN from scratch
input_size = X_train.shape[1]
hidden_size = 8
output_size = 1
rnn = RNN(input_size, hidden_size, output_size)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

learning_rate = 0.01
epochs = 30
losses = []
for epoch in range(epochs):
    epoch_loss = 0
    for x, y in zip(X_train, y_train):
        y_hat, _ = rnn.forward(x)
        y_pred = sigmoid(y_hat[0][0])
        loss = binary_cross_entropy(np.array([y_pred]), np.array([y]))
        epoch_loss += loss
    losses.append(epoch_loss / len(X_train))
    print(f"[Scratch RNN] Epoch {epoch+1}, Loss: {losses[-1]:.4f}")
plt.plot(losses)
plt.title('RNN Scratch Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('rnn_scratch_loss.png')
plt.close()

# RNN from scratch Evaluation
rnn_preds = [rnn.predict(x) for x in X_test]
rnn_acc = accuracy_score(y_test, rnn_preds)
rnn_cm = confusion_matrix(y_test, rnn_preds)

# PyTorch RNN Model
input_size = X_train.shape[1]
seq_len = 1  # bag-of-words as one-timestep sequence
hidden_size = 8
output_size = 1
pt_rnn = TorchRNN(input_size, hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(pt_rnn.parameters(), lr=0.01)

# Prepare data for PyTorch (batch, seq_len, input_size)
X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (batch, 1, input_size)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (batch, 1)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

pt_losses = []
for epoch in range(epochs):
    pt_rnn.train()
    optimizer.zero_grad()
    outputs = pt_rnn(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
    pt_losses.append(loss.item())
    print(f"[PyTorch RNN] Epoch {epoch+1}, Loss: {loss.item():.4f}")

plt.plot(pt_losses)
plt.title('PyTorch RNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('pytorch_rnn_loss.png')
plt.close()

# PyTorch RNN Evaluation
pt_rnn.eval()
with torch.no_grad():
    logits = pt_rnn(X_test_torch)
    preds = (torch.sigmoid(logits) > 0.5).int().squeeze().numpy()
pt_acc = accuracy_score(y_test, preds)
pt_cm = confusion_matrix(y_test, preds)

# confusion matrices
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(rnn_cm, cmap='Blues')
plt.title('Scratch RNN Confusion Matrix')
plt.subplot(1,3,2)
plt.imshow(pt_cm, cmap='Oranges')
plt.title('PyTorch RNN Confusion Matrix')
plt.subplot(1,3,3)
plt.axis('off')
plt.savefig('confusion_matrices.png')
plt.close()

print(f"Scratch RNN Accuracy: {rnn_acc:.2f}")
print(f"PyTorch RNN Accuracy: {pt_acc:.2f}")
