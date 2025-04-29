import torch
import numpy as np

def train_model(model, loader, criterion, optimizer, device, epochs=2):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(loader):.4f}")

def test_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f'Test Accuracy: {acc:.2f}%')
    return acc

def extract_features(model, loader, device):
    model.to(device)
    model.eval()
    features, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # LeNet5 forward akışı ile uyumlu özellik çıkarımı
            x = model.c1(images)
            x_skip = model.c2_1(x)
            x = model.c2_2(x)
            x = x + x_skip
            x = model.c3(x)
            x = x.view(x.size(0), -1)  # flatten
            features.append(x.cpu().numpy())
            labels_all.append(labels.numpy())
    features = np.concatenate(features, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return features, labels_all
