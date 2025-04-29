import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from models import LeNet5, LeNet5_DropoutBN, get_vgg11_for_mnist
from train_utils import train_model, test_model, extract_features
from hybrid_ml import run_svm, run_rf

def main():
    # Veri yükleme
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Kullanılan cihaz:', device)
    results = {}

    # Model 1: LeNet-5
    print('\n--- Model 1: LeNet-5 ---')
    model1 = LeNet5()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model1.parameters(), lr=0.001)
    train_model(model1, train_loader, criterion, optimizer, device, epochs=2)
    acc1 = test_model(model1, test_loader, device)
    results['LeNet5'] = acc1

    # Model 2: Dropout ve BatchNorm
    print('\n--- Model 2: LeNet-5 + Dropout/BatchNorm ---')
    model2 = LeNet5_DropoutBN()
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    train_model(model2, train_loader, criterion, optimizer2, device, epochs=2)
    acc2 = test_model(model2, test_loader, device)
    results['LeNet5_DropoutBN'] = acc2

    # Model 3: VGG11
    print('\n--- Model 3: VGG11 ---')
    model3 = get_vgg11_for_mnist()
    optimizer3 = optim.Adam(model3.parameters(), lr=0.001)
    train_model(model3, train_loader, criterion, optimizer3, device, epochs=2)
    acc3 = test_model(model3, test_loader, device)
    results['VGG11'] = acc3

    # Model 4: Hibrit (CNN + SVM/RF)
    print('\n--- Model 4: LeNet-5 Özellik + SVM/RF ---')
    features_train, labels_train = extract_features(model1, train_loader, device)
    features_test, labels_test = extract_features(model1, test_loader, device)
    np.save('features_train.npy', features_train)
    np.save('labels_train.npy', labels_train)
    np.save('features_test.npy', features_test)
    np.save('labels_test.npy', labels_test)
    acc_svm = run_svm(features_train, labels_train, features_test, labels_test)
    results['LeNet5+SVM'] = acc_svm
    acc_rf = run_rf(features_train, labels_train, features_test, labels_test)
    results['LeNet5+RF'] = acc_rf

    # Sonuçları yazdır
    print('\n--- Sonuçlar Karşılaştırması ---')
    for key, val in results.items():
        print(f'{key}: {val:.2f}%')

if __name__ == '__main__':
    main()
