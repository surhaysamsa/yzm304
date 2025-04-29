import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models

# LeNet-5
class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))
    def forward(self, img):
        return self.c1(img)

class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()
        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))
    def forward(self, img):
        return self.c2(img)

class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()
        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))
    def forward(self, img):
        return self.c3(img)

class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()
        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))
    def forward(self, img):
        return self.f4(img)

class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()
        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))
    def forward(self, img):
        return self.f5(img)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = C1()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5()
    def forward(self, img):
        output = self.c1(img)
        x = self.c2_1(output)
        output = self.c2_2(output)
        output += x
        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output

# LeNet-5 + Dropout ve BatchNorm
class LeNet5_DropoutBN(nn.Module):
    def __init__(self):
        super(LeNet5_DropoutBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# VGG11'i MNIST için adapte et
class VGG11_MNIST(nn.Module):
    def __init__(self):
        super(VGG11_MNIST, self).__init__()
        self.model = models.vgg11(pretrained=False)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.model.classifier[6] = nn.Linear(4096, 10)
    def forward(self, x):
        return self.model(x)

def get_vgg11_for_mnist():
    return VGG11_MNIST()
