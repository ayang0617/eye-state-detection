#
import torch
import cv2

class Lenet(torch.nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)  #46*46*32
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc1 = torch.nn.Linear(10*10*64, 1024)
        self.fc2 = torch.nn.Linear(1024, 2)
        self.leakyrelu1 = torch.nn.LeakyReLU()
        self.softmax1 = torch.nn.Softmax(dim=1)  #对第一维softmax

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.leakyrelu1(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax1(x)
        return x

