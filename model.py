# model.py
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        # use in_channels here
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # For 28x28 input after one pooling -> 14x14; for 32x32 input after one pooling -> 16x16
        # Use an adaptive flatten approach at runtime: compute feature map size dynamically in forward
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # default for MNIST
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        # If input size is 32x32 (CIFAR/SVHN), the pool will make it 16x16 -> need different fc size.
        # handle dynamic flattening:
        x = x.view(x.size(0), -1)
        # if shape mismatch for fc1, recreate a linear on the fly (simple hack for small projects)
        if x.size(1) != self.fc1.in_features:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
